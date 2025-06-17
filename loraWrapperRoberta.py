import torch 
import torch.nn as nn
from pathlib import Path
import math
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel, RobertaConfig
from torch.nn import MultiheadAttention
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
from typing import Optional, Tuple, Union, Dict

from torch.nn import functional as F

class LoraRobertaSelfAttention(RobertaSelfAttention):

    def __init__(self, r=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        d= self.all_head_size


        # We are initializing the LoRA matrices. B is initialized to zero. A is initialized to gaussian noise.

        self.lora_query_matrix_B = nn.Parameter(torch.zeros(d,r))
        self.lora_query_matrix_A = nn.Parameter(torch.rand(r,d))

        self.lora_key_matrix_B = nn.Parameter(torch.zeros(d,r))
        self.lora_key_matrix_A = nn.Parameter(torch.rand(r,d))

    
    def lora_query(self, x):

        lora_full_query_weights = torch.matmul(self.lora_query_matrix_B,self.lora_query_matrix_A)
        return self.query(x)+ F.linear(x, lora_full_query_weights)

    def lora_value(self,x):

        lora_full_value_weights = torch.matmul(self.lora_key_matrix_B,self.lora_key_matrix_A)
        return self.key(x) + F.linear(x, lora_full_value_weights)


    # This part is copied from the original RobertaSelfAttention class, 
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.lora_query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class LoraWrapperRoberta(nn.Module):

    def __init__(self,task_type, num_classes:int = None, dropout_rate=0.1, model_id:str="roberta-base",lora_rank:int=8, train_biases:bool = True, train_embedding:bool = False, train_layer_norm:bool = True):
        """
        Initializes a LoraWrapperRoberta instance, which is a wrapper around the RoBERTa model incorporating
        Low-Rank Adaptation (LoRA) to efficiently retrain the model for different NLP tasks such as GLUE benchmarks
        and SQuAD. LoRA allows for effective adaptation of large pre-trained models like RoBERTa with minimal updates.

        Parameters
        ----------
        task_type : str
            Type of task to configure the model for. Should be one of {'glue', 'squad_v1', 'squad_v2'}.
            For 'squad_v1' and 'squad', the number of classes is set to 2, and for 'squad_v2', it's set to 3,
            to accommodate the "no answer possible" scenario.
        num_classes : int, optional
            The number of classes for the classification layer on top of the RoBERTa model. The default value is
            determined by the task type if not provided. Has to be provided for the glue task.
        dropout_rate : float, default 0.1
            Dropout rate to be used in the dropout layers of the model.
        model_id : str, default "roberta-base"
            Identifier for the pre-trained RoBERTa model to be loaded.
        lora_rank : int, default 8
            Rank of the adaptation applied to the attention layers via LoRA.
        train_biases : bool, default True
            Flag indicating whether to update bias parameters during training.
        train_embedding : bool, default False
            Flag indicating whether to update embedding layer weights during training.
        train_layer_norms : bool, default True
            Flag indicating whether to update the layer norms during training. Usually this is a good idea.

        Examples
        --------
        To initialize a model for the 'glue' task type:

            model = LoraWrapperRoberta(task_type='squad_v1')
        """
        super().__init__()
        supported_task_types = ['squad']

        num_classes = 3

        self.model_id = model_id
        self.tokenizer = RobertaTokenizer.from_pretrained(model_id)
        self.model = RobertaModel.from_pretrained(model_id)

        self.model_config = self.model.config

        self.lora_rank = lora_rank
        self.train_biases = train_biases
        self.train_embeddings = train_embedding
        self.train_layer_norm = train_layer_norm


        # Output size of the base model
        d_model = self.model_config.hidden_size
        self.d_model = d_model

        self.num_classes = num_classes

        self.task_type = supported_task_types[0]

        self.fine_tune_head_norm = nn.LayerNorm(d_model)
        self.fine_tune_head_dropout = nn.Dropout(dropout_rate)
        self.fine_tune_head_classifier = nn.Linear(d_model, num_classes)

        self.replace_multihead_attention()
        self.freeze_parameters_except_lora_and_bias()

    def replace_multihead_attention(self):
        """
        Replaces the MultiheadAttention layers in the RoBERTa model with LoraRobertaSelfAttention layers.
        This allows for efficient adaptation of the model using Low-Rank Adaptation (LoRA).
        """

        self.nr_replaced_modules = 0
        self.replace_multihead_attention_recursion(self.model)
        # if verbose:
        print(f"Replaced {self.nr_replaced_modules} modules of RobertaSelfAttention with LoraRobertaSelfAttention")

       
    def replace_multihead_attention_recursion(self, model):
        


        for name, module in model.named_children():

            if isinstance(module, RobertaSelfAttention):
                self.nr_replaced_modules += 1

                new_layer = LoraRobertaSelfAttention(r=self.lora_rank, config=self.model_config)

                state_dict_old = module.state_dict()

                new_layer.load_state_dict(state_dict_old, strict=False) 

                state_dict_new = new_layer.state_dict()

                keys_old = set(state_dict_old.keys())
                keys_new = set(k for k in state_dict_new.keys() if not k.startswith("lora_"))
                assert keys_old == keys_new, f"Keys of the state dictionaries don't match (ignoring lora parameters):\n\tExpected Parameters: {keys_old}\n\tNew Parameters (w.o. LoRA): {keys_new}"

                # Replace the original layer with the new layer
                setattr(model, name, new_layer)
            else:
                # Recurse on the child modules
                self.replace_multihead_attention_recursion(module)

    def freeze_parameters_except_lora_and_bias(self):


        for name, param in self.model.named_parameters():

            if ("lora_" in name) or ("finetune_head_" in name) or (self.train_biases and "bias" in name) or (self.train_embeddings and "embeddings" in name) or (self.train_layer_norm and "layer_norm" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    
    def forward(self,x, attention_mask=None):

        output = self.model(x, attention_mask=attention_mask)

        x = output.last_hidden_state

        x = self.fine_tune_head_norm(x)
        x = self.fine_tune_head_dropout(x)
        x = self.fine_tune_head_classifier(x)

        start_logits, end_logits, na_prob_logits = x.split(1, dim=-1) # split the two output values

        # Flatten the outputs
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        na_prob_logits = na_prob_logits.squeeze(-1)

        # na_prob_logits has to be a single number, only take START token
        na_prob_logits = na_prob_logits[:, 0]

        # return the start and end logits
        return start_logits, end_logits, na_prob_logits

    def save_lora_state_dict(self, lora_filepath: Optional[Union[str, Path]] = None) -> Optional[Dict]:
        """
        Save the trainable parameters of the model into a state dict.
        If a file path is provided, it saves the state dict to that file.
        If no file path is provided, it simply returns the state dict.

        Parameters
        ----------
        lora_filepath : Union[str, Path], optional
            The file path where to save the state dict. Can be a string or a pathlib.Path. If not provided, the function
            simply returns the state dict without saving it to a file.

        Returns
        -------
        Optional[Dict]
            If no file path was provided, it returns the state dict. If a file path was provided, it returns None after saving
            the state dict to the file.
        """
        # Create a state dict of the trainable parameters
        state_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}

        # add addional parameters to state dict
        state_dict['model_id'] = self.model_id
        state_dict['task_type'] = self.task_type
        state_dict['lora_rank'] = self.lora_rank
        state_dict['num_classes'] = self.num_classes

        if lora_filepath is not None:
            # Convert string to pathlib.Path if necessary
            if isinstance(lora_filepath, str):
                lora_filepath = Path(lora_filepath)

            # Save the state dict to the specified file
            torch.save(state_dict, lora_filepath)
        else:
            # Return the state dict if no file path was provided
            return state_dict


    @staticmethod
    def load_lora_state_dict(lora_parameters: Union[str, Path, Dict] = None):
        """
        Load a state dict into the model from a specified file path or a state dict directly.
        This is a staticmethod to be used from the base clase, returning a fully initialized and LoRA loaded model.

        Parameters
        ----------
        lora_parameters : Union[str, Path, Dict]
            Either the file path to the state dict (can be a string or pathlib.Path) or the state dict itself. If a file path
            is provided, the function will load the state dict from the file. If a state dict is provided directly, the function
            will use it as is.

        Returns
        -------
        LoraWrapperRoberta object, initialized and with the LoRA weights loaded.
        """
        # Check if a filepath or state dict was provided
        if lora_parameters is not None:
            # Convert string to pathlib.Path if necessary
            if isinstance(lora_parameters, str):
                lora_parameters = Path(lora_parameters)

            # If the provided object is a Path, load the state dict from file
            if isinstance(lora_parameters, Path):
                state_dict = torch.load(lora_parameters)
            else:
                # If it's not a Path, assume it's a state dict
                state_dict = lora_parameters
        else:
            raise ValueError("No filepath or state dict provided")

        instance = LoraWrapperRoberta(task_type=state_dict['task_type'], num_classes = state_dict['num_classes'], model_id = state_dict['model_id'], lora_rank = state_dict['lora_rank'])

        # Load the state dict into the model
        instance.load_state_dict(state_dict, strict=False)

        return instance

    
