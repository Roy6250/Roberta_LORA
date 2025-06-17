import math
import time
import pathlib

import numpy as np
import pandas as pd

from typing import List, Tuple, Dict

# Huggingface imports
import transformers
import datasets

from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from tqdm.auto import trange, tqdm
import evaluate

class SquadDataLoader:

    def __init__(self, tokenizer, batch_size:int, debug=False):

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.dataset = datasets.load_dataset("squad_v2")
        # Split the train set into train and validation sets
        # ,split={"train":"train[:1000]"}

        # train_data = self.dataset["train"].select(range(0, 1000))
        # val_data = self.dataset["train"].select(range(1000, 1100))
        # self.dataset = {
        #     "train": train_data,
        #     "val": val_data
        # }
        split_dataset = self.dataset['train'].train_test_split(test_size=0.1, seed=42)
        self.dataset['train'] = split_dataset['train']        # self.dataset['train'] = split_dataset['train']
        self.dataset['val'] = split_dataset['test']
        self.metric = evaluate.load("squad_v2")
        self.debug = debug
        self.non_answerable = 0
        self.task="squad"

    @staticmethod
    def cleanse_squad_answers(text:str):

        text = text.strip()
        if text.endswith(',') or text.endswith(':'):
            text = text[:-1]

        # If the text contains an unmatched " or ' at the beginning or end, remove it
        if text.startswith('"') and text.count('"') == 1:
            text = text[1:]
        if text.endswith('"') and text.count('"') == 1:
            text = text[:-1]
        if text.startswith("'") and text.count("'") == 1:
            text = text[1:]
        if text.endswith("'") and text.count("'") == 1:
            text = text[:-1]

        # If it starts '(', remove the symbol
        if text.startswith('('):
            text = text[1:]

        if text.startswith('�'):
            text = text[1:]

        if text.endswith("%.") or text.endswith("%;") or text.endswith("%:") or text.endswith("%)"):
            text = text[:-1]

        if text.endswith("%).") or text.endswith("%);") or text.endswith("%):"):
            text = text[:-2]

        # The SQuAD text is rather inconsisteny with the dollar sign for monetary amounts -> probably best to leave is as is for now
        ## if text.startswith('$'):
        ##    text = text[1:]

        # If text has no more '(' in it after the above and it ends with ')' or ').' or '),' remove the closing parentheses expression
        if '(' not in text and (text.endswith(')') or text.endswith(').') or text.endswith('),')):
            text = text.rsplit(')', 1)[0]

        # If texts starts with '-' or '—' remove it
        if text.startswith('-') or text.startswith('—') or text.startswith('–') or text.startswith('–'):
            text = text[1:]

        return text


    def encode_squad_task(self, batch, DEBUG=False):
        """Convert SQuAD-like targets into tokenized input compatible targets.
        """
        # time.sleep(10)
        start_positions = []
        end_positions = []

        tokenized_main = self.tokenizer(text=batch[self.key_text_1], text_pair=batch[self.key_text_2], padding='longest', truncation=True)

        # List comprehension with conditional to handle unanswerable questions
        answer_texts = [item['text'][0] if item['text'] else "" for item in batch['answers']]

        # Create the non_answerable flag
        non_answerable = [1 if not item['text'] else 0 for item in batch['answers']]

        answer_starts = [item['answer_start'][0] if item['text'] else -1 for item in batch['answers']]
        answer_text_length = [len(t) for t in answer_texts]

        text_till_answer_start = [item[:start] for item, start in zip(batch['context'], answer_starts)]
        text_till_answer_end = [item[:(start+len_answer)] for item, start, len_answer in zip(batch['context'], answer_starts, answer_text_length)]
        recon_answer_text_raw = [item[start:(start+len_answer)] for item, start, len_answer in zip(batch['context'], answer_starts, answer_text_length)]

        tokenized_main = self.tokenizer(text=batch[self.key_text_1], text_pair=batch[self.key_text_2], padding='longest', truncation=True)

        for answer_orig, answer_reconstructed in zip(answer_texts, recon_answer_text_raw):
            if answer_orig != answer_reconstructed:
                print(f"No match for orig: {answer_orig} with extraction: {answer_reconstructed}")

        tok_texts_till_answer_start = self.tokenizer(text=batch[self.key_text_1], text_pair=text_till_answer_start, padding='do_not_pad', truncation=True)
        tok_texts_till_answer_end = self.tokenizer(text=batch[self.key_text_1], text_pair=text_till_answer_end, padding='do_not_pad', truncation=True)

        # -1 to account for last end token, which should not be counted
        # for start -2 to account for the added space with the first token
        start_positions = [len(p)-2 for p in tok_texts_till_answer_start['input_ids']]
        end_positions = [len(p)-1 for p in tok_texts_till_answer_end['input_ids']]

        # Overwrite if non-answerable with -1. We will ignore this index later in the loss function via ignore_index=-1
        start_positions = [-1 if na else sp for na, sp in zip(non_answerable, start_positions)]
        end_positions = [-1 if na else sp for na, sp in zip(non_answerable, end_positions)]

        # Reconstruct answer and check
        recon_answer_tokens = [ ts[start:end] for ts, start, end in zip(tokenized_main['input_ids'], start_positions, end_positions)]
        recon_answer = self.tokenizer.batch_decode(recon_answer_tokens, skip_special_tokens = True, clean_up_tokenization_spaces = True)

        recon_answer_clensed = [self.cleanse_squad_answers(t) for t in recon_answer]

        wrong_decoding_count = 0
        for answer_orig, answer_reconstructed, question, context, na in zip(answer_texts, recon_answer_clensed, batch['question'], batch['context'], non_answerable):
            # Sometimes there is a $ sign in the answers and sometimes not. Only count these mistakes if they are not
            if na==0:
                if answer_orig.replace("$", '').replace('.', '') != answer_reconstructed.replace("$", '').replace('.', ''):
                    if DEBUG:
                        print(f"No match for orig: '{answer_orig}' with decoded: '{answer_reconstructed}' for question {question}")
                        print(context)
                        print('-'*42)
                    wrong_decoding_count += 1
        if DEBUG: print(f"Number wrong SQuAD decodings: {wrong_decoding_count}")

        return {**tokenized_main,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'non_answerable': non_answerable,
            'id': batch['id'],
        }

    def epoch_iterator(self, split_type: str='train') -> Dict:
        """
        Iterate through the epoch.

        Parameters
        ----------
        split_type : str, optional
            The type of the data to be iterated over, by default 'train'.

        Yields
        -------
        Dict
            Dictionary with keys for each element
        """

    
        self.key_text_1 = "question"
        self.key_text_2 = "context"

        # Shuffle the dataset
        dataset = self.dataset[split_type].shuffle()

        if self.debug:
            dataset = dataset.filter(filter_first_100_rows, with_indices=True)
            print("WARNING: The dataset was filtered to 100 obs for debugging and testing")

        self.shuffled_dataset = dataset

        # Save the current logging level
        old_level = transformers.logging.get_verbosity()

        # Set the new logging level to prevent the weird, non-sense tokenizer warning, which is repeated for every batch
        transformers.logging.set_verbosity_error()

        # Apply the encoding function to all batches.
    
        dataset = dataset.map(self.encode_squad_task, batched=True, batch_size=self.batch_size)
        add_batch_keys = ['start_positions', 'end_positions', 'non_answerable', 'id']
    
        # Revert back to the old logging level
        transformers.logging.set_verbosity(old_level)

        # Set pytorch data loader for the batches
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', *add_batch_keys])
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        for batch in dataloader:
            # batch is a dictionary with keys corresponding to the features of the dataset
            # and values are the batched values. The main ones you wanne use are 'input_ids' and 'attention_mask'
            yield batch


    def get_nr_batches(self, split_type: str='train') -> int:
        """
        Get the number of batches.

        Parameters
        ----------
        split_type : str, optional
            The type of the data to get number of batches for, by default 'train'.

        Returns
        -------
        int
            The number of batches.
        """
        nr_obs = len(self.dataset[split_type])
        return math.ceil(nr_obs / self.batch_size)


    def get_nr_classes(self) -> int:
        """
        Get the number of classes for the current GLUE task.

        Returns
        -------
        int
            The number of classes.
        """
        if self.task == "stsb":
            return 1 # return 1 because it is actually a regression task, then the classification layer output with "1 class" can be used for regression
        elif self.task in ["squad", "squad_v1"]:
            # because it is a context mapper task it is actually None, but we will return the dimension of the linear output we require
            return 3
        elif self.task in ["squad_v2"]:
            # like squad_v1 but with one dimension more to predict whether the question is actually answerable
            return 3
        else:
            return pd.Series(self.dataset['train']['label']).nunique()




# If the script is called directly we download all datasets and test everything for roberta-large
if __name__ == "__main__":

    PRINT_EVERY_BATCH = False
    
    # Load Tokenizer
    from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel
    model_id = "roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_id)

    # for task in train_task_list:
    print(f"Testing class GlueSquadTaskLoader for task 'squad' ")
    task_loader = SquadDataLoader(tokenizer=tokenizer, batch_size=32)
    nr_batches = task_loader.get_nr_batches()
    print(f"  Nr train batches: {nr_batches}")
    nr_classes = 3
    print(f"  Nr classes: {nr_classes}")

    print("Testing loop over train dataset")

    tqdm_looper = tqdm(task_loader.epoch_iterator())
    
    print(task_loader.dataset.keys())
    for raw_batch in tqdm_looper: # , batch_target_labels
        if PRINT_EVERY_BATCH:
            print(raw_batch.keys())
            print(len(raw_batch['input_ids']))
            print(raw_batch['input_ids'].shape)
    print('-'*42)
