
import os

import sys

import math
import random
import numpy as np
from collections import Counter
import logging 

import torch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling

# Contains training set, validation set, and test set
class Corpus:
    def __init__(self, directory=None, batch_size=None, context_size=None, batches_per_buffer=None, stream=True, shuffle=True, stride=0, add_eos=False):

        fi_train = open(directory + "train.txt", "r")
        def get_training_corpus(add_eos=add_eos):
            for line in fi_train:
                if add_eos:
                    yield [line.strip() + " <eos>"]
                else:
                    yield [line]


        wt_tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        wt_tokenizer.pre_tokenizer = WhitespaceSplit() # Splits only on whitespace - assumes data are pre-processed
        wt_trainer = WordLevelTrainer(vocab_size=1000000, special_tokens=["<unk>", "<pad>"])
        wt_tokenizer.train_from_iterator(get_training_corpus(), wt_trainer)

        wrapped_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=wt_tokenizer,
                unk_token="<unk>",
                pad_token="<pad>",
                )

        self.tokenizer = wrapped_tokenizer

        self.train = LMDataLoader(filename=directory + "train.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, stream=stream, tokenizer=wrapped_tokenizer, shuffle=shuffle, stride=stride, add_eos=add_eos)
        self.valid = LMDataLoader(filename=directory + "valid.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, stream=stream, tokenizer=wrapped_tokenizer,shuffle=shuffle, stride=stride, add_eos=add_eos)
        self.test = LMDataLoader(filename=directory + "test.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, stream=stream, tokenizer=wrapped_tokenizer, shuffle=shuffle, stride=stride, add_eos=add_eos)

        # Could make the stride greater - e.g., context_size - 1
        self.stride_valid = LMDataLoader(filename=directory + "valid.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, stream=stream, tokenizer=wrapped_tokenizer, shuffle=shuffle, stride=context_size//2, add_eos=add_eos)
        self.stride_test = LMDataLoader(filename=directory + "test.txt", batch_size=batch_size, context_size=context_size, batches_per_buffer=batches_per_buffer, stream=stream, tokenizer=wrapped_tokenizer, shuffle=shuffle, stride=context_size//2, add_eos=add_eos)


# Designed to give you one batch at a time from the file
# at filename.
# For producing a standard (non-meta) dataset
class LMDataLoader:
    def __init__(self, filename=None, batch_size=None, context_size=None, batches_per_buffer=None, stream=True, shuffle=True, tokenizer=None, stride=0, add_eos=False):
       
        # Buffer of tokens drawn from the corpus
        self.current_text = []

        # Buffer of batches
        self.current_batches = []
        
        # Where in self.current_text and self.current_batches
        # we currently are
        self.text_pointer = 0
        self.batch_pointer = 0

        self.tokenizer = tokenizer
 
        # Whether to read data in a streaming way (as opposed to reading 
        # in the whole file at once)
        self.stream = stream

        # Whether to shuffle the batches within the buffer
        self.shuffle = shuffle

        # Number of un-loss-computed tokens to have at the start of each sequence
        self.stride = stride
        self.stride_on_current_text = 0

        # Whether to add an <eos> token at the end of every line
        self.add_eos = add_eos
       
        self.filename = filename
        self.fi = open(filename, "r")
        self.at_end_of_file = False

        self.context_size = context_size
        self.batch_size = batch_size
        self.batches_per_buffer = batches_per_buffer
        self.tokens_per_batch = self.context_size * self.batch_size

        # Number of tokens we need to have on hand to fill the buffer
        if self.stream:
            self.length_text_in_buffer = self.tokens_per_batch * self.batches_per_buffer
        
        else:
            # So that we read lines until the file ends
            self.length_text_in_buffer = math.inf
            self.batches_per_buffer = math.inf

        self.length = None

        # Populate the buffer
        self.reload_buffer()

    def __iter__(self):
        return self

    # Return the number of batches in the dataset
    def __len__(self):

        if self.length is not None:
            return self.length
        else:
            self.reset()
            n_batches = 0
            for _ in self:
                n_batches += 1
            self.length = n_batches
            self.reset()

            return n_batches

    # Produce the next batch
    def __next__(self):

        # We've used up all the batches in the buffer
        if self.batch_pointer == len(self.current_batches):
            self.batch_pointer = 0

            # We've reached the end of the file
            if self.at_end_of_file:

                # Indicate that we have reached the end of the file
                raise StopIteration

            # Not at the end of the file, so we simply reload the buffer in order
            # to have enough text to make a batch
            else:
                self.current_batches = []
                self.reload_buffer()

        if len(self.current_batches) == 0:
            raise StopIteration

        # Create a batch from the text in the buffer
        next_batch = self.current_batches[self.batch_pointer]
        self.batch_pointer += 1

        return next_batch
        
    # Reset (so that we can loop over the dataset again)
    def reset(self, hard_reset=False):

        if self.stream or hard_reset:
            # Need to reopen the file and start from the top
            self.fi.close()

            self.fi = open(self.filename, "r")
            self.at_end_of_file = False

            self.current_text = []

            self.current_batches = []
            self.text_pointer = 0
            self.batch_pointer = 0

            self.reload_buffer()

            # If not streaming, reloading the buffer would have used up
            # the whole file
            if not self.stream:
                self.fi.close()

        else:
            # No need to reopen the file: instead, just return
            # the pointer to the start of current_text

            if self.shuffle:
                random.shuffle(self.current_batches)

            self.text_pointer = 0
            self.batch_pointer = 0

    def current_text_to_batches(self, pad_to_complete_batches=False):
        
        if pad_to_complete_batches:
            # Make the current text long enough to evenly split into batches
            if len(self.current_text) % self.tokens_per_batch == 0:
                pass
            else:
                desired_length = self.tokens_per_batch * ((len(self.current_text) // self.tokens_per_batch) + 1)
                pad_token_length = desired_length - len(self.current_text)
                pad_tokens = [self.tokenizer.pad_token_id for _ in range(pad_token_length)]
                self.current_text = self.current_text + pad_tokens

        # Split the text into batches
        while self.text_pointer + self.tokens_per_batch <= len(self.current_text):
            batch = []
            labels = []
            for i in range(self.batch_size):
                seq = self.current_text[self.text_pointer:self.text_pointer + self.context_size]

                seq_labels = seq[:]
                for j in range(self.stride_on_current_text):
                    seq_labels[j] = self.tokenizer.pad_token_id
                
                # Don't include if it's all padding (arises from having everything past
                # the stride be padding)
                # Ignore first one, since it's never evaluated on
                if not all(x == self.tokenizer.pad_token_id for x in seq_labels[1:]): 
                    batch.append(seq)
                    labels.append(seq_labels)

                # Subtracting stride ensures that we reuse the end of the current sequence at the start of the next one
                self.text_pointer = self.text_pointer + self.context_size - self.stride
                self.stride_on_current_text = self.stride

            if len(batch) != 0:
                self.current_batches.append({"input_ids" : batch, "labels" : labels})

        self.current_text = self.current_text[self.text_pointer:]
        self.text_pointer = 0


    def reload_buffer(self):

        # First, read enough text from the file to
        # fill up the buffer
        while len(self.current_text) < self.length_text_in_buffer:
                
            # Read in 1000 lines at a time; saves a lot of time to 
            # reduce the calls to the tokenizer (tokenizing one
            # long string instead of many short strings)
            lines = ""
            for _ in range(1000):
                line = self.fi.readline()
                lines = lines + line
                if self.add_eos:
                    lines = lines.strip() + " <eos> "

                if line == "":
                    self.at_end_of_file = True
                    break

            if lines == "":
                break

            tokens = self.tokenizer.encode(lines)
            self.current_text = self.current_text + tokens

            if self.at_end_of_file:
                break

        # Then, parcel that text into batches (saving the remainder that
        # was left over for use in the next set of batches)
        self.current_text_to_batches(pad_to_complete_batches=self.at_end_of_file)

        if self.shuffle:
            random.shuffle(self.current_batches)





# A dataset of datasets
# Contains meta-training set, meta-validation set, and meta-test set
class MetaCorpus:
    def __init__(self, directory=None, context_size=None, batches_per_buffer=None, stream=True, shuffle=True):

        fi_train = open(directory + "train.txt", "r")
        def get_training_corpus():
            for line in fi_train:
                yield [" ".join(line.strip().split("\t"))]


        wt_tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        wt_tokenizer.pre_tokenizer = WhitespaceSplit() # Splits only on whitespace - assumes data are pre-processed
        wt_trainer = WordLevelTrainer(vocab_size=1000000, special_tokens=["<unk>", "<pad>"])
        wt_tokenizer.train_from_iterator(get_training_corpus(), wt_trainer)

        wrapped_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=wt_tokenizer,
                unk_token="<unk>",
                pad_token="<pad>",
                )

        self.tokenizer = wrapped_tokenizer

        self.train = LMMetaDataLoader(filename=directory + "train.txt", context_size=context_size, batches_per_buffer=batches_per_buffer, stream=stream, tokenizer=wrapped_tokenizer, shuffle=shuffle)
        self.valid = LMMetaDataLoader(filename=directory + "valid.txt", context_size=context_size, batches_per_buffer=batches_per_buffer, stream=stream, tokenizer=wrapped_tokenizer,shuffle=shuffle)
        self.test = LMMetaDataLoader(filename=directory + "test.txt", context_size=context_size, batches_per_buffer=batches_per_buffer, stream=stream, tokenizer=wrapped_tokenizer, shuffle=shuffle)






# Designed to give you one batch at a time from the file
# at filename.
class LMMetaDataLoader:
    def __init__(self, filename=None, context_size=None, batches_per_buffer=None, stream=True, shuffle=True, tokenizer=None):
        
        # Buffer of batches
        self.current_batches = []
        
        # Where in self.current_batches
        # we currently are
        self.batch_pointer = 0

        self.tokenizer = tokenizer
 
        # Whether to read data in a streaming way (as opposed to reading 
        # in the whole file at once)
        self.stream = stream

        # Whether to shuffle the batches within the buffer
        self.shuffle = shuffle

        self.filename = filename
        self.fi = open(filename, "r")
        self.at_end_of_file = False
        self.loaded_all = False

        self.context_size = context_size
        self.batches_per_buffer = batches_per_buffer

        self.length = None

        # Populate the buffer
        self.reload_buffer()

    def __iter__(self):
        return self

    # Return number of mini datasets in the larger dataset
    def __len__(self):

        if self.length is not None:
            return self.length
        else:
            self.reset()
            n_batches = 0
            for _ in self:
                n_batches += 1
            self.length = n_batches
            self.reset()

            return n_batches

    # Produce the next dataset for the next meta episode
    def __next__(self):

        # We've used up all the batches in the buffer
        if self.batch_pointer == len(self.current_batches):
            self.batch_pointer = 0

            # We've reached the end of the file
            if self.at_end_of_file:
                
                # Indicate that we have reached the end of the file
                raise StopIteration

            # Not at the end of the file, so we simply reload the buffer in order
            # to have enough text to make a batch
            else:
                self.current_batches = []
                self.reload_buffer()

        if len(self.current_batches) == 0:
            raise StopIteration

        # Create a batch from the text in the buffer
        next_batch = self.current_batches[self.batch_pointer]
        self.batch_pointer += 1

        return next_batch
        
    
    def reset(self, hard_reset=False):

        if self.stream or hard_reset:
            # Need to reopen the file and start from the top
            self.fi.close()

            self.fi = open(self.filename, "r")
            self.at_end_of_file = False

            self.current_batches = []
            self.batch_pointer = 0

            self.reload_buffer()

            # If not streaming, reloading the buffer would have used up
            # the whole file
            if not self.stream:
                self.fi.close()

        else:
            # No need to reopen the file: instead, just return
            # the pointer to the start of current_text

            if self.shuffle:
                random.shuffle(self.current_batches)

            self.batch_pointer = 0

    def text_to_seqs(self, text, pad_to_complete_batches=False):
        
        if pad_to_complete_batches:
            # Make the current text long enough to evenly split into batches
            if len(text) % self.context_size == 0:
                pass
            else:
                desired_length = self.context_size * ((len(text) // self.context_size) + 1)
                pad_token_length = desired_length - len(text)
                pad_tokens = [self.tokenizer.pad_token_id for _ in range(pad_token_length)]
                text = text + pad_tokens

        text_pointer = 0
        seqs = []
        labels = []
        # Split the text into batches
        while text_pointer + self.context_size <= len(text):
            
            seq = text[text_pointer:text_pointer + self.context_size]
            seq_labels = seq[:]
                
            # Must be longer than 1 token to be useful (since first token isn't evaluated)
            if len(seq) > 1:
                seqs.append(seq)
                labels.append(seq_labels)

            text_pointer = text_pointer + self.context_size

        return {"input_ids" : seqs, "labels" : labels}


    def reload_buffer(self):

        while len(self.current_batches) - self.batch_pointer < self.batches_per_buffer:
            self.current_batches = self.current_batches[self.batch_pointer:]
            self.batch_pointer = 0

            new_lines = []
            for line_index in range(1007):
                if self.loaded_all:
                    break

                line = self.fi.readline()

                if line == "":
                    self.at_end_of_file = True
                    break

                splits = line.strip().split("\t")
                train_split = self.tokenizer(splits[0])["input_ids"]
                test_split = self.tokenizer(splits[1])["input_ids"]

                train_seqs = self.text_to_seqs(train_split, pad_to_complete_batches=True)
                test_seqs = self.text_to_seqs(test_split, pad_to_complete_batches=True)

                batch = {"input_ids" : train_seqs["input_ids"], "labels" : train_seqs["labels"], "test_input_ids" : test_seqs["input_ids"], "test_labels" : test_seqs["labels"]}
                self.current_batches.append(batch)

            if self.at_end_of_file:
                break


        if self.shuffle:
            random.shuffle(self.current_batches)





# Performs some post-processing on batches
class LMDataCollator(DataCollatorForLanguageModeling):

    tensorizable_keys = ["input_ids", "labels", "attention_mask", "test_input_ids", "test_labels", "test_attention_mask"]
    
    def __init__(self, *args, meta=False, **kwargs):
        super(LMDataCollator, self).__init__(*args, **kwargs)
        self.meta = meta

    def torch_call(self, examples, tensorizable_keys=tensorizable_keys):
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {}
        for key in examples:
            if key in tensorizable_keys:
                batch[key] = torch.LongTensor(examples[key])
            else:
                batch[key] = examples[key]

        # Deal with the labels for the training set of this batch
        if "labels" in batch:
            labels = batch["labels"].clone()
        else:
            labels = batch["input_ids"].clone()

        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        # If it's meta, deal with the labels for the test set of this batch
        if self.meta:
            if "test_labels" in batch:
                test_labels = batch["test_labels"].clone()
            else:
                test_labels = batch["test_input_ids"].clone()

            if self.tokenizer.pad_token_id is not None:
                test_labels[test_labels == self.tokenizer.pad_token_id] = -100

            batch["test_labels"] = test_labels

        return batch


# Input: List of lists
# Output: List of lists that are all length_to_pad_to long
def pad_batch(batch, length_to_pad_to=None, padding_token=None, pad_to_const=None, pad_to_max=False, pad_left=False, context_size=None):

    if pad_to_const is not None:
        length_to_pad_to = pad_to_const
    elif pad_to_max:
        length_to_pad_to = max([len(x) for x in batch])

    if length_to_pad_to > context_size:
        length_to_pad_to = context_size

    new_batch = []

    for elt in batch:
        new_elt = elt[:]

        if pad_to_const is not None:
            if len(new_elt) > pad_to_const:
                new_elt = new_elt[:pad_to_const]

        if len(new_elt) > context_size:
            new_elt = new_elt[:context_size]

        num_pad_tokens_needed = length_to_pad_to - len(new_elt)
        pad_tokens = [padding_token for _ in range(num_pad_tokens_needed)]

        if pad_left:
            new_elt = pad_tokens + new_elt
        else:
            new_elt = new_elt + pad_tokens

        new_batch.append(new_elt)

    return new_batch






