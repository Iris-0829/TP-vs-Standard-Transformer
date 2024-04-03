
from packaging import version
from enum import Enum
import logging
import math
import os
from collections import Counter
import copy

import torch
import higher

from itertools import chain

from transformers import PreTrainedModel, GPT2LMHeadModel, TransfoXLLMHeadModel,AutoConfig, BertModel, BertConfig, BertForMaskedLM

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from lr_scheduler import *

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# Based closely on Hugging Face's Trainer class
# This is for standard training (not meta-training)
class Trainer:

    def __init__(self, model=None, train_dataset=None, eval_dataset=None, data_collator=None, fp16=False,
            max_grad_norm=1.0, n_epochs=None, weight_decay=0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, learning_rate=5e-5,
            lr_scheduler_type="linear", warmup_steps=0, eval_every=None, log_every=None, save_dir=None, model_name=None):

        # The model
        self.model = model
        self.model_name = model_name

        # The data
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.train_size = len(train_dataset)

        # How long to train for
        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps

        # Optimizer parameters
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.max_grad_norm = max_grad_norm

        # Evaluating and saving
        self.eval_every = eval_every
        self.log_every = log_every
        self.best_loss = math.inf
        self.save_dir = save_dir


    # Generator for the model's parameters that
    # can be trained
    def trainable_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param

    # Create optimizer and learning rate scheduler
    def create_optimizer_and_scheduler(self, num_training_steps=None):
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.trainable_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.trainable_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_kwargs = {
                    "betas": (self.adam_beta1, self.adam_beta2),
                    "eps": self.adam_epsilon,
                    "lr": self.learning_rate,
                }
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        
        self.lr_scheduler = get_scheduler(
                self.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps,
            )

    # Compute loss on batch "inputs"
    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            return outputs["loss"]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise ValueError("Unexpected output type from the model.")
        # return outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
    # Move all inputs to the correct
    # device (CPU/GPU)
    def _prepare_inputs(self, inputs):

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        return inputs

    # Train on one step
    def training_step(self, model, inputs):
        
        model.train()
        inputs = {"input_ids" : inputs["input_ids"], "labels" : inputs["labels"]}
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        loss.backward()

        return loss.detach()

    # Perform one step of evaluation
    def prediction_step(self, model, inputs):
        model.eval()
        inputs = {"input_ids" : inputs["input_ids"], "labels" : inputs["labels"]}
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss = self.compute_loss(model, inputs).mean().detach()

        return loss

    # Evaluate on eval_dataset
    def evaluate(self, eval_dataset=None, name="Validation", save=False):
        model = self.model

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        eval_dataset.reset()

        total_eval_loss = 0
        for batch_index, batch in enumerate(eval_dataset):
            collated_batch = self.data_collator(batch)
            loss = self.prediction_step(model, collated_batch).item()
            total_eval_loss += loss
           
        avg_eval_loss = total_eval_loss / batch_index
        logging.info(name + " loss: " + str(avg_eval_loss))
        logging.info(name + " perplexity: " + str(math.exp(avg_eval_loss)))

        # If the loss has improved, save the model weights
        if save and avg_eval_loss < self.best_loss:
            self.best_loss = avg_eval_loss
            self.save()

    # Save model weights
    def save(self):
        logging.info("Saving model checkpoint to %s", self.save_dir)
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if not isinstance(self.model, PreTrainedModel):
            state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(self.save_dir, self.model_name + ".weights"))
        else:
            output_dir = os.path.join(self.save_dir, self.model_name)
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)

    # Load model weights
    def load_best(self):
        logging.info("Loading model checkpoint from %s", os.path.join(self.save_dir, self.model_name))

        if isinstance(self.model, GPT2LMHeadModel):
            self.model = GPT2LMHeadModel.from_pretrained(os.path.join(self.save_dir, self.model_name)).to(device)
        elif isinstance(self.model, BertForMaskedLM):
            self.model = BertForMaskedLM.from_pretrained(os.path.join(self.save_dir, self.model_name)).to(device)
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name + ".weights")))

    # Load weights from saved filename
    def load(self, model_name):
        logging.info("Loading model checkpoint from %s", os.path.join(self.save_dir, model_name))

        if isinstance(self.model, GPT2LMHeadModel):
            self.model = GPT2LMHeadModel.from_pretrained(os.path.join(self.save_dir, model_name)).to(device)
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.save_dir, model_name + ".weights")))


    # Train model
    def train(self):
        
        # Create optimizer and learning rate scheduler
        self.max_steps = self.n_epochs * self.train_size
        self.create_optimizer_and_scheduler(num_training_steps=self.max_steps)

        # Put model in training mode
        self.model.train()

        # Keep track of loss and number of updates
        tr_loss = 0
        total_batches = 0
        total_updates = 0

        self.model.zero_grad()

        # Loop over the dataset self.n_epochs times
        for epoch in range(self.n_epochs):

            # Reset the training set for the new epoch
            self.train_dataset.reset()

            avg_tr_loss = 0

            # Loop over the training set
            for batch_index, batch in enumerate(self.train_dataset):

                # Evaluate and log, if applicable
                if total_updates % self.eval_every == 0:
                    self.evaluate(save=True)

                if total_updates % self.log_every == 0:
                    logging.info(
                        "Training step " + str(total_updates) + " out of " + str(self.max_steps) + "; Epoch " + str(
                            epoch) + "; Learning rate: " + str(self.lr_scheduler.get_last_lr()[0]))
                    logging.info(f'Average training loss: {avg_tr_loss:.2f}')
                    avg_tr_loss = 0

                total_batches += 1

                # Compute the loss on one batch
                collated_batch = self.data_collator(batch)
                loss = self.training_step(self.model, collated_batch)
                avg_tr_loss += loss

                # Clip the norm of the gradient
                if self.max_grad_norm is not None and self.max_grad_norm > 0:

                    if hasattr(self.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        self.optimizer.clip_grad_norm(self.max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Backpropagate
                self.optimizer.step()
                self.model.zero_grad()
                
                self.lr_scheduler.step()

                total_updates += 1


# Trainer class for performing meta-training with MAML
# Inherits from Trainer
class MetaTrainer(Trainer):

    def __init__(self, inner_lr=1e-1, train_batch_size=None, eval_batch_size=None, multi_step_loss=False, **kwargs):
        super(MetaTrainer, self).__init__(**kwargs)

        # Set up the inner-loop optimizer
        # (The outer-loop optimizer has already been set up
        # in the class we are inheriting from)
        self.inner_lr = inner_lr
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

        # Batch size for each inner-loop episode
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # Whether to do multi-step loss
        self.multi_step_loss = multi_step_loss

    # Do one step of MAML (meta-training on a single dataset)
    def meta_training_step(self, model, inputs):

        model.train()

        train = {"input_ids" : inputs["input_ids"], "labels" : inputs["labels"]}
        test = {"input_ids" : inputs["test_input_ids"], "labels" : inputs["test_labels"]}

        batch_size = self.train_batch_size
        n_batches = len(train["input_ids"]) // batch_size

        with higher.innerloop_ctx(model, self.inner_opt, copy_initial_weights=False) as (fmodel, diffopt):

            outer_loss = 0
            test = self._prepare_inputs(test)

            # Train on the training set for this episode
            for mini_batch_index in range(n_batches):

                # Create current batch
                mini_batch = {"input_ids" : train["input_ids"][mini_batch_index*batch_size:(mini_batch_index+1)*batch_size], "labels" : train["labels"][mini_batch_index*batch_size:(mini_batch_index+1)*batch_size]}
                mini_batch = self._prepare_inputs(mini_batch)

                # Do an inner-loop training step on that batch
                inner_loss = self.compute_loss(fmodel, mini_batch)
                diffopt.step(inner_loss)

                # If using multi-step loss, compute the
                # outer-loop loss after every batch in the
                # inner loop
                if self.multi_step_loss:
                    this_loss = self.compute_loss(fmodel, test)
                    this_loss.backward(retain_graph=True)
                    outer_loss += this_loss.detach()
            
            # If not using multi-step loss, only compute
            # the outer-loop loss after training on the entire
            # inner-loop training set
            if not self.multi_step_loss:
                outer_loss = self.compute_loss(fmodel, test)
                outer_loss.backward()
                return outer_loss.detach()
            else:
                # This is the loss we accumulated inside the inner loop
                return outer_loss

    # Do one MAML evaluation step (train on a single
    # dataset and then compute the loss on its test set)
    def meta_prediction_step(self, model, inputs):
        
        # Put the model in evaluation mode
        model.eval()

        train = {"input_ids" : inputs["input_ids"], "labels" : inputs["labels"]}
        test = {"input_ids" : inputs["test_input_ids"], "labels" : inputs["test_labels"]}

        # Temporary model that we will use for evaluation
        fmodel = copy.deepcopy(model)

        # Temporary optimizer that we will use for evaluation
        diffopt = torch.optim.SGD(fmodel.parameters(), lr=self.inner_lr)

        batch_size = self.train_batch_size
        n_batches = len(train["input_ids"]) // batch_size

        # Train on the training set for this episode
        for mini_batch_index in range(n_batches):
            
            # Prepare batch
            mini_batch = {"input_ids" : train["input_ids"][mini_batch_index*batch_size:(mini_batch_index+1)*batch_size], "labels" : train["labels"][mini_batch_index*batch_size:(mini_batch_index+1)*batch_size]}
            mini_batch = self._prepare_inputs(mini_batch)

            # Train model on this batch
            inner_loss = self.compute_loss(fmodel, mini_batch)
            inner_loss.backward()
            diffopt.step()
            fmodel.zero_grad()

        # Compute trained model's loss on test set
        test = self._prepare_inputs(test)
        outer_loss = self.compute_loss(fmodel, test).mean().detach()

        return outer_loss

    # Evaluate model on whole meta-evaluation set
    # (Training it on each evaluation dataset in turn
    # and then evaluating on that dataset)
    def meta_evaluate(self, eval_dataset=None, name="Validation", save=False):

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # Reset the eval dataset to the beginning
        eval_dataset.reset()

        # Loop over the eval dataset
        total_eval_loss = 0
        for batch_index, batch in enumerate(eval_dataset):
            collated_batch = self.data_collator(batch)
            with torch.backends.cudnn.flags(enabled=False):
                loss = self.meta_prediction_step(self.model, collated_batch).item()
            total_eval_loss += loss

        # Compute and log the eval loss
        avg_eval_loss = total_eval_loss / batch_index
        logging.info(name + " meta-loss: " + str(avg_eval_loss))
        if avg_eval_loss < 25:
            logging.info(name + " meta-perplexity: " + str(math.exp(avg_eval_loss)))

        # Save the model weights if applicable
        if save and avg_eval_loss < self.best_loss:
            self.best_loss = avg_eval_loss
            self.save()


    # Meta-train model
    def meta_train(self):

        # Set up optimizer and scheduler 
        max_steps = self.n_epochs * self.train_size
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Put model in training mode
        self.model.train()


        meta_tr_loss = 0
        self.model.zero_grad()
        total_training_steps = 0

        # Loop over the meta-training set for n_epochs times
        for epoch in range(self.n_epochs):

            # Loop over the batches in the meta-training set
            for batch_index, batch in enumerate(self.train_dataset):

                # If applicable, evaluate the model
                if total_training_steps % self.eval_every == 0:
                    with torch.backends.cudnn.flags(enabled=False):
                        self.meta_evaluate(save=True)

                # If applicable, log the current step
                if total_training_steps % self.log_every == 0:
                    logging.info("Training step " + str(total_training_steps) + " out of " + str(max_steps) + "; Epoch " + str(epoch) + "; Learning rate: " + str(self.lr_scheduler.get_last_lr()[0]))

                total_training_steps += 1

                # Compute outer-loop loss for this batch
                collated_batch = self.data_collator(batch)
                with torch.backends.cudnn.flags(enabled=False):
                    meta_tr_loss += self.meta_training_step(self.model, collated_batch)

                # Clip gradient
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    if hasattr(self.optimizer, "clip_grad_norm"):
                        self.optimizer.clip_grad_norm(self.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Update weights
                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad()


    # Train a model on one dataset (not as part of meta-training)
    def train_model(self, model, dataset, opt, opt_type="AdamW", sgd_epochs=1, batch_size=None):

        # Keep track of the best loss for saving the model
        best_loss = math.inf

        test_set = {"input_ids" : dataset["test_input_ids"], "labels" : dataset["test_labels"]}

        # Create the optimizer
        if opt_type == "SGD":
            opt = torch.optim.SGD(model.parameters(), lr=self.inner_lr)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

        # Train for at most 1000 epochs
        for epoch_index in range(1000):

            model.zero_grad()

            # Evaluate once per epoch
            model.eval() # Put model in eval mode
            valid_loss = self.compute_loss(model, test_set)
            logging.info("LOSS, PERPLEXITY: " + str(valid_loss.item()) + " " + str(torch.exp(valid_loss).item()))
            model.train() # Return model to training mode

            if valid_loss >= best_loss:
                # Stop training as soon as the loss fails to improve
                break
            else:
                # Save model if loss has improved
                best_loss = valid_loss
                best_model = copy.deepcopy(model)

            # Divide training set into batches and loop over them
            for example_index in range(len(dataset["input_ids"]) // batch_size):
                opt.zero_grad()

                mini_batch = {"input_ids" : dataset["input_ids"][example_index*batch_size:(example_index+1)*batch_size], "labels" : dataset["labels"][example_index*batch_size:(example_index+1)*batch_size]}

                train_loss = self.compute_loss(model, mini_batch)
                train_loss.backward()
                opt.step()
                opt.zero_grad()
        
        return best_model



