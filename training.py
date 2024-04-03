
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
import matplotlib.pyplot as plt
import datetime

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

        # plot
        self.train_losses = []
        self.val_losses = []


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
        self.val_losses.append(avg_eval_loss)

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
        total_batches = 0
        total_updates = 0

        self.model.zero_grad()

        # Loop over the dataset self.n_epochs times
        for epoch in range(self.n_epochs):

            # Reset the training set for the new epoch
            self.train_dataset.reset()

            # Loop over the training set
            for batch_index, batch in enumerate(self.train_dataset):

                # Evaluate and log, if applicable
                if total_updates % self.eval_every == 0:
                    self.evaluate(save=True)

                if total_updates % self.log_every == 0:
                    logging.info(
                        "Training step " + str(total_updates) + " out of " + str(self.max_steps) + "; Epoch " + str(
                            epoch) + "; Learning rate: " + str(self.lr_scheduler.get_last_lr()[0]))

                total_batches += 1

                # Compute the loss on one batch
                collated_batch = self.data_collator(batch)
                loss = self.training_step(self.model, collated_batch)
                self.train_losses.append(loss.item())

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

        self.plot_losses("plot")

    def plot_losses(self, save_dir):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        axs[0].plot(self.train_losses, label='Training losses')
        axs[0].set_title('Training losses')
        axs[0].legend()

        axs[1].plot(self.val_losses, label='Validation losses')
        axs[1].set_title('Validation losses')
        axs[1].legend()

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'loss_plots_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()

