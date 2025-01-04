"""
Utility functions for training and validating models.
"""

import time
import torch
import json
import torch.nn as nn

from tqdm import tqdm
from model.utils import correct_predictions


def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm):

    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader,delay=0.01)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        citing_paper_abstract = batch["citing_paper_abstract"].to(device)
        citing_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
        cited_paper_abstract = batch["citing_paper_abstract"].to(device)
        cited_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, probs = model(citing_paper_abstract,
                              citing_paper_abstract_lengths,
                              cited_paper_abstract,
                              cited_paper_abstract_lengths)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):

    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    preds = []
    label = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            citing_paper_abstract = batch["citing_paper_abstract"].to(device)
            citing_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
            cited_paper_abstract = batch["citing_paper_abstract"].to(device)
            cited_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
            labels = batch["label"].to(device)


            logits, probs = model(citing_paper_abstract,
                                  citing_paper_abstract_lengths,
                                  cited_paper_abstract,
                                  cited_paper_abstract_lengths)

            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy
