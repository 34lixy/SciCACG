"""
Utility functions for training and validating models.
"""
import json
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from until import mask_loss_func, mask_accuracy_func, indices_to_words
from rouge import Rouge
from until import pyrouge_score_all


def train(model,
          dataloader,
          criterion,
          optimizer,
          max_gradient_norm):
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    running_metric = 0.0
    tqdm_batch_iterator = tqdm(dataloader, delay=0.01)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        citing_paper_abstract = batch["citing_paper_abstract"].to(device)
        citing_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
        cited_paper_abstract = batch["citing_paper_abstract"].to(device)
        cited_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
        citation = batch["citation"].to(device)
        cit_input = citation[:, :-1]
        optimizer.zero_grad()

        logits = model(citing_paper_abstract,
                       citing_paper_abstract_lengths,
                       cited_paper_abstract,
                       cited_paper_abstract_lengths,
                       cit_input)
        tgt_out = citation[:, 1:]
        loss = mask_loss_func(tgt_out, logits)

        metric = mask_accuracy_func(tgt_out, logits)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        running_metric += metric.item()

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_metric = running_metric / len(dataloader)

    return epoch_time, epoch_loss, epoch_metric


def validate(model, dataloader,criterion):
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_metric = 0.0
    batch_time_avg = 0.0
    preds = []
    cits = []

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        tqdm_batch_iterator = tqdm(dataloader, delay=0.01)
        for batch_index, batch in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            # Move input and output data to the GPU if it is used.
            citing_paper_abstract = batch["citing_paper_abstract"].to(device)
            citing_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
            cited_paper_abstract = batch["citing_paper_abstract"].to(device)
            cited_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
            citation = batch["citation"].to(device)
            # 在 train 函数或 validate 函数的开头添加以下打印语句
            cit_input = citation[:, :-1]
            logits = model(citing_paper_abstract,
                           citing_paper_abstract_lengths,
                           cited_paper_abstract,
                           cited_paper_abstract_lengths,
                           cit_input)
            cit_output = citation[:, 1:]
            loss = mask_loss_func(cit_output, logits)
            metric = mask_accuracy_func(cit_output, logits)

            word_indices = logits.cpu().numpy()
            label = cit_output.cpu().numpy()
            word_indices = np.argmax(word_indices, axis=-1)
            pred, cit = indices_to_words(word_indices, label)
            for pr, re in zip(pred, cit):
                preds.append(' '.join(str(item) for item in pr))
                cits.append(' '.join(str(item) for item in re))
            running_loss += loss.item()
            running_metric += metric.item()
            batch_time_avg += time.time() - batch_start
            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
                .format(batch_time_avg / (batch_index + 1),
                        running_loss / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)
    rouge = Rouge()
    rouge_sorce = rouge.get_scores(preds, cits, avg=True)
    rouge_sorce2 = pyrouge_score_all(preds, cits)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_metric = running_metric / len(dataloader)

    return epoch_time, epoch_loss, epoch_metric, rouge_sorce, rouge_sorce2
