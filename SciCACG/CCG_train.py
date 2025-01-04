import json
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from until import mask_loss_func, mask_accuracy_func
from rouge import Rouge
from until import pyrouge_score_all,correct_predictions
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from bert_score import score


def train(model,
          dataloader,
          optimizer,
          criterion,
          device):
    # Switch the model to train mode.
    model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    running_metric = 0.0
    correct_preds = 0
    all_preds = []
    all_labels = []
    tqdm_batch_iterator = tqdm(dataloader, delay=0.01)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        p1 = batch["citing_paper_abstract"]
        p2 = batch["cited_paper_abstract"]
        p3 = batch["citation"]
        labels = batch["Is_compare"]

        label_to_int = {label: idx for idx, label in enumerate(set(labels))}
        labels = [label_to_int[label] for label in labels]
        labels = torch.tensor(labels).to(device)

        optimizer.zero_grad()
        model.zero_grad()

        pred_class,cit_logits, tgt_out,citation, pre_citation = model(p1, p2, p3)

        loss1 = mask_loss_func(tgt_out, cit_logits)
        loss2 = criterion(pred_class, labels)
        loss = loss1 + loss2
        metric = mask_accuracy_func(tgt_out, cit_logits)

        loss.backward()

        pred = torch.argmax(pred_class, dim=1)
        correct_preds += correct_predictions(pred, labels)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        running_metric += metric.item()

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_metric = running_metric / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_metric,epoch_accuracy,precision,recall,f1


def  evaluate(model, dataloader, criterion):
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_metric = 0.0
    batch_time_avg = 0.0
    preds = []
    cits = []
    correct_preds = 0
    all_preds = []
    all_labels = []

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        tqdm_batch_iterator = tqdm(dataloader, delay=0.01)
        for batch_index, batch in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            p1 = batch["citing_paper_abstract"]
            p2 = batch["cited_paper_abstract"]
            p3 = batch["citation"]
            labels = batch["Is_compare"]

            label_to_int = {label: idx for idx, label in enumerate(set(labels))}
            labels = [label_to_int[label] for label in labels]
            labels = torch.tensor(labels).to(device)

            pred_class,cit_logits, tgt_out,citation, pre_citation = model(p1, p2, p3)

            loss1 = mask_loss_func(tgt_out, cit_logits)
            loss2 = criterion(pred_class, labels)
            loss = loss1 + loss2
            metric = mask_accuracy_func(tgt_out, cit_logits)

            pred = torch.argmax(pred_class, dim=1)
            correct_preds += correct_predictions(pred, labels)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for pr, re in zip(citation, p3):
                preds.append(pr)
                cits.append(re)

            running_loss += loss.item()
            running_metric += metric.item()
            batch_time_avg += time.time() - batch_start
            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
                .format(batch_time_avg / (batch_index + 1),
                        running_loss / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    rouge_sorce = pyrouge_score_all(preds, cits)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_metric = running_metric / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_metric, epoch_accuracy,rouge_sorce,precision,recall,f1

