import json
import random
import time
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix, recall_score, f1_score

from until import plot1, plot2, correct_predictions
from data_preprocess import read_dataset, PaperDataset

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self, modelname, device, num_labels):
        super(Model, self).__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.model = AutoModel.from_pretrained(modelname)

        for param in self.model.parameters():
            param.requires_grad = False

        self._word_embedding = nn.Embedding(self.tokenizer.vocab_size,
                                            256)

        self._encoder = nn.LSTM(256,
                                300,
                                num_layers=6,
                                dropout=0.1,
                                bidirectional=True)

        self.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                        nn.Linear(1024, 600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(600, num_labels))

    def forward(self, p1, p2, p3):
        encoding1 = self.tokenizer(p1,
                                   truncation=True,
                                   padding=True,
                                   max_length=512,
                                   return_tensors='pt')['input_ids'].to(self.device)
        encoding2 = self.tokenizer(p2,
                                   truncation=True,
                                   padding=True,
                                   max_length=512,
                                   return_tensors='pt')['input_ids'].to(self.device)

        # emb_p1 = self._word_embedding(encoding1['input_ids'])
        # emb_p2 = self._word_embedding(encoding2['input_ids'])
        #
        # encode_p1,_ = self._encoder(emb_p1)
        # encode_p2,_ = self._encoder(emb_p2)
        #
        # logits_p1 = encode_p1[:,0]
        # logits_p2 = encode_p2[:,0]

        logits = torch.cat((encoding1,encoding2), dim=1).to(self.device)
        self.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                        nn.Linear(logits.shape[1], 600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(600, num_labels))

        output = self.classifier(logits.to(torch.float).to(self.device)).to(self.devcie)
        output = nn.functional.softmax(output, dim=-1)

        return output


# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    epoch_start = time.time()
    batch_time_avg = 0.0
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

        outputs = model(p1, p2, p3)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        loss.backward()
        pred = torch.argmax(outputs, dim=1)
        correct_preds += correct_predictions(pred, labels)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        batch_time_avg += time.time() - batch_start

        optimizer.step()

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    total_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    epoch_time = time.time() - epoch_start
    loss = total_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return loss, epoch_time, epoch_accuracy, precision, recall, f1


# 验证函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    epoch_start = time.time()
    batch_time_avg = 0.0
    correct_preds = 0
    all_preds = []
    all_labels = []
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

            outputs = model(p1, p2, p3)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct_preds += correct_predictions(pred, labels)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            batch_time_avg += time.time() - batch_start
            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
                .format(batch_time_avg / (batch_index + 1),
                        total_loss / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    loss = total_loss / len(dataloader)
    epoch_time = time.time() - epoch_start
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return loss, epoch_time, epoch_accuracy, precision, recall, f1


# 测试函数
def test(model, dataloader, device):
    model.eval()
    epoch_start = time.time()
    batch_time_avg = 0.0
    correct_preds = 0
    all_preds = []
    all_labels = []
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

            outputs = model(p1, p2, p3)

            pred = torch.argmax(outputs, dim=1)
            correct_preds += correct_predictions(pred, labels)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_time_avg += time.time() - batch_start
            description = "Avg. batch proc. time: {:.4f}s".format(batch_time_avg / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    epoch_time = time.time() - epoch_start
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_accuracy, precision, recall, f1, conf_matrix


# 设置训练超参数
batch_size = 32
epochs = 64
num_labels = 2
learning_rate = 1e-4
patience = 10

print(20 * "=", " Preparing for training ", 20 * "=")

# 创建训练集和测试集的数据加载器
train_data, test_data = read_dataset('data/ComparePaper.jsonl')

print("\t* Loading training data...")
train_dataset = PaperDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

print("\t* Loading validation data...")
test_dataset = PaperDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size)

# Model loading
print("\t* Building model...")

modelname = "allenai/scibert_scivocab_uncased"
model = Model(modelname, device, num_labels).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 开始训练和评估过程
start_epoch = 1
best_score = 0
patience_counter = 0

# Data for loss curves plot.
epochs_count = []
train_losses = []
valid_losses = []
train_acc = []
vaild_acc = []

print("\n", 20 * "=", "Training model on device: {}".format(device), 20 * "=")
for epoch in range(start_epoch, epochs + 1):
    epochs_count.append(epoch)

    print("* Training epoch {}:".format(epoch))

    train_loss, train_time, train_precision, train_accuary, train_recall, train_f1 = train(model, train_loader,
                                                                                           optimizer, criterion, device)

    train_losses.append(train_loss)
    train_acc.append(train_accuary)
    print(
        "-> Training time: {:.4f}s, loss: {:.4f},accuracy: {:.4f}%".format(train_time, train_loss, train_accuary * 100))
    print("-> precision: {:.4f}%,recall: {:.4f}%, f1: {:.4f}%".format(train_precision * 100, train_recall * 100,
                                                                      train_f1 * 100))

    print("* Validation for epoch {}:".format(epoch))
    dev_loss, dev_time, val_accuary, val_precision, val_recall, val_f1 = evaluate(model, test_loader, criterion, device)

    valid_losses.append(dev_loss)
    vaild_acc.append(val_accuary)
    print("-> Valid. time: {:.4f}s, loss: {:.4f},accuracy: {:.4f}%".format(dev_time, dev_loss, val_accuary * 100))
    print("-> precision: {:.4f}%,recall: {:.4f}%, f1: {:.4f}%".format(val_precision * 100, val_recall * 100,
                                                                      val_f1 * 100))

    if val_accuary < best_score:
        patience_counter += 1
    else:
        best_score = val_accuary
        patience_counter = 0
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "recall": val_recall,
                    "f1": val_f1},
                   os.path.join('checkpoint', "bestbaseline.pth.tar"))

    if patience_counter >= patience:
        print("-> Early stopping: patience limit reached, stopping...")
        break


# 加载最佳模型进行测试
checkpoint = torch.load("checkpoint/bestbaseline.pth.tar")
model.load_state_dict(checkpoint['model'])
print("\n", 20 * "=", "Testing model on device: {}".format(device), 20 * "=")

test_time, acc, precision, recall, f1, conf_matrix = test(model, test_loader, device)
print("-> Test. time: {:.4f}s,accuracy: {:.4f}%".format(test_time, acc * 100))
print("-> precision: {:.4f}%,recall: {:.4f}%, f1: {:.4f}%".format(precision * 100, recall * 100, f1 * 100))


