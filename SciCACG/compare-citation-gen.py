import json
import random
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from matplotlib import pyplot as plt

from data_preprocess import read_dataset, PaperDataset
from CCG_train import train, evaluate
from CCG_Testing import test

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

        self.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                        nn.Linear(self.model.config.hidden_size * 2, self.model.config.hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(self.model.config.hidden_size, num_labels))

        self._word_embedding = nn.Embedding(self.tokenizer.vocab_size,
                                            self.model.config.hidden_size)

        self._decodelayer = TransformerDecoderLayer(self.model.config.hidden_size,
                                                    nhead=6,
                                                    batch_first=True)

        self._decode = TransformerDecoder(self._decodelayer,
                                          num_layers=5)

        self.generator = nn.Linear(self.model.config.hidden_size,
                                   self.tokenizer.vocab_size)

    def forward(self, p1, p2, p3):
        encoding1 = self.tokenizer(p1,
                                   truncation=True,
                                   padding=True,
                                   max_length=512,
                                   return_tensors='pt').to(self.device)
        encoding2 = self.tokenizer(p2,
                                   truncation=True,
                                   padding=True,
                                   max_length=512,
                                   return_tensors='pt').to(self.device)
        labels = self.tokenizer(p3,
                                truncation=True,
                                padding=True,
                                max_length=30,
                                return_tensors='pt')['input_ids'].to(self.device)

        logits1 = self.model(input_ids=encoding1['input_ids'],
                             attention_mask=encoding1['attention_mask'])
        logits2 = self.model(input_ids=encoding2['input_ids'],
                             attention_mask=encoding2['attention_mask'])

        logits1 = logits1.last_hidden_state
        logits2 = logits2.last_hidden_state

        logits_p1 = logits1[:, 0]
        logits_p2 = logits2[:, 0]

        logits = torch.cat((logits1, logits2), dim=1)
        logits_compare = torch.cat((logits_p1, logits_p2), dim=1)

        logits_class = self.classifier(logits_compare)
        pred_class = nn.functional.softmax(logits_class, dim=-1)

        cit_input = torch.cat((torch.argmax(pred_class, dim=1).unsqueeze(1), labels[:, 1:-1]), dim=-1)
        tgt_out = labels[:, 1:]

        tgt = self._word_embedding(cit_input)
        tgt_seq_len = cit_input.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        cit_logits = self._decode(tgt, logits, tgt_mask)
        cit_logits = F.log_softmax(self.generator(cit_logits), dim=-1)

        citation = torch.argmax(cit_logits, dim=-1)

        citation = self.tokenizer.batch_decode(citation)
        pre_citation = self.tokenizer.batch_decode(tgt_out)

        return pred_class, cit_logits, tgt_out, citation, pre_citation

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=self.device), diagonal=1)
        return mask

    def generate(self, p1, p2, p3):
        encoding1 = self.tokenizer(p1,
                                   truncation=True,
                                   padding=True,
                                   max_length=512,
                                   return_tensors='pt').to(self.device)
        encoding2 = self.tokenizer(p2,
                                   truncation=True,
                                   padding=True,
                                   max_length=512,
                                   return_tensors='pt').to(self.device)
        labels = self.tokenizer(p3,
                                truncation=True,
                                padding=True,
                                max_length=30,
                                return_tensors='pt')['input_ids'].to(self.device)

        logits1 = self.model(input_ids=encoding1['input_ids'],
                             attention_mask=encoding1['attention_mask'])
        logits2 = self.model(input_ids=encoding2['input_ids'],
                             attention_mask=encoding2['attention_mask'])

        logits1 = logits1.last_hidden_state
        logits2 = logits2.last_hidden_state

        logits_p1 = logits1[:, 0]
        logits_p2 = logits2[:, 0]

        logits = torch.cat((logits1, logits2), dim=1)
        logits_compare = torch.cat((logits_p1, logits_p2), dim=1)

        logits_class = self.classifier(logits_compare)
        pred_class = nn.functional.softmax(logits_class, dim=-1)

        summary = torch.argmax(pred_class, dim=1).unsqueeze(1)

        for i in range(labels.shape[1] - 1):

            tgt = self._word_embedding(summary)

            tgt_seq_len = summary.shape[1]

            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

            cit = self._decode(tgt, logits, tgt_mask)
            cit = F.log_softmax(self.generator(cit), dim=-1)

            next_word_index = cit[:, -1, :].argmax(dim=1)

            summary = torch.cat([summary, next_word_index.unsqueeze(1)], dim=1)

            if (next_word_index == self.tokenizer.sep_token_id).all():
                break

        summary = summary[:,1:]
        labels = labels[:,1:]
        pred_citation = self.tokenizer.batch_decode(summary)
        citation = self.tokenizer.batch_decode(labels)

        return pred_citation, citation,pred_class


# 设置训练超参数
batch_size = 64
epochs = 24
num_labels = 2
learning_rate = 1e-4
patience = 5

print(20 * "=", " Preparing for training ", 20 * "=")

# 创建训练集和测试集的数据加载器
train_data, test_data = read_dataset('data/new_ComparePaper.jsonl')

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

    train_time, train_loss, train_accuary1, \
        train_accuracy2, train_precision, train_recall, train_f1 = train(model,
                                                                         train_loader,
                                                                         optimizer,
                                                                         criterion,
                                                                         device)

    train_losses.append(train_loss)
    train_acc.append(train_accuary1)
    print("-> Training time: {:.4f}s, loss: {:.4f},gen-accuracy: {:.4f}%,com-accuracy: {:.4f}%" \
          .format(train_time, train_loss,
                  (train_accuary1 * 100), (train_accuracy2 * 100)))
    print("-> precision: {:.4f}%,recall: {:.4f}%, f1: {:.4f}%" \
          .format(train_precision * 100, train_recall * 100,
                  train_f1 * 100))

    print("* Validation for epoch {}:".format(epoch))
    dev_time, dev_loss, val_accuary1, \
        val_accuary2, rouge, val_precision, val_recall, val_f1 = evaluate(model,
                                                                          test_loader,
                                                                          criterion)

    valid_losses.append(dev_loss)
    vaild_acc.append(val_accuary1)
    print("-> Valid. time: {:.4f}s, loss: {:.4f},gen-accuracy: {:.4f}%,com-accuracy: {:.4f}%" \
          .format(dev_time, dev_loss, (val_accuary1 * 100), (val_accuary2 * 100)))
    print("-> precision: {:.4f}%,recall: {:.4f}%, f1: {:.4f}%" \
          .format(val_precision * 100, val_recall * 100,
                  val_f1 * 100))
    print(f'ROUGE-1 Score: {rouge["rouge-1"]["f"] * 100:.2f}',
          f'ROUGE-2 Score: {rouge["rouge-2"]["f"] * 100:.2f}',
          f'ROUGE-L Score: {rouge["rouge-l"]["f"] * 100:.2f}')

    if val_accuary1 < best_score:
        patience_counter += 1
    else:
        best_score = val_accuary1
        patience_counter = 0
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score},
                   os.path.join('checkpoint', "bestcomparegen.pth.tar"))

    if patience_counter >= patience:
        print("-> Early stopping: patience limit reached, stopping...")
        break


def plot1(epochs_count, train_losses, valid_losses):
    plt.figure()
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    save_path = os.path.join('fig', "compare_citation_gen_loss.png")
    plt.savefig(save_path)


def plot2(epochs_count, train_acc, valid_acc):
    plt.figure()
    plt.plot(epochs_count, train_acc, "-r")
    plt.plot(epochs_count, valid_acc, "-b")
    plt.xlabel("epoch")
    plt.ylabel("accaury")
    plt.legend(["Training accuary", "Validation accuary"])
    plt.title("model accuary")
    save_path = os.path.join('fig', "compare_citation_gen_accuary.png")
    plt.savefig(save_path)


# Plotting of the loss curves for the train and validation sets.
plot1(epochs_count, train_losses, valid_losses)
plot2(epochs_count, train_acc, vaild_acc)

# 加载最佳模型进行测试
checkpoint = torch.load("checkpoint/bestcomparegen.pth.tar")
model.load_state_dict(checkpoint['model'])
print("\n", 20 * "=", "Testing model on device: {}".format(device), 20 * "=")

batch_time, total_time, test_accuary,rouge,\
    test_precision,test_recall,test_f1 = test(model, test_loader)

print("-> Average batch processing time: {:.4f}s, total test time:,{:.4f}s,accuracy: {:.4f}%"\
      .format(batch_time, total_time,test_accuary*100))

print("-> precision: {:.4f}%,recall: {:.4f}%, f1: {:.4f}%".\
      format(test_precision * 100,test_recall*100, test_f1*100))

print(f'ROUGE-1 Score: {rouge["rouge-1"]["f"] * 100:.2f}',
      f'ROUGE-2 Score: {rouge["rouge-2"]["f"] * 100:.2f}',
      f'ROUGE-L Score: {rouge["rouge-l"]["f"] * 100:.2f}')
