import time
import argparse
import pickle
import torch
import os
import json
from torch.utils.data import DataLoader
from data import ComparePaperDataset
from model.model import ESIM
from utils import correct_predictions


def test(model, dataloader):

    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            citing_paper_abstract = batch["citing_paper_abstract"].to(device)
            citing_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
            cited_paper_abstract = batch["citing_paper_abstract"].to(device)
            cited_paper_abstract_lengths = batch["cited_paper_abstract_length"].to(device)
            labels = batch["label"].to(device)

            _, probs = model(citing_paper_abstract,
                             citing_paper_abstract_lengths,
                             cited_paper_abstract,
                             cited_paper_abstract_lengths)

            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy


def testing(test_file, pretrained_file, batch_size=32):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    checkpoint = torch.load(pretrained_file)

    # Retrieving model parameters from checkpoint.
    vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
    embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
    hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
    num_classes = checkpoint["model"]["_classification.4.weight"].size(0)

    print("\t* Loading test data...")
    with open(test_file, "rb") as pkl:
        test_data = ComparePaperDataset(pickle.load(pkl))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    model = ESIM(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)

    model.load_state_dict(checkpoint["model"])

    print(20 * "=",
          " Testing model model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, accuracy = test(model, test_loader)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy * 100)))



testing("preprocessed_data/test_data.pkl", "checkpoints/best.pth.tar", 64)
