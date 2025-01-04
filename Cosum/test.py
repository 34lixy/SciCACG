import time
import argparse
import pickle
import torch
import os
import json
from rouge import Rouge
from torch.utils.data import DataLoader
from data import ComparePaperDataset
from model.model import ESCG
from until import indices_to_words, pyrouge_score_all
from tqdm import tqdm


def test(model, dataloader):
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    preds = []
    cits = []
    batch_time_avg = 0.0
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
            batch_size = citation.size(0)
            start_symbol_tensor = torch.tensor([[2]], device=device)

            summary = start_symbol_tensor.repeat(batch_size, 1)
            for i in range(citation.shape[1]-1):
                output = model(citing_paper_abstract,
                               citing_paper_abstract_lengths,
                               cited_paper_abstract,
                               cited_paper_abstract_lengths,
                               summary)
                next_word_index = output[:, -1, :].argmax(dim=1)

                summary = torch.cat([summary, next_word_index.unsqueeze(1)], dim=1)

                if (next_word_index == 3).all():
                    break

            word_indices = summary.cpu().numpy()
            label = citation.cpu().numpy()
            pred, cit = indices_to_words(word_indices, label)
            for pr, re in zip(pred, cit):
                preds.append(' '.join(str(item) for item in pr))
                cits.append(' '.join(str(item) for item in re))
            batch_time += time.time() - batch_start
            batch_time_avg += time.time() - batch_start
            description = "Avg. batch proc. time: {:.4f}s" .format(batch_time_avg / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)
    rouge = Rouge()
    rouge_sorce = rouge.get_scores(preds, cits, avg=True)
    rouge_sorce2 = pyrouge_score_all(preds,cits)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    an = []
    with open("test.jsonl", "w") as f:
        for p, r in zip(preds, cits):
            newdata = {
                "pre": p,
                "cit": r
            }
            an.append(newdata)
        for i in an:
            json.dump(i, f)
            f.write('\n')
    return batch_time, total_time, rouge_sorce,rouge_sorce2


def testing(test_file, pretrained_file, batch_size=32):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    checkpoint = torch.load(pretrained_file)

    # Retrieving model parameters from checkpoint.
    vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
    embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
    # hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
    # num_classes = checkpoint["model"]["_classification.4.weight"].size(0)
    with open(test_file, "rb") as pkl:
        test_data = ComparePaperDataset(pickle.load(pkl))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    model = ESCG(vocab_size, embedding_dim, 300, dropout=0.6, device=device).to(device)

    model.load_state_dict(checkpoint["model"])

    print(20 * "="," Testing model model on device: {} ".format(device),20 * "=")

    batch_time, total_time, rouge_score,pyrouge_score = test(model, test_loader)

    print("-> Average batch processing time: {:.4f}s, total test time:{:.4f}s".format(batch_time, total_time))
    print(f'ROUGE-1 Score: {rouge_score["rouge-1"]["f"] * 100:.2f}',
          f'ROUGE-2 Score: {rouge_score["rouge-2"]["f"] * 100:.2f}',
          f'ROUGE-L Score: {rouge_score["rouge-l"]["f"] * 100:.2f}')
    print(f'ROUGE-1 Score: {pyrouge_score["rouge-1"]["f"] * 100:.2f}',
          f'ROUGE-2 Score: {pyrouge_score["rouge-2"]["f"] * 100:.2f}',
          f'ROUGE-L Score: {pyrouge_score["rouge-l"]["f"] * 100:.2f}')

# #
# testing("preprocessed_data/test_data.pkl", "checkpoints/best.pth.tar", 16)
