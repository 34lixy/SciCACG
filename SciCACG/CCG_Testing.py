import json
import time
import torch
from tqdm import tqdm
from until import pyrouge_score_all,correct_predictions
from sklearn.metrics import confusion_matrix, recall_score, f1_score

def test(model, dataloader):
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    preds = []
    cits = []
    batch_time_avg = 0.0
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

            pred_citation, citation,pred_class = model.generate(p1, p2, p3)

            pred = torch.argmax(pred_class, dim=1)
            correct_preds += correct_predictions(pred, labels)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for pr, re in zip(pred_citation, citation):
                preds.append(pr)
                cits.append(re)
            batch_time += time.time() - batch_start
            batch_time_avg += time.time() - batch_start
            description = "Avg. batch proc. time: {:.4f}s".format(batch_time_avg / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    rouge_sorce = pyrouge_score_all(preds, cits)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    epoch_accuracy = correct_preds / len(dataloader.dataset)
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

    return batch_time, total_time, epoch_accuracy,rouge_sorce,precision,recall,f1