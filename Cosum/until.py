from matplotlib import pyplot as plt
import pickle
import os
import random
import shutil
import string
import torch
import re
import datetime

loss_object = torch.nn.CrossEntropyLoss(reduction='none')

# 几种损失函数
def mask_loss_func(real, pred):
    _loss = loss_object(pred.transpose(-1, -2), real)  # [B, E]
    mask = torch.logical_not(real.eq(0)).type(_loss.dtype)
    _loss *= mask
    return _loss.sum() / mask.sum().item()


def mask_loss_func2(real, pred):
    _loss = loss_object(pred.transpose(-1, -2), real)
    mask = torch.logical_not(real.eq(0))
    _loss = _loss.masked_select(mask)
    return _loss.mean()


# 几种准确率指标函数
def mask_accuracy_func(real, pred):
    _pred = pred.argmax(dim=-1)  # [B, E, V]=>[B, E]
    corrects = _pred.eq(real)  # [B, E]
    mask = torch.logical_not(real.eq(0))
    corrects *= mask
    return corrects.sum().float() / mask.sum().item()


# 另一种实现方式
def mask_accuracy_func2(real, pred):
    _pred = pred.argmax(dim=-1)
    corrects = _pred.eq(real).type(torch.float32)
    mask = torch.logical_not(real.eq(0))
    corrects = corrects.masked_select(mask)
    return corrects.mean()


def mask_accuracy_func3(real, pred):
    _pred = pred.argmax(dim=-1)
    corrects = _pred.eq(real)
    mask = torch.logical_not(real.eq(0))
    corrects = torch.logical_and(corrects, mask)
    return corrects.sum().float() / mask.sum().item()


def plot(epochs_count, train_losses, valid_losses):
    plt.figure()
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    plt.savefig("loss.png")

def indices_to_words (word_indices, label):
    with open("preprocessed_data/reverse_worddict.pkl", "rb") as word_file:
        reverse_worddict = pickle.load(word_file)

    # Convert decoder output to words
    summary_words = [[reverse_worddict[i] for i in row] for row in word_indices]

    label_words = [[reverse_worddict[i] for i in row] for row in label]

    return summary_words,label_words

def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


_ROUGE_PATH = "/home/lxy/ROUGE-1.5.5"
_PYROUGE_TEMP_FILE = "/home/lxy/ROUGE-1.5.5/temp"

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}

def clean(x):
    x = x.lower()
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)

def pyrouge_score_all(hyps_list, refer_list, remap=True):
    from pyrouge import Rouge155
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join(_PYROUGE_TEMP_FILE, nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT, 'result')
    MODEL_PATH = os.path.join(PYROUGE_ROOT, 'gold')
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)
    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        model_file = os.path.join(MODEL_PATH, 'Reference.A.%d.txt' % i)

        refer = clean(refer_list[i]) if remap else refer_list[i]
        hyps = clean(hyps_list[i]) if remap else hyps_list[i]

        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))
        with open(model_file, 'wb') as f:
            f.write(refer.encode('utf-8'))

    r = Rouge155(_ROUGE_PATH, log_level='WARNING')

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Reference.[A-Z].#ID#.txt'

    try:
        output = r.convert_and_evaluate(rouge_args="-e %s -a -m -n 2 -d" % os.path.join(_ROUGE_PATH, "data"))
        output_dict = r.output_to_dict(output)
    finally:
        shutil.rmtree(PYROUGE_ROOT)

    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = output_dict['rouge_1_precision'], \
        output_dict['rouge_1_recall'], output_dict['rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = output_dict['rouge_2_precision'], \
        output_dict['rouge_2_recall'], output_dict['rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = output_dict['rouge_l_precision'], \
        output_dict['rouge_l_recall'], output_dict['rouge_l_f_score']
    return scores