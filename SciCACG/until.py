from matplotlib import pyplot as plt
import os
import shutil
import torch
import re
import datetime

def correct_predictions(output, targets):
    # out_classes = torch.argmax(output, dim=1)
    correct = torch.sum(output == targets).item()
    return correct

def plot1(epochs_count, train_losses, valid_losses):
    plt.figure()
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    save_path = os.path.join('fig', "compare_loss.png")
    plt.savefig(save_path)


def plot2(epochs_count, train_acc, valid_acc):
    plt.figure()
    plt.plot(epochs_count, train_acc, "-r")
    plt.plot(epochs_count, valid_acc, "-b")
    plt.xlabel("epoch")
    plt.ylabel("accaury")
    plt.legend(["Training accuary", "Validation accuary"])
    plt.title("model accuary")
    save_path = os.path.join('fig', "compare_accuary.png")
    plt.savefig(save_path)


loss_object = torch.nn.CrossEntropyLoss(reduction='none')


def mask_loss_func(real, pred):
    _loss = loss_object(pred.transpose(-1, -2), real)  # [B, E]
    mask = torch.logical_not(real.eq(0)).type(_loss.dtype)
    _loss *= mask
    return _loss.sum() / mask.sum().item()


def mask_accuracy_func(real, pred):
    _pred = pred.argmax(dim=-1)  # [B, E, V]=>[B, E]
    corrects = _pred.eq(real)  # [B, E]
    mask = torch.logical_not(real.eq(0))
    corrects *= mask
    return corrects.sum().float() / mask.sum().item()


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