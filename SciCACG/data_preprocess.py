import random
import json
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# 读取JSON文件数据
def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)
    # 然后，将索引划分为训练集和临时集 (test + validation)
    train_data, test_data = train_test_split(data, test_size=0.1)
    return train_data, test_data


# 定义自定义数据集类
class PaperDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]