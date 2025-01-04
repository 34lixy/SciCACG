import pickle
import string
import torch
import numpy as np
import json
from collections import Counter
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt



class Preprocessor(object):
    def __init__(self, lowercase=True,
                 ignore_punctuation=True,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None):

        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos

    def read_data(self, filepath):
        with open(filepath, "r", encoding="utf8") as input_data:
            citing_paper_abstracts, \
                cited_paper_abstracts, \
                labels, citations = [], [], [], []

            # 用于跟踪词汇频率的计数器
            word_frequency = Counter()

            for line in input_data:
                data = json.loads(line)

                citing_paper_abstract = data["citing_paper_abstract"]
                cited_paper_abstract = data["cited_paper_abstract"]
                label = data["Is_compare"]
                citation = data["citation"]

                if self.lowercase:
                    citing_paper_abstract = citing_paper_abstract.lower()
                    cited_paper_abstract = cited_paper_abstract.lower()
                    citation = citation.lower()

                if self.ignore_punctuation:
                    citing_paper_abstract = citing_paper_abstract.translate(str.maketrans("", "", string.punctuation))
                    citing_paper_abstract = ''.join(
                        [c for c in citing_paper_abstract if c not in string.punctuation and not c.isdigit()])
                    cited_paper_abstract = cited_paper_abstract.translate(str.maketrans("", "", string.punctuation))
                    cited_paper_abstract = ''.join(
                        [c for c in cited_paper_abstract if c not in string.punctuation and not c.isdigit()])
                    citation = citation.translate(str.maketrans(",", string.punctuation))
                    citation = ''.join([c for c in citation if c not in string.punctuation and not c.isdigit()])

                # 分词并更新词汇频率计数器
                citing_words = [w for w in citing_paper_abstract.rstrip().split() if w not in self.stopwords]
                cited_words = [w for w in cited_paper_abstract.rstrip().split() if w not in self.stopwords]
                citation_words = [w for w in citation.rstrip().split() if w not in self.stopwords]

                word_frequency.update(citing_words)
                word_frequency.update(cited_words)
                word_frequency.update(citation_words)

                citing_paper_abstracts.append(citing_words)
                cited_paper_abstracts.append(cited_words)
                labels.append(label)
                citations.append(citation_words)

            # 仅保留频率较高的词汇
            min_word_frequency = 4
            citing_paper_abstracts = [[word for word in words if word_frequency[word] > min_word_frequency] for words in
                                      citing_paper_abstracts]
            cited_paper_abstracts = [[word for word in words if word_frequency[word] > min_word_frequency] for words in
                                     cited_paper_abstracts]
            citations = [[word for word in words if word_frequency[word] > min_word_frequency] for words in citations]

            return {"citing_paper_abstracts": citing_paper_abstracts,
                    "cited_paper_abstracts": cited_paper_abstracts,
                    "labels": labels,
                    "citation": citations}

    def build_worddict(self, data):
        words = set()
        [words.update(abstract) for abstract in data["citing_paper_abstracts"]]
        [words.update(abstract) for abstract in data["cited_paper_abstracts"]]
        [words.update(citation) for citation in data["citation"]]
        counts = Counter(words)

        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}
        self.reverse_worddict = {}
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1

        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset += 1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset += 1

        for i, word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i + offset

        if self.labeldict == {}:
            label_names = set(data["labels"])
            self.labeldict = {label_name: i
                              for i, label_name in enumerate(label_names)}
        self.reverse_worddict = {value: key for key, value in self.worddict.items()}

    def words_to_indices(self, sentence):
        indices = []
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                index = self.worddict["_OOV_"]
            indices.append(index)

        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def indices_to_words(self, indices):

        return [self.reverse_worddict[idx][0] for idx in indices]

    def transform_to_indices(self, data):
        transformed_data = {"citing_paper_abstracts": [],
                            "cited_paper_abstracts": [],
                            "labels": [],
                            "citation": []}

        for i, abstract_1 in enumerate(data["citing_paper_abstracts"]):
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            transformed_data["labels"].append(self.labeldict[label])

            indices = self.words_to_indices(abstract_1)
            transformed_data["citing_paper_abstracts"].append(indices)

            indices = self.words_to_indices(data["cited_paper_abstracts"][i])
            transformed_data["cited_paper_abstracts"].append(indices)

            indices = self.words_to_indices(data["citation"][i])
            transformed_data["citation"].append(indices)

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix


class ComparePaperDataset(Dataset):
    def __init__(self,
                 data,
                 padding_idx=0,
                 max_abstract_length=512,
                 max_citation_lenth = 30):
        self.citing_paper_abstracts_lengths = [len(seq) for seq in data["citing_paper_abstracts"]]
        self.cited_paper_abstracts_lengths = [len(seq) for seq in data["cited_paper_abstracts"]]
        self.citation_lengths = [len(seq) for seq in data["citation"]]

        if max_abstract_length == None:
            self.max_abstract_length = max(self.citing_paper_abstracts_lengths + self.cited_paper_abstracts_lengths)
        else:
            self.max_abstract_length = max_abstract_length
        if max_citation_lenth == None:
            self.max_citation_length = max(self.citation_lengths)
        else:
            self.max_citation_length = max_citation_lenth

        self.num_sequences = len(data["citing_paper_abstracts"])

        self.data = {
            "citing_paper_abstracts": torch.ones((self.num_sequences,
                                                  self.max_abstract_length),
                                                 dtype=torch.long) * padding_idx,
            "cited_paper_abstracts": torch.ones((self.num_sequences,
                                                 self.max_abstract_length),
                                                dtype=torch.long) * padding_idx,
            "labels": torch.tensor(data["labels"], dtype=torch.long),
            "citations":torch.ones((self.num_sequences,self.max_citation_length),
                                   dtype=torch.long)*padding_idx
        }

        for i, citing_paper_abstract in enumerate(data["citing_paper_abstracts"]):
            end = min(len(citing_paper_abstract), self.max_abstract_length)
            self.data["citing_paper_abstracts"][i][:end] = torch.tensor(citing_paper_abstract[:end])

        for i, cited_paper_abstract in enumerate(data["cited_paper_abstracts"]):
            end = min(len(cited_paper_abstract), self.max_abstract_length)
            self.data["cited_paper_abstracts"][i][:end] = torch.tensor(cited_paper_abstract[:end])

        for i, citation in enumerate(data["citation"]):
            end = min(len(citation),self.max_citation_length)
            self.data["citations"][i][:end] = torch.tensor(citation[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):

        return {
            "citing_paper_abstract": self.data["citing_paper_abstracts"][index],
            "citing_paper_abstract_length": min(
                self.citing_paper_abstracts_lengths[index], self.max_abstract_length
            ),
            "cited_paper_abstract": self.data["cited_paper_abstracts"][index],
            "cited_paper_abstract_length": min(
                self.cited_paper_abstracts_lengths[index], self.max_abstract_length
            ),
            "label": self.data["labels"][index],
            "citation":self.data["citations"][index],
            "citation_length":min(
                self.citation_lengths[index],self.max_citation_length
            )
        }

# # 实例化 Preprocessor 类并设置参数
# preprocessor = Preprocessor(lowercase=True, ignore_punctuation=True, num_words=1000, stopwords=["这是"])
#
# # 读取数据并进行预处理
# data = preprocessor.read_data("zuizhong.jsonl")
#
# # 构建单词字典和嵌入矩阵
# preprocessor.build_worddict(data)
# embedding_matrix = preprocessor.build_embedding_matrix("glove.6B.300d.txt")
# with open(os.path.join('preprocessed_data', "embeddings.pkl"), "wb") as pkl_file:
#     pickle.dump(embedding_matrix, pkl_file)
# # 转换数据为索引
# transformed_data = preprocessor.transform_to_indices(data)
#
# # 实例化 ComparePaperDataset 类
# dataset = ComparePaperDataset(transformed_data, padding_idx=0, max_abstract_length=100)
#
# # 打印数据集大小
# print("Dataset size:", len(dataset))
#
# # 获取第一个数据样本
# sample = dataset[100]
# print("Sample 1 - Abstract 1:", sample["citing_paper_abstract"])
# print("Sample 1 - Abstract 1 Length:", sample["citing_paper_abstract_length"])
# print("Sample 1 - Abstract 2:", sample["cited_paper_abstract"])
# print("Sample 1 - Abstract 2 Length:", sample["cited_paper_abstract_length"])
# print("Sample 1 - Label:", sample["label"])