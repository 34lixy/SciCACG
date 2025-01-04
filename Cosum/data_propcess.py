"""
Preprocess the compare paper dataset and word embeddings to be used by the model model.
"""

import os
import pickle
import argparse
import fnmatch
import json
import random
from data import Preprocessor
from sklearn.model_selection import train_test_split


def PreprocessDataset(inputdir,
                      targetdir,
                      lowercase=True,
                      ignore_punctuation=True,
                      num_words=None,
                      stopwords=[],
                      labeldict={}):
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # -------------------- Shuffle dataset file -------------------- #
    print(20 * "=", " Shuffle dataset file ", 20 * "=")
    data_file1 = 'data/total.jsonl'
    # with open(data_file1, 'r', encoding='utf-8') as f:
    #     data = [json.loads(line) for line in f]
    with open(data_file1, 'r', encoding='utf-8') as f:
         train_data = [json.loads(line) for line in f]

    data_file2 = 'data/cs.jsonl'

    with open(data_file2, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    # random.shuffle(data)
    # 然后，将索引划分为训练集和临时集 (test + validation)
    # train_data, test_data = train_test_split(data, test_size=0.05)

    print("\t* Retrieve the train, dev and test data files... ")
    with open(os.path.join(inputdir, "train.jsonl"), "w", encoding="utf-8") as train_file:
        for item in train_data:
            json.dump(item, train_file, ensure_ascii=False)
            train_file.write("\n")

    with open(os.path.join(inputdir, "test.jsonl"), "w", encoding="utf-8") as test_file:
        for item in test_data:
            json.dump(item, test_file, ensure_ascii=False)
            test_file.write("\n")

    with open(os.path.join(inputdir, "val.jsonl"), "w", encoding="utf-8") as val_file:
        for item in test_data:
            json.dump(item, val_file, ensure_ascii=False)
            val_file.write("\n")

    train_file = "train.jsonl"
    test_file = "test.jsonl"
    dev_file = "val.jsonl"

    # -------------------- Word Dictionary Building -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words,
                                stopwords=stopwords,
                                labeldict=labeldict)

    # -------------------- Worddict Buidling -------------------- #
    print(20 * "=", " Worddict Buidling ", 20 * "=")
    data = preprocessor.read_data(data_file1)

    preprocessor.build_worddict(data)

    print("\t* word_dict_length:", len(preprocessor.worddict))
    print("\t* reverse_length:", len(preprocessor.reverse_worddict))

    with open(os.path.join(targetdir, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)
    with open(os.path.join(targetdir, "reverse_worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.reverse_worddict, pkl_file)

    # -------------------- Train data preprocessing -------------------- #
    print(20 * "=", " Preprocessing train Dataset ", 20 * "=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, train_file))

    print("\t* Transforming words to indices...")
    transformed_data = preprocessor.transform_to_indices(data)

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20 * "=", " Preprocessing dev Dataset ", 20 * "=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, dev_file))

    print("\t* Transforming words to indices...")
    transformed_data = preprocessor.transform_to_indices(data)

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20 * "=", " Preprocessing test Dataset ", 20 * "=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, test_file))

    print("\t* Transforming words to indices...")
    transformed_data = preprocessor.transform_to_indices(data)

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "test_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)


if __name__ == "__main__":
    default_config = "config.json"

    parser = argparse.ArgumentParser(description="Preprocess the compare paper dataset")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing compare paper"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    PreprocessDataset(os.path.normpath(os.path.join(script_dir, config["data_dir"])),
                      os.path.normpath(os.path.join(script_dir, config["target_dir"])),
                      lowercase=config["lowercase"],
                      ignore_punctuation=config["ignore_punctuation"],
                      num_words=config["num_words"],
                      stopwords=config["stopwords"],
                      labeldict=config["labeldict"])
