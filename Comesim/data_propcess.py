"""
Preprocess the compare paper dataset and word embeddings to be used by the model model.
"""
import argparse
import os
import pickle
import json
import random
from data import Preprocessor
from sklearn.model_selection import train_test_split


def PreprocessDataset(inputdir,
                      embeddings_file,
                      targetdir,
                      lowercase=False,
                      ignore_punctuation=False,
                      num_words=None,
                      stopwords=[],
                      label_dict={},
                      bos=None,
                      eos=None):
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    data_file = 'data/ComparePaper.jsonl'

    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Shuffle the data to ensure randomness
    random.shuffle(data)

    # 然后，将索引划分为训练集和临时集 (test + validation)
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)

    # 接着，将临时集的索引划分为测试集和验证集
    val_data, test_data = train_test_split(temp_data, test_size=0.3, random_state=42)

    # Save the split data to separate JSONL files
    with open(os.path.join(inputdir, "train.jsonl"), "w", encoding="utf-8") as train_file:
        for item in train_data:
            json.dump(item, train_file, ensure_ascii=False)
            train_file.write("\n")

    with open(os.path.join(inputdir, "test.jsonl"), "w", encoding="utf-8") as test_file:
        for item in test_data:
            json.dump(item, test_file, ensure_ascii=False)
            test_file.write("\n")

    with open(os.path.join(inputdir, "val.jsonl"), "w", encoding="utf-8") as val_file:
        for item in val_data:
            json.dump(item, val_file, ensure_ascii=False)
            val_file.write("\n")

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = "train.jsonl"
    dev_file = "test.jsonl"
    test_file = "val.jsonl"

    # -------------------- Building data dictionary -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase, ignore_punctuation=ignore_punctuation, num_words=num_words,
                                stopwords=stopwords,labeldict=label_dict, bos=bos, eos=eos)
    print(20 * "=", " vocab Building ", 20 * "=")
    print("\t* Reading data...")
    data = preprocessor.read_data(data_file)

    print("\t* Computing worddict and saving it...")
    preprocessor.build_worddict(data)
    with open(os.path.join(targetdir, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)
    # -------------------- Train data preprocessing -------------------- #
    print(20 * "=", " Preprocessing train set ", 20 * "=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, train_file))

    print("\t* Transforming words in paper abstract to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20 * "=", " Preprocessing dev set ", 20 * "=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, dev_file))

    print("\t* Transforming words in paper_abstract to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20 * "=", " Preprocessing test set ", 20 * "=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, test_file))

    print("\t* Transforming words in paper abstract to indices...")
    transformed_data = preprocessor.transform_to_indices(data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "test_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20 * "=", " Preprocessing embeddings ", 20 * "=")
    print("\t* Building embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(embeddings_file)
    with open(os.path.join(targetdir, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


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
                      os.path.normpath(os.path.join(script_dir, config["embeddings_file"])),
                      os.path.normpath(os.path.join(script_dir, config["target_dir"])), lowercase=config["lowercase"],
                      ignore_punctuation=config["ignore_punctuation"], num_words=config["num_words"],
                      stopwords=config["stopwords"],label_dict=config["label_dict"], bos=config["bos"], eos=config["eos"])
