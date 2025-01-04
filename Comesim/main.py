import argparse
import json
import os
import train


if __name__ == "__main__":
    default_config = 'config.json'

    parser = argparse.ArgumentParser(description="Train the model model on Comparepaper")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to a checkpoint file to resume training")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    train.main(os.path.normpath(os.path.join(script_dir, config["train_data"])),
               os.path.normpath(os.path.join(script_dir, config["valid_data"])),
               os.path.normpath(os.path.join(script_dir, config["embeddings"])),
               os.path.normpath(os.path.join(script_dir, config["check_point_dir"])),
               config["vocab_size"],
               config["emd_dim"],
               config["hidden_size"],
               config["dropout"],
               config["num_classes"],
               config["epochs"],
               config["batch_size"],
               config["lr"],
               config["patience"],
               config["max_gradient_norm"],
               args.checkpoint)


