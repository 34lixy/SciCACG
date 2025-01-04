import os
import argparse
import pickle
import torch
import json

import torch.nn as nn
from torch.utils.data import DataLoader
from data import ComparePaperDataset
from model.model import ESCG
from train import train, validate
from test import test
from until import plot,TripletLoss,ContrastiveLoss

def main(train_file,
         valid_file,
         target_dir,
         vocab_size,
         emb_dim=64,
         hidden_size=32,
         dropout=0.5,
         epochs=64,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    with open(train_file, "rb") as pkl:
        train_data = ComparePaperDataset(pickle.load(pkl))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    print("\t* Loading validation data...")
    with open(valid_file, "rb") as pkl:
        valid_data = ComparePaperDataset(pickle.load(pkl))

    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = ESCG(vocab_size,
                 emb_dim,
                 hidden_size,
                 device,
                 dropout).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    start_epoch = 1
    best_acc = 0
    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []
    epoch_time, epoch_loss, epoch_acc, acc, rouge_score, pyrouge_score = validate(model, valid_loader, criterion)

    print("-> Valid. time: {:.4f}s, loss: {:.4f},accuracy: {:.4f},accuracy: {:.4f}".format(epoch_time, epoch_loss,
                                                                                           epoch_acc, (acc * 100)))
    print(f'ROUGE-1 Score: {rouge_score["rouge-1"]["f"] * 100:.2f}',
          f'ROUGE-2 Score: {rouge_score["rouge-2"]["f"] * 100:.2f}',
          f'ROUGE-L Score: {rouge_score["rouge-l"]["f"] * 100:.2f}')
    print(f'ROUGE-1 Score: {pyrouge_score["rouge-1"]["f"] * 100:.2f}',
          f'ROUGE-2 Score: {pyrouge_score["rouge-2"]["f"] * 100:.2f}',
          f'ROUGE-L Score: {pyrouge_score["rouge-l"]["f"] * 100:.2f}')
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training ESCG model on device: {}".format(device), 20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_acc = train(model,
                                                  train_loader,
                                                  criterion,
                                                  optimizer,
                                                  max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f},accuracy: {:.4f}".format(epoch_time, epoch_loss, epoch_acc))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_acc, acc,rouge_score, pyrouge_score = validate(model, valid_loader, criterion)

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f},accuracy: {:.4f},accuracy: {:.4f}".format(epoch_time, epoch_loss, epoch_acc,(acc*100)))
        print(f'ROUGE-1 Score: {rouge_score["rouge-1"]["f"] * 100:.2f}',
              f'ROUGE-2 Score: {rouge_score["rouge-2"]["f"] * 100:.2f}',
              f'ROUGE-L Score: {rouge_score["rouge-l"]["f"] * 100:.2f}')
        print(f'ROUGE-1 Score: {pyrouge_score["rouge-1"]["f"] * 100:.2f}',
              f'ROUGE-2 Score: {pyrouge_score["rouge-2"]["f"] * 100:.2f}',
              f'ROUGE-L Score: {pyrouge_score["rouge-l"]["f"] * 100:.2f}')

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_loss)

        # Early stopping on validation accuracy.
        if epoch_acc < best_acc:
            patience_counter += 1
        else:
            best_score = epoch_acc
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


    checkpoint = torch.load("checkpoints/best.pth.tar")
    model.load_state_dict(checkpoint["model"])
    print(checkpoint['best_score'])
    print(20 * "=",
          " Testing model model on device: {} ".format(device),
          20 * "=")

    print("\t* Loading test data...")
    with open("preprocessed_data/test_data.pkl", "rb") as pkl:
        test_data = ComparePaperDataset(pickle.load(pkl))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    batch_time, total_time, rouge_score, pyrouge_score = test(model, test_loader)

    print("-> Average batch processing time: {:.4f}s, total test time:\
     {:.4f}s".format(batch_time, total_time))
    print(f'ROUGE-1 Score: {rouge_score["rouge-1"]["f"] * 100:.2f}',
          f'ROUGE-2 Score: {rouge_score["rouge-2"]["f"] * 100:.2f}',
          f'ROUGE-L Score: {rouge_score["rouge-l"]["f"] * 100:.2f}')
    print(f'ROUGE-1 Score: {pyrouge_score["rouge-1"]["f"] * 100:.2f}',
          f'ROUGE-2 Score: {pyrouge_score["rouge-2"]["f"] * 100:.2f}',
          f'ROUGE-L Score: {pyrouge_score["rouge-l"]["f"] * 100:.2f}')

    # Plotting of the loss curves for the train and validation sets.
    plot(epochs_count, train_losses, valid_losses)


if __name__ == "__main__":
    default_config = 'config.json'

    parser = argparse.ArgumentParser(description="Train the model ")
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

    main(os.path.normpath(os.path.join(script_dir, config["train_data"])),
         os.path.normpath(os.path.join(script_dir, config["valid_data"])),
         os.path.normpath(os.path.join(script_dir, config["check_point_dir"])),
         config["vocab_size"],
         config["emb_dim"],
         config["hidden_size"],
         config["dropout"],
         config["epochs"],
         config["batch_size"],
         config["lr"],
         config["patience"],
         config["max_gradient_norm"])
