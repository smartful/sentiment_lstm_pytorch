import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

def accuracy_fn(y_pred, y_true):
    """
    Calculates accuracy between predictions and truth labels.
    """
    try:
        if (len(y_pred) != len(y_true)):
            raise ValueError("Size Error")
    except ValueError:
        print("y_pred and y_true have not the same size !")
    else:
        size = len(y_pred)
        correct = torch.eq(y_pred, y_true).sum().item()
        acc = (correct / size) * 100
        return acc


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    train_loss, train_acc = 0, 0

    for batch, (x, seq_lenght, y) in enumerate(dataloader):
        model.train()
        # Envoyer les data au GPU
        x, y = x.to(device), y.to(device)

        y_pred_logits = model(x, seq_lenght)
        # Calcul de la loss
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss

        # Calcul de l'accuracy
        y_pred = (torch.sigmoid(y_pred_logits) > 0.5)
        accuracy = accuracy_fn(y_true=y, y_pred=y_pred)
        train_acc += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calcul de la loss/acc moyenne par batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return (train_loss, train_acc)


def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for x, seq_lenght, y in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred_logits = model(x, seq_lenght)
            # Calcul de la loss
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss

            # Calcul de l'accuracy
            y_pred = (torch.sigmoid(y_pred_logits) > 0.5)
            accuracy = accuracy_fn(y_true=y, y_pred=y_pred)
            test_acc += accuracy

        # Calcul de la loss/acc moyenne par batch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        return (test_loss, test_acc)


def plot_training_data(typology: str, train_data: list, test_data: list):
    plt.figure()
    plt.title(f"{typology}")
    plt.plot(range(len(train_data)), train_data, label=f"Training {typology}")
    plt.plot(range(len(test_data)), test_data, label=f"Test {typology}")
    plt.xlabel("epochs")
    plt.ylabel(f"{typology}")
    plt.legend()
    plt.show()