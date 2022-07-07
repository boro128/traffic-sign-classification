import torch
import numpy as np


class ModelWrapper():

    def __init__(self, model, criterion, optimizer) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

    def train(self, epochs_num) -> None:
        for epoch in range(epochs_num):
            train_loss = self._mini_batch()
            self.losses.append(train_loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            print(
                f"epoch: {epoch}    loss: {train_loss}   val_loss: {val_loss}")

            self.total_epochs += 1

    def predict(self, x):
        self.model.eval()
        return self.model(x)

    def set_dataloaders(self, train_loader, val_loader) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _mini_batch(self, validation=False):
        if validation:
            self.model.eval()
            loader = self.val_loader
            step = self._perform_val_step
        else:
            self.model.train()
            loader = self.train_loader
            step = self._perform_train_step

        mini_batch_losses = []

        for x_batch, y_batch in loader:
            loss = step(x_batch, y_batch)
            mini_batch_losses.append(loss)

        return np.mean(mini_batch_losses)

    def _perform_train_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        y_hat = self.model(y)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _perform_val_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(y)
        loss = self.criterion(y_hat, y)

        return loss.item()
