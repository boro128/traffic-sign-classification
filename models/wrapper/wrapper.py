import torch
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class ModelWrapper():

    def __init__(self, model, criterion, optimizer) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.writer = None

        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

    def train(self, epochs_to_train, seed=42) -> None:
        self.set_seed(seed)

        start_epoch = self.total_epochs
        final_epoch = self.total_epochs + epochs_to_train
        for epoch in range(start_epoch, final_epoch):
            train_loss = self._mini_batch()
            self.losses.append(train_loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            print(
                f"epoch: {epoch}    loss: {train_loss}   val_loss: {val_loss}")

            if self.writer is not None:
                self.writer.add_scalars(
                    main_tag='loss',
                    global_step=epoch,
                    tag_scalar_dict={
                        'training': train_loss,
                        'validation': val_loss
                    }
                )
                self.writer.flush()

            self.total_epochs += 1

        if self.writer is not None:
            self.writer.close()

    def predict(self, x):
        self.model.eval()
        x = torch.as_tensor(x).float()
        x = x.to(self.device)
        y_hat = self.model(x)
        return y_hat.detach().cpu()

    def save(self, path):
        checkpoint = {
            'total_epochs': self.total_epochs,
            'losses': self.losses,
            'val_losses': self.val_losses,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)

        self.total_epochs = checkpoint['total_epochs']
        self.losses = checkpoint['losses']
        self.val_losses = checkpoint['val_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def set_dataloaders(self, train_loader, val_loader) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_writer(self, filename, subdirectory=None) -> None:
        timestamp = datetime.now().strftime(r'%Y%m%d%H%M%S')
        if subdirectory is not None:
            parent_dir = f'runs/{subdirectory}'
        else:
            parent_dir = 'runs'
        self.writer = SummaryWriter(f'{parent_dir}/{filename}_{timestamp}')

    def set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

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

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _perform_val_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        return loss.item()
