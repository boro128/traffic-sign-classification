import argparse
import torch

from torch import nn
from torch import optim
from sklearn.metrics import accuracy_score
from time import perf_counter
from datetime import timedelta

from models import CNN
from models import ModelWrapper
from utils import get_targets, get_loaders, get_train_dataset, get_train_dataset_no_transforms
from utils.plots import plot_targets_distribution, plot_sample_transforms, plot_confusion_matrix


def main():
    start_time = perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs-num', default=100, type=int)
    parser.add_argument('--n_filters', default=10, type=int)
    parser.add_argument('-p', default=0.5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--writer-filename',
                        default='default_writer', type=str)
    parser.add_argument('--writer-subdirectory', type=str)
    parser.add_argument('--produce-plots', action='store_true')
    parser.add_argument('--save-dir', type=str)

    args = parser.parse_args()

    epochs_num = args.epochs_num
    n_filters = args.n_filters
    p = args.p
    lr = args.lr
    writer_filename = args.writer_filename
    writer_subdirectory = args.writer_subdirectory
    produce_plots = args.produce_plots
    save_dir = args.save_dir

    if produce_plots:
        train_dataset = get_train_dataset()
        train_dataset_no_transforms = get_train_dataset_no_transforms()

        targets = get_targets(train_dataset)
        plot_targets_distribution(targets)

        plot_sample_transforms(train_dataset, train_dataset_no_transforms)

    #### setup ####

    train_loader, val_loader, test_loader = get_loaders()

    model = CNN(n_filters=n_filters, p=p)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mw = ModelWrapper(model, criterion, optimizer)
    mw.set_dataloaders(train_loader, val_loader)
    mw.set_writer(writer_filename, writer_subdirectory)

    #### training ####

    mw.train(epochs_num)

    if save_dir is not None:
        mw.save(save_dir)

    #### evaluation ####

    y_true = torch.empty(0)
    y_hat = torch.empty(0)

    for x, y in test_loader:
        y_true = torch.cat([y_true, y])

        preds = torch.argmax(mw.predict(x), dim=1)
        y_hat = torch.cat([y_hat, preds])

    print(f"Accuracy: {accuracy_score(y_true, y_hat)}")

    if produce_plots:
        mw.plot_loss()
        plot_confusion_matrix(y_true, y_hat)

    finish_time = perf_counter()
    difference = timedelta(seconds=finish_time-start_time)
    print(f"Execution time: {difference}")


if __name__ == '__main__':
    main()
