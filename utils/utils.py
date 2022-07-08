import torch
import torchvision

from utils.transforms import train_transform, eval_transform
from utils.consts import CLASSES


def label_to_class(label):
    return CLASSES[label]


def get_targets(dataset):
    """
    Returns targets tensor of a dataset (GTSRB dataset does not have targets attribute)
    """
    targets = torch.zeros(len(dataset))

    for i in range(len(dataset)):
        _, label = dataset[i]
        targets[i] = label

    return targets


def get_train_dataset():
    return torchvision.datasets.GTSRB(
        root='./data', split='train', download=True, transform=train_transform)


def get_train_dataset_no_transforms():
    return torchvision.datasets.GTSRB(
        root='./data', split='train', download=True)


def get_eval_dataset():
    return torchvision.datasets.GTSRB(
        root='./data', split='test', download=True, transform=eval_transform)


def get_loaders(batch_size=128):
    generator = torch.Generator().manual_seed(42)

    train_dataset = get_train_dataset()
    eval_dataset = get_eval_dataset()

    val_dataset, test_dataset = torch.utils.data.random_split(
        eval_dataset,
        [len(eval_dataset)//2, len(eval_dataset)//2], generator=generator)

    targets = get_targets(train_dataset)
    _, counts = targets.unique(return_counts=True)

    # dataset is imbalanced, so WeightedRandomSampler is used to mitigate that imbalance
    weights = 1 / counts.float()
    sample_weights = weights[targets.long()]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
