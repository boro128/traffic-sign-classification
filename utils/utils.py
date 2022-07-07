import torch


def get_targets(dataset):
    """
    Returns targets tensor of a dataset (GTSRB dataset does not have targets attribute)
    """
    targets = torch.zeros(len(dataset))

    for i in range(len(dataset)):
        _, label = dataset[i]
        targets[i] = label

    return targets
