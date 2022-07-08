import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix


sns.set()


def plot_targets_distribution(targets, path="images/target_distribution.png", show=False):
    classes, counts = targets.unique(return_counts=True)

    plt.figure(figsize=(8, 6))

    plt.bar(x=classes.int(), height=counts)
    plt.title("Distribution of Target Classes", fontsize=14)
    plt.xlabel("Class")
    plt.ylabel("Counts")
    plt.savefig(path)

    if show:
        plt.show()


def plot_sample_transforms(dataset, dataset_no_transform, path="images/sample_transforms.png",
                           im1=17000, im2=6000, im3=10000, show=False):

    regular = [dataset_no_transform[im1][0],
               dataset_no_transform[im2][0], dataset_no_transform[im3][0]]
    transformed = [dataset[im1][0], dataset[im2][0], dataset[im3][0]]

    to_img = transforms.ToPILImage()
    for idx, e in enumerate(transformed):
        transformed[idx] = to_img(e)

    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(10, 6)

    for row in ax:
        for axis in row:
            axis.grid(False)
            axis.set_xticks([])
            axis.set_yticks([])

    ax[0, 0].imshow(regular[0])
    ax[0, 1].imshow(regular[1])
    ax[0, 2].imshow(regular[2])

    ax[1, 0].imshow(transformed[0], cmap='gray')
    ax[1, 1].imshow(transformed[1], cmap='gray')
    ax[1, 2].imshow(transformed[2], cmap='gray')

    plt.savefig(path)

    if show:
        plt.show()


def plot_confusion_matrix(y_true, y_hat, path="images/confusion_matrix.png", show=False):
    cm = confusion_matrix(y_true.int(), y_hat.int())

    plt.figure(figsize=(10, 10))

    plt.imshow(cm, cmap='inferno')

    plt.title("Confusion matrix", fontsize=20)
    plt.xlabel("Predicted class", fontsize=14)
    plt.ylabel("True class", fontsize=14)

    plt.savefig(path)

    if show:
        plt.show()
