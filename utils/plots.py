import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


def plot_targets_distribution(targets, show=False):
    classes, counts = targets.unique(return_counts=True)

    plt.bar(x=classes.int(), height=counts)
    plt.title("Distribution of Target Classes", fontsize=14)
    plt.xlabel("Class")
    plt.ylabel("Counts")
    plt.savefig("images/target_distribution.png")

    if show:
        plt.show()
