from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):

    def __init__(self, n_filters, p=0.0):
        super().__init__()

        self.n_filters = n_filters
        self.p = p

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=n_filters,
                               out_channels=n_filters, kernel_size=3)

        # convolution layers produce tensor of shape: n_filters*6*6
        self.f1 = nn.Linear(in_features=n_filters*6*6, out_features=100)
        # 43 classes
        self.f2 = nn.Linear(in_features=100, out_features=43)

        self.dropout = nn.Dropout(p=self.p)

    def forward(self, x):
        x = self._convolve(x)
        x = self._classify(x)
        return x

    def _convolve(self, x):
        # first convolutional block
        # 1@32x32 -> n_filters@30x30 -> n_filters@15x15
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # second convolutional block
        # n_filters@15x15 -> n_filters@13x13 -> n_filters@6x6
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Output shape: n_filters*6*6
        x = nn.Flatten()(x)
        return x

    def _classify(self, x):

        if self.p > 0:
            x = self.dropout(x)
        x = self.f1(x)
        x = F.relu(x)

        if self.p > 0:
            x = self.dropout(x)
        x = self.f2(x)

        # model returns logits (no softmax at the end) so CrossEntropyLoss should be used
        return x
