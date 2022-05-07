'''
File: Contains the code of QCNN models.
'''

import torch
import torch.nn as nn

from qconv import QCONV2d


# A basic QCNN network
class Basic_QCNN(nn.Module):
    def __init__(self,
            num_blocks,
            width,
            expected_n_channels,
            batch_size):
        super(Basic_QCNN, self).__init__()

        self.num_blocks = num_blocks
        self.width = width
        self.expected_n_channels = expected_n_channels
        self.batch_size = batch_size

        self.qconv_list = []
        self.bn_list = []
        for _ in range(self.num_blocks):
            self.qconv_list.append(QCONV2d(width, 3, padding="same", batch_size=self.batch_size))
            self.bn_list.append(nn.BatchNorm(width))

        self.conv = nn.Conv2d(width, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x
        for i in range(self.num_blocks):
            x = self.qconv_list[i](x)
            x = self.bn_list(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.relu(x)

        return x


# Architecture QCNN_RDD (original)
class QCNN_RDD(nn.Module):
    def __init__(self, num_blocks, width, expected_n_channels, batch_size):
        super(QCNN_RDD, self).__init__()

        self.num_blocks = num_blocks
        self.width = width
        self.expected_n_channels = expected_n_channels
        self.batch_size = batch_size
        self.dropout_value = 0.3
        self.d_rate = 1

        self.bn1 = nn.BatchNorm2d(expected_n_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(expected_n_channels, width, 3, padding=1)
        self.dropout = nn.Dropout(self.dropout_value)

        self.bn2 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, 3, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.qconv_list = []
        self.dconv_list = []
        self.bn_list = []
        for _ in range(self.num_blocks):
            self.qconv_list.append(QCONV2d(width, 3, padding="same", batch_size=self.batch_size))
            self.dconv_list.append(nn.Conv2d(width, width, 3, padding=1, dilation=self.d_rate))
            if self.d_rate == 1:
                self.d_rate = 2
            elif self.d_rate == 2:
                self.d_rate = 4
            else:
                self.d_rate = 1
            self.bn_list.append(nn.BatchNorm2d(width))

    def forward(self, x):

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        for i in range(self.num_blocks):
            x_in = self.bn_list[i](x)
            x_in = self.relu(x_in)
            x_in = self.qconv_list[i](x_in)
            x_in = self.dropout(x_in)
            x_in = self.relu(x_in)
            x_in = self.dconv_list[i](x_in)
            x = x_in + x

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        return x


# Architecture QCNN_RDD_Distances
class QCNN_RDD_Distances(nn.Module):
    def __init__(self, num_blocks, width, expected_n_channels, batch_size):
        super(QCNN_RDD_Distances, self).__init__()

        self.num_blocks = num_blocks
        self.width = width
        self.expected_n_channels = expected_n_channels
        self.batch_size = batch_size
        self.dropout_value = 0.3
        self.d_rate = 1

        self.bn1 = nn.BatchNorm2d(expected_n_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(expected_n_channels, width, 3, padding=1)
        self.dropout = nn.Dropout(self.dropout_value)

        self.bn2 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, 1, 3, padding=1)

        self.qconv_list = []
        self.dconv_list = []
        self.bn_list = []
        for _ in range(self.num_blocks):
            self.qconv_list.append(QCONV2d(width, 3, padding="same", batch_size=self.batch_size))
            self.dconv_list.append(nn.Conv2d(width, width, 3, padding=1, dilation=self.d_rate))
            if self.d_rate == 1:
                self.d_rate = 2
            elif self.d_rate == 2:
                self.d_rate = 4
            else:
                self.d_rate = 1
            self.bn_list.append(nn.BatchNorm2d(width))

    def forward(self, x):

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        for i in range(self.num_blocks):
            x_in = self.bn_list[i](x)
            x_in = self.relu(x_in)
            x_in = self.qconv_list[i](x_in)
            x_in = self.dropout(x_in)
            x_in = self.relu(x_in)
            x_in = self.dconv_list[i](x_in)
            x = x_in + x

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x

