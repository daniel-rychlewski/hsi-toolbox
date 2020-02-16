import torch

import torch.nn as nn
import torch.nn.functional as F

# justification for choice: taken from comparison https://paperswithcode.com/sota/hyperspectral-image-classification-on-indian

"""
HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification
https://arxiv.org/pdf/1902.06701v2.pdf
https://github.com/gokriznastic/HybridSN"""
class Roy19(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, n_bands, n_classes, window_size):
        super(Roy19, self).__init__()
        self.S = window_size
        self.L = n_bands
        self.output_units = n_classes

        ## convolutional layers
        self.conv_layer1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 7))
        self.conv_layer2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 5))
        self.conv_layer3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))

        # calculate dimensions for flattened layers
        self.features_size = self._get_final_flattened_size()

        # fully connected layers - lots of VRAM (like 12GB) required, 5GB of which for the following line
        self.dense_layer1 = nn.Linear(in_features=self.features_size, out_features=256)
        self.dense_dropout1 = nn.Dropout(0.4)
        self.dense_layer2 = nn.Linear(in_features=256, out_features=128)
        self.dense_dropout2 = nn.Dropout(0.4)
        self.output_layer = nn.Linear(in_features=128, out_features=self.output_units)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.L,
                             self.S, self.S))
            x = self.conv_layer1(x)
            x = self.conv_layer2(x)
            x = self.conv_layer3(x)
            x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4])
            x = self.conv_layer4(x)

            t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv_layer1(x))
        x = F.relu(self.conv_layer2(x))
        x = F.relu(self.conv_layer3(x))
        x = x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4])
        x = F.relu(self.conv_layer4(x))
        x = x.view(-1, self.features_size)

        x = F.relu(self.dense_layer1(x))
        x = self.dense_dropout1(x)
        x = F.relu(self.dense_layer2(x))
        x = self.dense_dropout2(x)

        x = self.output_layer(x)
        return x

"""
BASS Net: Band-Adaptive Spectral-Spatial Feature Learning Neural Network for Hyperspectral Image Classification
https://arxiv.org/pdf/1612.00144v2.pdf
https://github.com/hbutsuak95/BASS-Net
"""
class Santara16(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, n_channels, block1_conv1, n_bands, patch_size, n_classes):
        super(Santara16, self).__init__()
        self.n_channels = n_channels
        self.block1_conv1 = block1_conv1 # number of filters in the 1*1 convolution
        self.n_bands = n_bands
        self.band_size = self.block1_conv1 // self.n_bands
        self.patch_size = patch_size
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.block1_conv1, kernel_size=(1,1))

        """ now construct a list of parallel models, one for every band
        weight sharing done automatically in PyTorch, unlike Torch (Lua), where the share method is necessary
        Just use the same model on different outputs: https://discuss.pytorch.org/t/how-to-share-weights-between-two-nets/4319 """
        self.parallel1 = nn.Conv1d(in_channels=self.patch_size*self.patch_size, out_channels=20, kernel_size=3)
        self.parallel2 = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3)
        self.parallel3 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3)
        self.parallel4 = nn.Conv1d(in_channels=10, out_channels=5, kernel_size=5)

        self.linear1 = nn.Linear(in_features=self.n_bands*(self.band_size-10)*5, out_features=100)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(in_features=100, out_features=self.n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        if len(x.shape) != 5:
            x = torch.unsqueeze(x, dim=1)
        x = x.reshape(shape=(-1, self.n_channels, self.patch_size, self.patch_size))
        x = F.relu(self.conv1(x))
        x = x.view(x.shape[0], self.n_bands, self.band_size, self.patch_size*self.patch_size)

        all_tensors = []

        # tensors = torch.chunk(x, 1, dim=0)  # no split
        # tensors = torch.chunk(x, 2, dim=0)  # split into two tensors
        tensors = torch.split(x, 1, dim=1) # this is how many parallel nets will be traversed, decided by n_bands

        all_tensors.extend(tensors)

        tensors_to_merge = []
        for tensor in all_tensors:
            tensor = tensor.view(-1, self.patch_size * self.patch_size, self.band_size)
            tensor = F.relu(self.parallel1(tensor))
            tensor = F.relu(self.parallel2(tensor))
            tensor = F.relu(self.parallel3(tensor))
            tensor = F.relu(self.parallel4(tensor))
            tensor = tensor.view(-1,(self.band_size-10)*5)
            tensors_to_merge.append(tensor)

        x = torch.cat(tensors_to_merge, dim=1)

        x = x.reshape(shape=(-1, self.n_bands * (self.band_size - 10) * 5))
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.linear2(x))

        return x

"""
Hyperspectral Image Classification with Markov Random Fields and a Convolutional Neural Network
https://arxiv.org/pdf/1705.00727v2.pdf
https://github.com/xiangyongcao/CNN_HSIC_MRF
"""
# padding does not matter
class Cao17(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, patch_size, num_band, num_classes):
        super(Cao17, self).__init__()

        self.patch_size = patch_size
        self.num_band = num_band
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=self.num_band, out_channels=300, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=300, out_channels=200, kernel_size=3)

        if torch.__version__ == '1.2.0':
            print("warning: cao used kernel size 1 due to pytorch 1.2.0, please downgrade to pytorch 1.1.0\n")
            self.pool2 = nn.MaxPool2d(kernel_size=1)
        else: # especially torch 1.1
            self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(in_features=self.features_size, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=self.num_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.num_band, self.patch_size, self.patch_size))
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = x.flatten()
        return list(x.size())[0]

    def forward(self, x):
        x = x.reshape(shape=(-1, self.num_band, self.patch_size, self.patch_size))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten()

        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

"""
Shorten Spatial-spectral RNN with Parallel-GRU (St-SS-pGRU)
https://arxiv.org/pdf/1810.12563v1.pdf
https://github.com/codeRimoe/DL_for_RSIs/tree/master/St-SS-pGRU
"""
# SS==ST==1, padding does not matter
class LuoRNN(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, std=0.1)
            nn.init.constant_(m.bias, 0.1)

    def __init__(self, n_bands, window_size, n_filter, m_filter, hidden_size, n_time, n_classes, batch_size):
        super(LuoRNN, self).__init__()
        self.r_win = window_size
        self.n_bands = n_bands
        self.n_filter = n_filter
        self.m_filter = m_filter
        self.hidden_size = hidden_size
        self.n_time = n_time
        self.n_classes = n_classes
        self.batch_size = batch_size

        n_same = n_bands % n_time
        self.n_f = n_bands // n_time + n_same
        self.n_same = self.n_f - n_same

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.n_filter, kernel_size=(1, 5, 5), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=1, out_channels=self.n_filter, kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,1,1))
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=self.n_filter, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1,5,5), stride=(1,1,1))

        self.conv4 = nn.Conv1d(in_channels=m_filter, out_channels=hidden_size, kernel_size=self.n_f, stride=n_same)

        self.lstm = nn.LSTMCell(hidden_size, batch_size)

    def forward(self, x):
        # SS part
        x = x.reshape(shape=(-1, 1, self.n_bands, self.r_win, self.r_win))

        x1 = F.relu(self.conv1(x))
        x2 = self.pool1(F.relu(self.conv2(x)))
        x3 = self.pool2(F.relu(self.conv3(x)))

        x = x1 + x2 + x3
        x = x.reshape(shape=(-1, self.m_filter, self.n_f))

        # ST part
        x = F.relu(self.conv4(x))
        x = x.squeeze()

        x, hidden = self.lstm(x)

        # x = x.view(self.n_classes, -1)

        return x