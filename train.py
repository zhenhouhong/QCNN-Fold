'''
File: Contains the code to train and test learning real-valued distances, binned-distances and contact maps with QCNN model.
'''

import os
import sys
import numpy as np
import datetime
import argparse
import torch

flag_plots = False

if flag_plots:
    #%matplotlib inline
    from plots import *

if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True


def get_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            epilog='EXAMPLE:\npython3 train.py -w distance.hdf5 -n 200 -c 64 -e 2 -d 8 -f 16 -p ../dl-training-data/ -o /tmp/')
    parser.add_argument('-w', type=str, required = True, dest = 'file_weights', help="hdf5 weights file")
    parser.add_argument('-n', type=int, required = True, dest = 'dev_size', help="number of pdbs to use for training (use -1 for ALL)")
    parser.add_argument('-c', type=int, required = True, dest = 'training_window', help="crop size (window) for training, 64, 128, etc. ")
    parser.add_argument('-e', type=int, required = True, dest = 'training_epochs', help="# of epochs")
    parser.add_argument('-o', type=str, required = True, dest = 'dir_out', help="directory to write .npy files")
    parser.add_argument('-d', type=int, required = True, dest = 'arch_depth', help="residual arch depth")
    parser.add_argument('-f', type=int, required = True, dest = 'filters_per_layer', help="number of convolutional filters in each layer")
    parser.add_argument('-p', type=str, required = True, dest = 'dir_dataset', help="path where all the data (including .lst) is located")
    args = parser.parse_args()
    return args

args = get_args()

file_weights = args.file_weights
dev_size = args.dev_size
training_window = args.training_window
training_epochs = args.training_epochs
arch_depth = args.arch_depth
filters_per_layer = args.filters_per_layer
dir_dataset = args.dir_dataset
dir_out = args.dir_out
pad_size = 10
batch_size = 2
expected_n_channels = 57
learning_rate = 0.01
log_interval = 20

# Import after argparse because this can throw warnings with "-h" option
from dataio import *
from metrics import *
from generator import *
from qmodels import *
from losses import *

print('Start ' + str(datetime.datetime.now()))

print('')
print('Parameters:')
print('dev_size', dev_size)
print('file_weights', file_weights)
print('training_window', training_window)
print('training_epochs', training_epochs)
print('arch_depth', arch_depth)
print('filters_per_layer', filters_per_layer)
print('pad_size', pad_size)
print('batch_size', batch_size)
print('dir_dataset', dir_dataset)
print('dir_out', dir_out)

os.system('mkdir -p ' + dir_out)

all_feat_paths = [dir_dataset + '/deepcov/features/', dir_dataset + '/psicov/features/', dir_dataset + '/cameo/features/']
all_dist_paths = [dir_dataset + '/deepcov/distance/', dir_dataset + '/psicov/distance/', dir_dataset + '/cameo/distance/']

deepcov_list = load_list(dir_dataset + '/deepcov.lst', dev_size)

length_dict = {}
for pdb in deepcov_list:
    (ly, seqy, cb_map) = np.load(dir_dataset + '/deepcov/distance/' + pdb + '-cb.npy', allow_pickle = True)
    length_dict[pdb] = ly

print('')
print('Split into training and validation set..')
valid_pdbs = deepcov_list[:int(0.3 * len(deepcov_list))]
train_pdbs = deepcov_list[int(0.3 * len(deepcov_list)):]
if len(deepcov_list) > 200:
    valid_pdbs = deepcov_list[:100]
    train_pdbs = deepcov_list[100:]

print('Total validation proteins : ', len(valid_pdbs))
print('Total training proteins   : ', len(train_pdbs))

print('')
print('Validation proteins: ', valid_pdbs)

train_generator = DistGenerator(train_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, batch_size, expected_n_channels, label_engineering = '16.0')
valid_generator = DistGenerator(valid_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, batch_size, expected_n_channels, label_engineering = '16.0')

print('')
print('len(train_generator) : ' + str(len(train_generator)))
print('len(valid_generator) : ' + str(len(valid_generator)))

X, Y = train_generator[1]
print('Actual shape of X    : ' + str(X.shape))
print('Actual shape of Y    : ' + str(Y.shape))

print('')
print('Channel summaries:')
summarize_channels(X[0, :, :, :], Y[0])

if flag_plots:
    print('')
    print('Inputs/Output of protein', 0)
    plot_protein_io(X[0, :, :, :], Y[0, :, :, 0])

print('')
print('Build a model..')
model = ''
model = QCNN_RDD_Distances(arch_depth, filters_per_layer, expected_n_channels, batch_size)
if use_cuda:
    model.cuda()

optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

if os.path.exists(file_weights):
    print('')
    print('Loading existing weights..')
    model.load_weights(file_weights)


def train(e):

    model.train()

    for batch_idx, (X, Y) in enumerate(train_generator):

        X = torch.Tensor(X).transpose(1, 3)
        Y = torch.Tensor(Y).transpose(1, 3)
        if use_cuda:
            X.cuda()
            Y.cuda()

        output = model(X)
        train_loss = in_log_cosh(output, Y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {}, Train Loss: {:6.f}".format(e, train_loss.item()))


def valid(e):

    model.eval()

    for batch_idx, (X, Y) in enumerate(valid_generator):

        X = torch.Tensor(X).transpose(1, 3)
        Y = torch.Tensor(Y).transpose(1, 3)
        if use_cuda:
            X.cuda()
            Y.cuda()

        output = model(X)
        valid_loss = in_log_cosh(output, Y)

    print("Valid Epoch: {}, Valid Loss: {:6.f}".format(e, valid_loss.item()))


if __name__ == "__main__":

    for e in range(training_epochs):
        train(e)
        valid(e)
