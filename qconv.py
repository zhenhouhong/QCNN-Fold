'''
File: Contains the code of qconv
'''

import torch
import torch.nn as nn

import pennylane as qml
import numpy as np

import time
import os
import sys

from qcircuit import vqc, vqc_1mout


class QCONV2d(nn.Module):
    def __init__(self,
                 output_filters,
                 kernel_size=3,
                 strides=1,
                 padding="same",
                 batch_size=1,
                 n_qlayers=1,
                 qembed_type="angle",
                 qlayer_type="basic"):
        super(QCONV2d, self).__init__()
        self.kernel_size = kernel_size
        self.output_filters = output_filters
        self.strides = strides
        self.n_qubits = kernel_size**2
        self.padding = padding
        self.batch_size = batch_size
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type

        self.VQC = [vqc_1mout(self.n_qubits, self.n_qlayers, self.qembed_type, self.qlayer_type) for _ in range(int(batch_size))]

    def forward(self, x):

        _, x_original_filters, x_original_height, x_original_width = x.shape
        h_pad_size = 0
        w_pad_size = 0
        if "same" == self.padding:
            h_pad_size = int(((x_original_height-1) * self.strides - x_original_height + (self.kernel_size-1) + 1) / 2)
            w_pad_size = int(((x_original_width-1) * self.strides - x_original_width + (self.kernel_size-1) + 1) / 2)
        pad_opt = nn.ZeroPad2d((w_pad_size, w_pad_size, h_pad_size, h_pad_size))
        x = pad_opt(x)
        x_batch_size, x_filters, x_new_height, x_new_width = x.shape

        x_output_height = int((x_original_height - (self.kernel_size-1) + 2 * h_pad_size - 1) / self.strides + 1)
        x_output_width = int((x_original_width - (self.kernel_size-1) + 2 * w_pad_size - 1) / self.strides + 1)

        x_new = torch.zeros((x_batch_size, self.output_filters, x_output_height, x_output_width))

        f_map = [(l, m) for l in range(0, x_output_height, self.strides) for m in range(0, x_output_width, self.strides)]

        vqc_index = 0
        for i in range(x_batch_size):
            for j in range(self.output_filters):
                for l, m in f_map:
                    vqc_output = self.VQC[vqc_index](
                            torch.Tensor(
                                x[i, :, l:l+self.kernel_size, m:m+self.kernel_size]).reshape(-1, self.kernel_size**2)
                            )
                    x_new[i, j, l, m] = vqc_output.sum()
            vqc_index += 1

        return x_new


class QCONV2d_1x1(nn.Module):
    def __init__(self,  
                 output_filters,
                 strides=1,
                 padding="same",
                 n_qlayers=1,
                 qembed_type="angle",
                 qlayer_type="basic"):
        super(QCONV2d_1x1, self).__init__()
        self.kernel_size = 1
        self.output_filters = output_filters
        self.strides = self.strides
        self.n_qubits = 1**2
        self.padding = padding
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type

        #self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = vqc_1mout(self.n_qubits, self.n_qlayers, self.qembed_type, self.qlayer_type)
        #self.clayer_out = torch.nn.Linear(self.n_qubits, self.output_filters)

    def forward(self, x):

        _, x_original_filters, x_original_height, x_original_width = x.shape
        h_pad_size = 0
        w_pad_size = 0
        if "same" == self.padding:
            h_pad_size = int(((x_original_height-1) * self.strides - x_original_height + (self.kernel_size-1) + 1) / 2)
            w_pad_size = int(((x_original_width-1) * self.strides - x_original_width + (self.kernel_size-1) + 1) / 2)
        pad_opt = nn.ZeroPad2d((w_pad_size, w_pad_size, h_pad_size, h_pad_size))
        x = pad_opt(x)
        x_batch_size, x_filters, x_new_height, x_new_width = x.shape

        x_output_height = int((x_original_height - (self.kernel_size-1) + 2 * h_pad_size - 1) / self.strides + 1)
        x_output_width = int((x_original_width - (self.kernel_size-1) + 2 * w_pad_size - 1) / self.strides + 1)

        x_new = torch.zeros((x_batch_size, self.output_filters, x_output_height, x_output_width))

        for i in range(x_batch_size):
            for j in range(self.output_filters):
                for k in range(x_original_filters):
                    for l in range(0, x_output_height, self.strides):
                        for m in range(0, x_output_width, self.strides):
                            x_new[i, j, l, m] = self.VQC(
                                torch.Tensor([x[i, k, l, m]])
                            )
        return x_new


class QCONV2d_2x2(nn.Module):
    def __init__(self,  
                 output_filters,
                 strides=1,
                 padding="same",
                 n_qlayers=1,
                 qembed_type="angle",
                 qlayer_type="basic"):
        super(QCONV2d_2x2, self).__init__()
        self.kernel_size = 2
        self.output_filters = output_filters
        self.strides = self.strides
        self.n_qubits = 2**2
        self.padding = padding
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type

        #self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = vqc_1mout(self.n_qubits, self.n_qlayers, self.qembed_type, self.qlayer_type)
        #self.clayer_out = torch.nn.Linear(self.n_qubits, self.output_filters)

    def forward(self, x):

        _, x_original_filters, x_original_height, x_original_width = x.shape
        h_pad_size = 0
        w_pad_size = 0
        if "same" == self.padding:
            h_pad_size = int(((x_original_height-1) * self.strides - x_original_height + (self.kernel_size-1) + 1) / 2)
            w_pad_size = int(((x_original_width-1) * self.strides - x_original_width + (self.kernel_size-1) + 1) / 2)
        pad_opt = nn.ZeroPad2d((w_pad_size, w_pad_size, h_pad_size, h_pad_size))
        x = pad_opt(x)
        x_batch_size, x_filters, x_new_height, x_new_width = x.shape

        x_output_height = int((x_original_height - (self.kernel_size-1) + 2 * h_pad_size - 1) / self.strides + 1)
        x_output_width = int((x_original_width - (self.kernel_size-1) + 2 * w_pad_size - 1) / self.strides + 1)

        x_new = torch.zeros((x_batch_size, self.output_filters, x_output_height, x_output_width))

        for i in range(x_batch_size):
            for j in range(self.output_filters):
                for k in range(x_original_filters):
                    for l in range(0, x_output_height, self.strides):
                        for m in range(0, x_output_width, self.strides):
                            x_new[i, j, l, m] = self.VQC(
                                torch.Tensor([
                                    x[i, k, l, m],
                                    x[i, k, l, m+1],
                                    x[i, k, l+1, m],
                                    x[i, k, l+1, m+1]])
                            )
        return x_new


class QCONV2d_3x3(nn.Module):
    def __init__(self,  
                 output_filters,
                 strides=1,
                 padding="same",
                 n_qlayers=1,
                 qembed_type="angle",
                 qlayer_type="basic"):
        super(QCONV2d_3x3, self).__init__()
        self.kernel_size = 3
        self.output_filters = output_filters
        self.strides = self.strides
        self.n_qubits = 3**2
        self.padding = padding
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type

        #self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = vqc_1mout(self.n_qubits, self.n_qlayers, self.qembed_type, self.qlayer_type)
        #self.clayer_out = torch.nn.Linear(self.n_qubits, self.output_filters)

    def forward(self, x):

        _, x_original_filters, x_original_height, x_original_width = x.shape
        h_pad_size = 0
        w_pad_size = 0
        if "same" == self.padding:
            h_pad_size = int(((x_original_height-1) * self.strides - x_original_height + (self.kernel_size-1) + 1) / 2)
            w_pad_size = int(((x_original_width-1) * self.strides - x_original_width + (self.kernel_size-1) + 1) / 2)
        pad_opt = nn.ZeroPad2d((w_pad_size, w_pad_size, h_pad_size, h_pad_size))
        x = pad_opt(x)
        x_batch_size, x_filters, x_new_height, x_new_width = x.shape

        x_output_height = int((x_original_height - (self.kernel_size-1) + 2 * h_pad_size - 1) / self.strides + 1)
        x_output_width = int((x_original_width - (self.kernel_size-1) + 2 * w_pad_size - 1) / self.strides + 1)

        x_new = torch.zeros((x_batch_size, self.output_filters, x_output_height, x_output_width))

        for i in range(x_batch_size):
            for j in range(self.output_filters):
                for k in range(x_original_filters):
                    for l in range(0, x_output_height, self.strides):
                        for m in range(0, x_output_width, self.strides):
                            x_new[i, j, l, m] = self.VQC(
                                torch.Tensor([
                                    x[i, k, l, m],
                                    x[i, k, l, m+1],
                                    x[i, k, l, m+2],
                                    x[i, k, l+1, m],
                                    x[i, k, l+1, m+1],
                                    x[i, k, l+1, m+2],
                                    x[i, k, l+2, m],
                                    x[i, k, l+2, m+1],
                                    x[i, k, l+2, m+2]])
                            )
        return x_new


class QCONV2d_5x5(nn.Module):
    def __init__(self,  
                 output_filters,
                 strides=1,
                 padding="same",
                 n_qlayers=1,
                 qembed_type="angle",
                 qlayer_type="basic"):
        super(QCONV2d_5x5, self).__init__()
        self.kernel_size = 5
        self.output_filters = output_filters
        self.strides = self.strides
        self.n_qubits = 5**2
        self.padding = padding
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type

        #self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = vqc_1mout(self.n_qubits, self.n_qlayers, self.qembed_type, self.qlayer_type)
        #self.clayer_out = torch.nn.Linear(self.n_qubits, self.output_filters)

    def forward(self, x):

        _, x_original_filters, x_original_height, x_original_width = x.shape
        h_pad_size = 0
        w_pad_size = 0
        if "same" == self.padding:
            h_pad_size = int(((x_original_height-1) * self.strides - x_original_height + (self.kernel_size-1) + 1) / 2)
            w_pad_size = int(((x_original_width-1) * self.strides - x_original_width + (self.kernel_size-1) + 1) / 2)
        pad_opt = nn.ZeroPad2d((w_pad_size, w_pad_size, h_pad_size, h_pad_size))
        x = pad_opt(x)
        x_batch_size, x_filters, x_new_height, x_new_width = x.shape

        x_output_height = int((x_original_height - (self.kernel_size-1) + 2 * h_pad_size - 1) / self.strides + 1)
        x_output_width = int((x_original_width - (self.kernel_size-1) + 2 * w_pad_size - 1) / self.strides + 1)

        x_new = torch.zeros((x_batch_size, self.output_filters, x_output_height, x_output_width))

        for i in range(x_batch_size):
            for j in range(self.output_filters):
                for k in range(x_original_filters):
                    for l in range(0, x_output_height, self.strides):
                        for m in range(0, x_output_width, self.strides):
                            x_new[i, j, k, l] = self.VQC(
                                torch.Tensor([
                                    x[i, j, k, l],
                                    x[i, j, k, l+1],
                                    x[i, j, k, l+2],
                                    x[i, j, k, l+3],
                                    x[i, j, k, l+4],
                                    x[i, j, k+1, l],
                                    x[i, j, k+1, l+1],
                                    x[i, j, k+1, l+2],
                                    x[i, j, k+1, l+3],
                                    x[i, j, k+1, l+4],
                                    x[i, j, k+2, l],
                                    x[i, j, k+2, l+1],
                                    x[i, j, k+2, l+2],
                                    x[i, j, k+2, l+3],
                                    x[i, j, k+2, l+4],
                                    x[i, j, k+3, l],
                                    x[i, j, k+3, l+1],
                                    x[i, j, k+3, l+2],
                                    x[i, j, k+3, l+3],
                                    x[i, j, k+3, l+4],
                                    x[i, j, k+4, l],
                                    x[i, j, k+4, l+1],
                                    x[i, j, k+4, l+2],
                                    x[i, j, k+4, l+3],
                                    x[i, j, k+4, l+4]])
                            )
        return x_new


if __name__ == "__main__":
    a = torch.ones((1, 3, 32, 32))
    a1 = torch.ones((1, 3, 32))
    a11 = torch.ones((1, 32, 48))
    #s_1 = time.time()
    #qconv1dbk = QCONV1d(output_filters=32)
    #g = qconv1dbk.forward(a11)
    #print(time.time() - s_1)
    #print(g.shape)
    #s_2 = time.time()
    #qconv1d1 = QCONV1d1(output_filters=32)
    #h = qconv1d1.forward(a11)
    #print(time.time() - s_2)
    #print(h.shape)
    #s_3 = time.time()
    #qconvbk = QCONVbk(output_filters=6)
    #b = qconvbk.forward(a)
    #print(time.time() - s_3)
    #print(b.shape)
    #s_4 = time.time()
    #qconv = QCONV(output_filters=6)
    #c = qconv.forward(a)
    #print(time.time() - s_4)
    #print(c.shape)
    #s_5 = time.time()
    #qconv1 = QCONV1(output_filters=6)
    #d = qconv1.forward(a)
    #print(time.time() - s_5)
    #print(d.shape)
    #s_6 = time.time()
    #qconv1d2 = QCONV1d2(output_filters=32)
    #k = qconv1d2.forward(a11)
    #print(time.time() - s_6)
    #print(k.shape)
    s_7 = time.time()
    qconv2d = QCONV2d(output_filters=32)
    j = qconv2d.forward(a)
    print(time.time() - s_7)
    print(j.shape)

