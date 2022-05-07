'''
File: Contains custom loss functions
'''

import torch
import torch.nn as nn

epsilon = 1e-7


def inv_log_cosh(y_true, y_pred):

    y_true = y_true.to(y_pred.dtype)
    softplut_f = nn.Softplus()

    def _logcosh(x):

        return x + softplus_f(-2. * x) - torch.log(torch.Tensor([2.]))

    return torch.mean(_logcosh(100.0 / (y_pred + epsilon) - 100.0 / (y_true + epsilon) ), dim=1)

