'''
File: Contains the code of qcircuit
'''

import torch
import torch.nn as nn

import pennylane as qml


class HybridVQC(nn.Module):
    def __init__(self,
            input_filters,
            output_filters,
            n_qubits,
            n_qlayers,
            qembed_type="angle",
            qlayer_type="basic"):
        super(HybridVQC, self).__init__()
    
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.qembed_type = qembed_type
        self.qlayer_type = qlayer_type
        
        self.clayer_in = torch.nn.Linear(self.input_filters, self.n_qubits)
        self.VQC = vqc(self.n_qubits, self.n_qlayers, self.qembed_type, self.qlayer_type)
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.output_filters)

    def forward(self, x):

        x_in = self.clayer_in(x)
        vqc_out = self.VQC(x_in)
        x_out = self.clayer_out(vqc_out)

        return x_out


def vqc(n_qubits,
        n_qlayers,
        qembed_type="angle",
        qlayer_type="basic"):

    dev = qml.device("default.qubit", wires=n_qubits)
    def _circuit(inputs, weights):
        # setting embedding
        if "angle" == qembed_type:
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        if "amplitude" == qembed_type:
            qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits))
        if "basic" == qembed_type:
            qml.templates.BasisEmbedding(inputs, wires=range(n_qubits))
        # setting layer
        if "basic" == qlayer_type:
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        if "strong" == qlayer_type:
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    qlayer = qml.QNode(_circuit, dev, interface="torch")

    weight_shapes = {"weights": (n_qlayers, n_qubits)}

    return qml.qnn.TorchLayer(qlayer, weight_shapes)


def vqc_1mout(n_qubits,
              n_qlayers,
              qembed_type="angle",
              qlayer_type="basic"):

    dev = qml.device("default.qubit", wires=n_qubits)
    def _circuit(inputs, weights):
        # setting embedding
        if "angle" == qembed_type:
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        if "amplitude" == qembed_type:
            qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits))
        if "basic" == qembed_type:
            qml.templates.BasisEmbedding(inputs, wires=range(n_qubits))
        # setting layer
        if "basic" == qlayer_type:
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        if "strong" == qlayer_type:
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(wires=0))
    qlayer = qml.QNode(_circuit, dev, interface="torch")

    weight_shapes = {"weights": (n_qlayers, n_qubits)}

    return qml.qnn.TorchLayer(qlayer, weight_shapes)


if __name__ == "__main__":
    a = torch.ones((8, 32))
    #V = vqc(4, 1)
    #b = V(a1)
    #print(b.shape)
    hvqc = HybridVQC(32, 16, 4, 1)
    c = hvqc(a)
    print(c.shape)

