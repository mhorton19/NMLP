import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import numpy as np

class OrthonormalLinear(nn.Module):
    def __init__(self, num_groups, num_neurons, outputs_per_neuron, device=None, dtype=None):
        super(OrthonormalLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_groups = num_groups
        self.num_neurons = num_neurons
        self.outputs_per_neuron = outputs_per_neuron
        self.weight = Parameter(torch.empty((num_neurons, num_neurons, outputs_per_neuron, outputs_per_neuron), **factory_kwargs))

        self.reset_parameters()

    #DOESNT REALLY MATTER
    def reset_parameters(self) -> None:
        init.uniform_(self.weight, a=-1, b=1)

    def get_orthonormal_weights(self):
        weight = self.weight / torch.linalg.vector_norm(self.weight, dim=-1, keepdim=True)

        rows = weight[:,:,0:1,:]

        for i in range(1,self.outputs_per_group):
            new_row = weight[:,:,i,:].unsqueeze(-1)
            row_correlations = torch.matmul(rows, new_row).squeeze(-1).unsqueeze(-2)
            adjustment = torch.matmul(row_correlations, rows).squeeze(-2).unsqueeze(-1)
            new_row_adjusted = new_row - adjustment
            new_row_adjusted = new_row_adjusted / torch.linalg.vector_norm(new_row_adjusted, dim=-2, keepdim=True)
            rows = torch.cat([rows, new_row_adjusted.squeeze(-1).unsqueeze(-2)], dim=-2)

        #test = torch.matmul(rows, torch.transpose(rows, -2, -1))
        return torch.transpose(rows, -2, -3) / math.sqrt(self.num_groups)

    def forward(self, x, orthoganal_weights, attention_mask=None):
        x_reshaped = x.reshape((x.shape[0], -1))
        w = orthoganal_weights.reshape(self.num_neurons*self.outputs_per_neuron, self.num_neurons*self.outputs_per_neuron)
        out = torch.matmul(x_reshaped, w)
        return out.reshape(x.shape)

class NeuronalPerceptron(nn.Module):
    def __init__(self, inp_size, neuron_groups, outputs_per_neuron):
        super(NeuronalPerceptron, self).__init__()
        self.neuron_groups = neuron_groups
        self.num_groups = len(neuron_groups)
        self.total_neurons = sum(neuron_groups)
        self.outputs_per_neuron = outputs_per_neuron

        self.ols = []
        for i in range(self.num_groups):

            self.ols.append(OrthonormalLinear(self.total_neurons, outputs_per_neuron))
        self.ln = nn.LayerNorm(self.total_neurons)
        self.il = nn.Linear(inp_size, self.neuron_groups[0] * outputs_per_neuron, bias=False)

    def get_weights(self):
        return self.ol.get_orthonormal_weights()

    def forward(self, x, weights, prev_state=0):
        x_combined = prev_state + self.il(x).reshape((-1, self.neuron_groups[0], *([1] * (self.num_groups-1)), self.outputs_per_neuron)) / math.sqrt(self.outputs_per_neuron)
        y = self.ol(x_combined, weights)
        magnitudes = torch.linalg.vector_norm(y, dim=-1, keepdims=True)
        x_combined_unit = x_combined / magnitudes

        new_mag = torch.relu(self.ln(magnitudes[:,:,0]))
        return new_mag.unsqueeze(-1)*x_combined_unit, new_mag.squeeze(-1)

def main():
    ol = OrthonormalLinear(5, 3)
    ow = ol.get_orthonormal_weights()
    perceptron = NeuronalPerceptron(10, 15, 15)
    weights = perceptron.get_weights()

    rm = torch.tensor(np.random.normal(size=(16, 11, 10)).astype(np.float32))
    prev_state = 0
    for i in range(rm.shape[1]):
        prev_state, mags = perceptron(rm[:,i,:], weights, prev_state=prev_state)
    print('here')

if __name__ == "__main__":
    main()