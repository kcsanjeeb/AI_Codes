import torch

# Define the MP Neuron python class implementing AND gate
class MPNeuronAND(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(2))  # AND gate has 2 inputs with weights of 1
        self.threshold = 2  # Threshold for AND gate is 2

    def forward(self, x):
        weighted_sum = torch.sum(x * self.weights)  # Weighted sum of inputs
        output = 1.0 if weighted_sum >= self.threshold else 0.0  # Apply threshold
        return output

# Instantiate the AND neuron
mp_neuron_and = MPNeuronAND()

# Define inputs for the AND gate which will be used for testing with other input combinations
input_data_11 = torch.Tensor([1, 1])  
input_data_00 = torch.Tensor([0, 0])
input_data_01 = torch.Tensor([0, 1])
input_data_10 = torch.Tensor([1, 0])

# Perform forward pass
output_11 = mp_neuron_and(input_data_11)
output_00 = mp_neuron_and(input_data_00)
output_01 = mp_neuron_and(input_data_01)
output_10 = mp_neuron_and(input_data_10)

print(f"Output of AND Gate (1, 1): {output_11}")
print(f"Output of AND Gate (0, 0): {output_00}")
print(f"Output of AND Gate (0, 1): {output_01}")
print(f"Output of AND Gate (1, 0): {output_10}")