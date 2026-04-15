import torch
data_test=torch.load('cortex_mnist/test.pt',weights_only=False)

print(data_test['spikes_on'].shape)