import torch
from opencood.models.torch_tdl.Transformer import TransformerModel
import torch.nn as nn
import torch.nn.functional as F
import math
test_model=TransformerModel()
input=torch.randn(1,14,768)
output=test_model(input)
print(output.shape)

