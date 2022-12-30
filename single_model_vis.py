import torch
from Model import Model
from torch.utils.tensorboard import SummaryWriter

x = torch.rand(1, 1, 32, 32)
model = Model()
print(model)

with SummaryWriter(comment='Model') as w:
    w.add_graph(model, x)


