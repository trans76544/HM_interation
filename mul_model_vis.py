import torch
from config.config import Config
from models.crnn import CRNN
from torch.utils.tensorboard import SummaryWriter

config = Config()
x = torch.rand(1, 3, 32, 120)
nclass = len(config.label_classes) + 1
model = CRNN(config.img_height, config.nc, nclass, config.hidden_size, config.n_layers)
print(model)

# with SummaryWriter(comment='CRNN') as w:
#     w.add_graph(model, x)