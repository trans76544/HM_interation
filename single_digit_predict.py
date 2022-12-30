import torch
from torch import nn
import torchvision
import cv2
from Model import Model
import numpy as np

model = torch.load('single_digit.pth', map_location='cpu')
img = cv2.imread('15.jpg')
img = cv2.resize(img, (32, 32))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
image = transform(img)
image = image.view(1, *image.size())

model.eval()
output = model(image)
_, pred = torch.max(output, 1)
prediction = pred.numpy()[0]
print(prediction)