import time
import numpy as np

from utils.getkeys import key_check

import time
import cv2

import torch
import torch.nn as nn

import sys
sys.path.append('D:/DATA_SCIENCE/pytorch-image-models')
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def image_to_tensor(image, mode = 'bgr'): #image mode
    if mode=='bgr':
        image = image[:,:,::-1]
    x = image
    x = x.transpose(2,0,1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x


# ====================================================
# MODEL 
# ====================================================
class CustomNet(nn.Module):
    """ using the resnet arch and modifying the final classification head"""
    def __init__(self, model_name = 'resnet10t', pretrained=False):
        super().__init__()
        self.model = timm.create_model('resnet10t', pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 5)

    def forward(self, x):
        x = self.model(x)
        return x

model = CustomNet('resnet10t', pretrained = False)
model.load_state_dict(torch.load('D:/DATA_SCIENCE/subway_surfers/weights/resnet10t_fold0_best.pth',
                                map_location = 'cpu')['model'])
model.eval()
print('loaded weights....')

image = cv2.imread('C:/Users/hp/Desktop/axial sagittal coronal.png')
image = cv2.resize(image , (384, 384))

t_image = image_to_tensor(image)

start_time = time.time()
result = model(t_image.unsqueeze(0))
result = result.softmax(1)
action = int(result.argmax(1)[0])

print(action)
