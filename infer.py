import time
import numpy as np
from utils.getkeys import key_check
import pydirectinput
import keyboard
import time
import cv2
from utils.grabscreen import grab_screen
from utils.directkeys import PressKey, ReleaseKey, W, D, A
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
    def __init__(self, model_name = 'resnet18', pretrained=False):
        super().__init__()
        self.model = timm.create_model('resnet18', pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 5)

    def forward(self, x):
        x = self.model(x)
        return x

model = CustomNet('resnet18', pretrained = False)
model.load_state_dict(torch.load('D:/DATA_SCIENCE/subway_surfers/weights/resnet18_fold0_best.pth',
                                map_location = 'cpu')['model'])
model.eval()
print('loaded weights....')


sleepy = 0.1

# Wait for me to push B to start.
keyboard.wait('B')
time.sleep(sleepy)

while True:

    image = grab_screen(region=(50, 100, 1287, 724))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image , (224, 224))

    
    t_image = image_to_tensor(image)

    start_time = time.time()
    result = model(t_image.unsqueeze(0))
    result = result.softmax(1)
    action = int(result.argmax(1)[0])
    print(action)

    
    if action == 0:
        print("JUmp")
        keyboard.press("w")
        keyboard.release("a")
        keyboard.release("d")
        time.sleep(sleepy)

    elif action == 1:
        print("Go left")
        keyboard.press("a")
        keyboard.release("d")
        #keyboard.release("space")
        time.sleep(sleepy)

    elif action == 2:
        print("Go right")
        keyboard.press("d")
        keyboard.release("a")
        #keyboard.release("space")
        time.sleep(sleepy)

    elif action == 3:
        print("Roll")
        keyboard.press("s")
        keyboard.release("a")
        #keyboard.release("space")
        time.sleep(sleepy)

    '''elif action == 4:
        print("Straight")
        keyboard.release("d")
        keyboard.release("a")
        keyboard.release("w")
        keyboard.release("s")
        time.sleep(sleepy)'''

    # End simulation by hitting h
    keys = key_check()
    if keys == "H":
        break

keyboard.release('W')
