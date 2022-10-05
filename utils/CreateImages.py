import cv2
import numpy as np

run = 10

data = np.load(f"../data/training_data{run}.npy", allow_pickle=True)
targets = np.load("../data/target_data{run}.npy", allow_pickle=True)

print(f'Image Data Shape: {data.shape}')
print(f'targets Shape: {targets.shape}')

# Lets see how many of each type of move we have.
unique_elements, counts = np.unique(targets, return_counts=True)
print(np.asarray((unique_elements, counts)))

# Store both data and targets in a list.
# We may want to shuffle down the road.

holder_list = []
for i, image in enumerate(data):
    holder_list.append([data[i], targets[i]])

count_jump = 0
count_left = 0
count_right = 0
count_roll = 0

for data in holder_list:
    #removing nothing since we have too many
    if data[1] == 'W':
        count_jump += 1
        cv2.imwrite(f"D:/DATA_SCIENCE/fall_guys_grind/Fall-Guys-AI/subway_grouped_data/Jump/{count_jump}_run{run}.png", data[0]) 
    elif data[1] == 'A':
        count_left += 1
        cv2.imwrite(f"D:/DATA_SCIENCE/fall_guys_grind/Fall-Guys-AI/subway_grouped_data/Left/{count_left}_run{run}.png", data[0]) 
    elif data[1] == 'D':
        count_right += 1
        cv2.imwrite(f"D:/DATA_SCIENCE/fall_guys_grind/Fall-Guys-AI/subway_grouped_data/Right/{count_right}_run{run}.png", data[0]) 
    elif data[1] == 'S':
        count_roll += 1
        cv2.imwrite(f"D:/DATA_SCIENCE/fall_guys_grind/Fall-Guys-AI/subway_grouped_data/Roll/{count_roll}_run{run}.png", data[0]) 
