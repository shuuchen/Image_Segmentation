import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils import data

class RoomDataset(Dataset):
    def __init__(self, file_path, train=True, augment=False):
        self.file_path = file_path
        self.augment = augment
        self.train = train

        if self.train:
          self.img_list = [f for f in os.listdir(file_path) if 'image' in f]
          self.label_list = [f.replace('image', 'room') for f in self.img_list]
        else:
          self.img_list = [f for f in os.listdir(file_path) if '.npy' in f]

    def __len__(self):
        return len(self.img_list)

    def _to_tensor(self, array, is_label=False):
        assert (isinstance(array, np.ndarray))
        tensor = torch.from_numpy(array)
        return tensor.long() if is_label else tensor.float()

    def __getitem__(self, index):
        
        image = np.load(os.path.join(self.file_path, self.img_list[index])).transpose(1,2,0)

        if not self.train:
          return self._to_tensor(image).permute(2,0,1), self.img_list[index]

        label = np.load(os.path.join(self.file_path, self.label_list[index]))

        height, width = label.shape

        if self.train and self.augment:

          # random rotations
          random_rotation(image, label)

          # random h-flips
          horizontal_flip(image, label)

          # random v-flips
          vertical_flip(image, label)

          # random crops
          scale_augmentation(image, label)

        return self._to_tensor(image).permute(2,0,1), self._to_tensor(label, is_label=True)

def get_loader(image_path, image_size, batch_size, num_workers=16, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = RoomDataset(image_path, True, False)
	data_loader = data.DataLoader(dataset=dataset, 
                                      batch_size=batch_size, 
                                      shuffle=True,
	                              num_workers=num_workers)
	return data_loader
