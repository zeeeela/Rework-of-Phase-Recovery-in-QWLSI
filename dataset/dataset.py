from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


from PIL import Image

import os
import random


class QWLSIDataset(Dataset):
    def __init__(self, interferogram_dir, phase_dir, transform=None):
        self.interferogram_dir = interferogram_dir
        self.phase_dir = phase_dir
        self.transform = transform

        self.interferogram_images = sorted(os.listdir(interferogram_dir))
        self.phase_images = sorted(os.listdir(phase_dir))

    def __len__(self):
        return len(self.interferogram_images)

    def __getitem__(self, idx):
        interferogram_path = os.path.join(self.interferogram_dir, self.interferogram_images[idx])
        phase_path = os.path.join(self.phase_dir, self.phase_images[idx])

        interferogram_image = Image.open(interferogram_path).convert('L')  # Grayscale
        phase_image = Image.open(phase_path).convert('L')  # Grayscale

        if self.transform:
            interferogram_image = self.transform(interferogram_image)
            phase_image = self.transform(phase_image)

        return interferogram_image, phase_image