import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images_path = os.path.join(root_dir, "images")
        self.labels_path = os.path.join(root_dir, "labels")
        self.images = sorted(os.listdir(self.images_path))
        self.labels = sorted(os.listdir(self.labels_path))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
            image_path = os.path.join(self.images_path, self.images[index])
            label_path = os.path.join(self.labels_path, self.labels[index])
            image = Image.open(image_path).convert("RGB")
            with open(label_path, "r", encoding="utf-8") as file:
                label = file.read()
            return image, label[::-1]
            