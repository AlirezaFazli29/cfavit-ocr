import os
from PIL import Image
from transformers import AutoProcessor
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """
    A custom dataset for OCR tasks with images and corresponding text labels.

    The dataset expects a directory structure like:

    root_dir/
        images/
            img_1.png
            img_2.png
            ...
        labels/
            img_1.txt
            img_2.txt
            ...

    Each image has a matching text file with the same sorted order.

    Args:
        root_dir (str): Root directory containing `images/` and `labels/` subdirectories.

    Returns:
        Tuple[Image.Image, str]: A tuple containing the image (RGB) and its corresponding label
        string (reversed for right-to-left languages like Farsi).
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.images_path = os.path.join(root_dir, "images")
        self.labels_path = os.path.join(root_dir, "labels")
        self.images = sorted(os.listdir(self.images_path))
        self.labels = sorted(os.listdir(self.labels_path))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, index):
        """
        Load an image and its corresponding label.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[Image.Image, str]: (image, reversed label).
        """
        image_path = os.path.join(self.images_path, self.images[index])
        label_path = os.path.join(self.labels_path, self.labels[index])
        image = Image.open(image_path).convert("RGB")
        with open(label_path, "r", encoding="utf-8") as file:
            label = file.read()
        return image, label[::-1]


class CreateLoader:
    """
    Utility class to create a DataLoader with a custom collate function
    using a HuggingFace AutoProcessor for OCR tasks.

    Args:
        root_dir (str): Root dataset directory containing `images/` and `labels/`.
        processor (AutoProcessor): HuggingFace processor (e.g., TrOCRProcessor or AutoProcessor).
        device (str, optional): Device for tensors ("cpu" or "cuda"). Default is "cpu".

    Usage:
        >>> from transformers import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        >>> dataset, dataloader = CreateLoader("dataset_path", processor, device="cuda")(batch_size=16)
    """
    def __init__(
            self,
            root_dir: str,
            processor: AutoProcessor,
            device: str="cpu",
    ):
        self.dataset = CustomDataset(root_dir)
        self.processor = processor
        self.device = device

    def collate_fn(self, batch):
        """
        Collate function for batching OCR data.

        Args:
            batch (List[Tuple[Image.Image, str]]): A batch of (image, label) pairs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - pixel_values: Preprocessed image tensors of shape (batch, C, H, W).
                - label_ids: Tokenized label IDs tensor of shape (batch, seq_len).
        """
        images, labels = zip(*batch)
        pixel_values = self.processor(
            list(images),
            return_tensors="pt",
            device=self.device
        ).pixel_values
        label_ids = self.processor.tokenizer(
            list(labels),
            return_tensors="pt",
            padding=True
        ).input_ids.to(self.device)
        return pixel_values, label_ids

    def __call__(
            self,
            batch_size: int=16,
            shuffle:
            bool=True,
            *args, **kwds
    ):
        """
        Create a PyTorch DataLoader for the dataset.

        Args:
            batch_size (int, optional): Number of samples per batch. Default is 16.
            shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
            *args, **kwds: Additional arguments passed to torch.utils.data.DataLoader.

        Returns:
            Tuple[CustomDataset, DataLoader]: The dataset and its corresponding DataLoader.
        """
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            *args, **kwds
        )
        return self.dataset, dataloader