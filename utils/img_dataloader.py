from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os


class ImageDataset(Dataset):

    def __init__(
        self,
        metadata_path,
        image_dir,
        prompt,
        resolution=512,
    ):

        with open(metadata_path, "r") as f:
            self.items = json.load(f)

        self.image_dir = image_dir
        self.prompt = prompt

        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        item = self.items[idx]

        image_path = os.path.join(
            self.image_dir,
            item["image"],
        )

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return {
            "pixel_values": image,
            "prompt": self.prompt,
            "seed": item["seed"],
        }