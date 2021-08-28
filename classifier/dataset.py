import glob
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class RetinalDataset(Dataset):
    def __init__(self, 
        image_folders: list=[], 
        label_files: list=[], 
        training=True, 
        experiment=False, 
        experiment_batch=3
    ):
        super().__init__()

        if training:
            self.transformation = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.ColorJitter(brightness=10, contrast=10, saturation=10, hue=0.1),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], # Mean of ImageNet dataset
                                     [0.229, 0.224, 0.225])  # Std of ImageNet dataset
            ])
        else:
            self.transformation = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ])

        if label_files is not None:
            assert len(image_folders) == len(label_files)

        # load image path
        image_path = list()
        for image_folder in image_folders:
            image_path.extend(sorted(glob.glob(f"{image_folder}/*jpg")))
        self.image_path = image_path if not experiment else image_path[:experiment_batch]
        
        # load labels
        if label_files:
            labels = None
            filenames = None
            for label_file in label_files:                
                df = pd.read_csv(label_file)
                df = df.sort_values(by='filename')
                if labels is None:
                    labels = df.iloc[:,1:].to_numpy()
                    filenames = df.filename.tolist()
                else:
                    labels = np.concatenate([labels, df.iloc[:,1:].to_numpy()], axis=0)
                    filenames.extend(df.filename.tolist())

            assert all([path.split('/')[-1] == filename for path, filename in zip(image_path, filenames)])
            
            self.labels = labels if not experiment else labels[:experiment_batch]
        else:
            self.labels = None

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx]).convert("RGB")
        label = torch.FloatTensor()
        if self.labels is not None:
            label = torch.FloatTensor(self.labels[idx])
        return self.transformation(image), label