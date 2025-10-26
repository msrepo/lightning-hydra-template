import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch

class CustomCSVDataset(Dataset):
    def __init__(self, image_root_dir, label_csv_file, csv_has_header = False, separator=',',transform=None, target_transform=None):
        """
        Args:
            label_csv_file (string): Path to the csv file that contains image_file_name and label columns.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_csv(label_csv_file, header=0 if csv_has_header else None, sep=separator)
        self.image_transform = transform
        self.label_transform = target_transform
        self.image_root_dir = image_root_dir

    def __len__(self):
        return len(self.dataframe)

    def _read_image(self, image_path):
        # Implement image reading logic here, e.g., using PIL or OpenCV
        from PIL import Image
        return np.asarray(Image.open(image_path).convert('RGB'))
    
    def num_classes(self):
        # convert the last column into ints and find unique values
        return self.dataframe.iloc[:, -1].nunique()
        
    
    def __getitem__(self, idx):
        # Example: Assuming the last column is the label and first column is image file name
        full_image_path = f"{self.image_root_dir}/{self.dataframe.iloc[idx, 0]}" 
        image = torch.tensor(self._read_image(full_image_path), dtype=torch.float32)
        label = torch.tensor(self.dataframe.iloc[idx, -1], dtype=torch.long) # or float32 for regression


        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)
            
        return image, label