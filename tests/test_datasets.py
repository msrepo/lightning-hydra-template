from pathlib import Path
import pytest
import torch
from torchvision.datasets import ImageFolder

from src.data.components.csv_dataset import CustomCSVDataset

@pytest.fixture
def val_dataset_params():
    data_dir = "../PUBLIC_DATASETS/ophthalmology/ADAM/ADAM/"
    val_image_root_dir = Path(data_dir,'Validation/image')
    val_label_csv_file = Path(data_dir, 'Validation/validation_classification_GT.txt')
    return str(val_image_root_dir), str(val_label_csv_file)

@pytest.fixture
def test_dataset_params():
    data_dir = "../PUBLIC_DATASETS/ophthalmology/ADAM/ADAM/"
    test_image_root_dir = Path(data_dir,'Test/Test-image-400')
    test_label_csv_file = Path(data_dir, 'Test/test_classification_GT.txt')
    return str(test_image_root_dir), str(test_label_csv_file)

@pytest.fixture
def train_dataset_params():
    data_dir = "../PUBLIC_DATASETS/ophthalmology/ADAM/ADAM/"
    train_image_root_dir = Path(data_dir,'Train/Training-image-400')
    return str(train_image_root_dir)

def test_train_dataset_initialization(train_dataset_params):
    trainset = ImageFolder(root=train_dataset_params)
    assert trainset.class_to_idx == {'Non-AMD': 0, 'AMD': 1}
    
def test_val_dataset_initialization(val_dataset_params):
    image_root_dir, label_csv_file = val_dataset_params
    dataset = CustomCSVDataset(image_root_dir=image_root_dir, label_csv_file=label_csv_file,separator=' ')
    assert dataset is not None, "CSVDataset initialization failed."
    
    assert len(dataset) == 400, "Dataset length mismatch."
    
    assert dataset.num_classes() == 2, "Number of classes mismatch."
    x,y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

    
def test_test_dataset_initialization(test_dataset_params):
    image_root_dir, label_csv_file = test_dataset_params
    dataset = CustomCSVDataset(image_root_dir=image_root_dir, label_csv_file=label_csv_file,separator=' ')
    assert dataset is not None, "CSVDataset initialization failed."
    
    assert len(dataset) == 400, "Dataset length mismatch."
    
    assert dataset.num_classes() == 2, "Number of classes mismatch."
    assert dataset.num_classes() == 2, "Number of classes mismatch."
    x,y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64    


