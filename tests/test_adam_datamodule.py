from pathlib import Path

import pytest
import torch

from src.data.adam_datamodule import ADAMDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_adam_datamodule(batch_size: int) -> None:
    """Tests `ADAMDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "../PUBLIC_DATASETS/ophthalmology/ADAM/ADAM/"
    train_dir = 'Train/Training-image-400/'
    val_dir = 'Validation/image'
    test_dir = 'Test/Test-image-400'
    val_label_csv_file = 'Validation/validation_classification_GT.txt'
    test_label_csv_file = 'Test/test_classification_GT.txt'

    dm = ADAMDataModule(data_dir=data_dir, train_dir=train_dir, val_dir=val_dir,test_dir=test_dir,
                        val_label_csv=val_label_csv_file,test_label_csv=test_label_csv_file,batch_size=batch_size,separator=' ')
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir).exists()
    assert Path(data_dir, train_dir).exists()
    assert Path(data_dir, val_dir).exists()
    assert Path(data_dir, test_dir).exists()

    dm.setup()
    assert dm.data_train 
    assert dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    assert dm.data_train.classes == ['AMD', 'Non-AMD'] or dm.data_train.classes == ['Non-AMD', 'AMD']
    assert dm.data_train.class_to_idx['Non-AMD'] == 0
    assert dm.data_train.class_to_idx['AMD'] == 1
    
    assert len(dm.data_train) == 400 
    assert len(dm.data_val) == 400
    assert len(dm.data_test) == 400
    
    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 1200
    
    assert len(dm.data_train.classes) == dm.data_val.num_classes() == dm.data_test.num_classes() == 2

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
