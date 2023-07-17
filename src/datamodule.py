import sys; sys.path.append('.')
import torch
import numpy as np
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.dataset import KWaterMLP, KWaterGRU


class KWaterMLPModule(LightningDataModule):
    def __init__(self, 
                 train_valid_data_dir: str,
                 test_data_dir: str,
                 batch_size: int,
                 num_workers: int):
        super().__init__()
        self.train_valid_data_dir = train_valid_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_data()
        self.setup()
    
    def prepare_data(self):
        self.full_data = np.load(self.train_valid_data_dir)
        self.test_data = np.load(self.test_data_dir)

    def setup(self, stage: Optional[str] = None):
        '''
        dataset returns:
            x: <numpy.array> [8] input feature
            y: <numpy.array> [1] coaugulent
        '''
        full_dataset = KWaterMLP(self.full_data)
        train_size = int(0.8 * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
        test_dataset = KWaterMLP(self.test_data)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def train_dataloader(self, *args, **kwargs):
        return self.train_loader

    def val_dataloader(self, *args, **kwargs):
        return self.valid_loader

    def test_dataloader(self, *args, **kwargs):
        return self.test_loader


class KWaterGRUModule(LightningDataModule):
    def __init__(self, 
                 train_valid_data_dir: str,
                 test_data_dir: str,
                 batch_size: int,
                 windows_size: int,
                 sliding: int=1,
                 num_workers: int=4):
        super().__init__()
        self.train_valid_data_dir = train_valid_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.windows_size = windows_size
        self.sliding = sliding
        self.num_workers = num_workers
        self.prepare_data()
        self.setup()
    
    def prepare_data(self):
        self.full_data = np.load(self.train_valid_data_dir)
        self.test_data = np.load(self.test_data_dir)

    def setup(self, stage: Optional[str] = None):
        full_dataset = KWaterGRU([self.full_data], self.windows_size, self.sliding)
        train_size = int(0.8 * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
        test_dataset = KWaterGRU([self.test_data], self.windows_size, self.sliding)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def train_dataloader(self, *args, **kwargs):
        return self.train_loader

    def val_dataloader(self, *args, **kwargs):
        return self.valid_loader

    def test_dataloader(self, *args, **kwargs):
        return self.test_loader


class KWaterOurModule(LightningDataModule):
    def __init__(self, 
                 train_valid_data_dir: str,
                 test_data_dir: str,
                 batch_size: int,
                 windows_size: int,
                 sliding: int=1,
                 num_workers: int=4):
        super().__init__()
        self.train_valid_data_dir = train_valid_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.windows_size = windows_size
        self.sliding = sliding
        self.num_workers = num_workers
        self.prepare_data()
        self.setup()
    
    def prepare_data(self):
        self.full_data = np.load(self.train_valid_data_dir)
        self.test_data = np.load(self.test_data_dir)

    def setup(self, stage: Optional[str] = None):
        full_dataset = KWaterGRU([self.full_data], self.windows_size, self.sliding)
        train_size = int(0.8 * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
        test_dataset = KWaterGRU([self.test_data], self.windows_size, self.sliding)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def train_dataloader(self, *args, **kwargs):
        return self.train_loader

    def val_dataloader(self, *args, **kwargs):
        return self.valid_loader

    def test_dataloader(self, *args, **kwargs):
        return self.test_loader


if __name__ == '__main__':
    print('--- TEST of MLP Data Module ---')
    datamodule = KWaterMLPModule(
        train_valid_data_dir = 'data/data_toc/scale_2016-2020.npy',
        test_data_dir = 'data/data_toc/scale_2021.npy',
        batch_size = 32,
        num_workers = 4
    )
    x, y = next(iter(datamodule.train_dataloader()))
    print(f'Input Shape: {x.size()}')
    print(f'Option Shape: {y.size()}')

    print('--- TEST of GRU Data Module ---')
    datamodule = KWaterGRUModule(
        train_valid_data_dir = 'data/data_toc/scale_2016-2020.npy',
        test_data_dir = 'data/data_toc/scale_2021.npy',
        batch_size = 32,
        windows_size = 200, 
        sliding = 1,
        num_workers = 4
    )
    x, y, z = next(iter(datamodule.train_dataloader()))
    print(f'Input Shape: {x.size()}')
    print(f'Reconstruction Shape: {y.size()}')
    print(f'Prediction Shape: {y.size()}')

    print('--- TEST of Our Data Module ---')
    datamodule = KWaterOurModule(
        train_valid_data_dir = 'data/data_toc/scale_2016-2020.npy',
        test_data_dir = 'data/data_toc/scale_2021.npy',
        batch_size = 32,
        windows_size = 200, 
        sliding = 1,
        num_workers = 4
    )
    x, y, z = next(iter(datamodule.train_dataloader()))
    print(f'Input Shape: {x.size()}')
    print(f'Reconstruction Shape: {y.size()}')
    print(f'Prediction Shape: {y.size()}')
