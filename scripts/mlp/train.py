import sys; sys.path.append('.')
import argparse
import torch
import pytorch_lightning as pl
from box import Box
from src.model import MLP
from src.datamodule import KWaterMLPModule
from src.logger import setup_logger


# Read Parameters
parser = argparse.ArgumentParser()
parser.add_argument('-cf', '--config_file', help='Config File Path', default='scripts/mlp/config.yaml')
args = parser.parse_args()
config = Box.from_yaml(filename=args.config_file)

# Setup random seed
torch.manual_seed(config.general.seed)

# Load Dataset
datamodule = KWaterMLPModule(
    train_valid_data_dir=config.data.train_valid_data_dir,
    test_data_dir=config.data.test_data_dir,
    batch_size=config.data.batch_size,
    num_workers=config.data.num_workers
)

# Init Module
model = MLP(
    in_dim=config.model.in_dim,
    out_dim=config.model.out_dim,
    hdim=config.model.hdim,
    lr=config.model.lr
)

# Init Logger
logger = setup_logger(
    name=config.general.run_name,
    project_name=config.general.project_name,
    use_wandb=config.general.use_wandb
)

# Init Trainner
trainer = pl.Trainer(
    max_epochs=config.train.epoch,
    logger=logger, 
    devices=config.general.device,
    accelerator=config.general.accelerator
)

trainer.fit(model, datamodule)
trainer.validate(model, datamodule)
