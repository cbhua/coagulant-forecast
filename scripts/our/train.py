import sys; sys.path.append('.')
import argparse
import torch
import pytorch_lightning as pl
from box import Box
from src.model import Our
from src.datamodule import KWaterGRUModule
from src.logger import setup_logger


# Read Parameters
parser = argparse.ArgumentParser()
parser.add_argument('-cf', '--config_file', help='Config File Path', default='scripts/mlp/config.yaml')
args = parser.parse_args()
config = Box.from_yaml(filename=args.config_file)

# Setup random seed
torch.manual_seed(config.general.seed)

# Load Dataset
# Use the same dataset with GRU model
datamodule = KWaterGRUModule(
    train_valid_data_dir=config.data.train_valid_data_dir,
    test_data_dir=config.data.test_data_dir,
    batch_size=config.data.batch_size,
    windows_size=config.data.windows_size,
    sliding=config.data.sliding,
    num_workers=config.data.num_workers
)

# Init Module
model = Our(
    n_features=config.model.n_features,
    window_size=config.model.windows_size,
    out_dim=config.model.out_dim,
    kernel_size=config.model.kernel_size, # todo
    use_gatv2=config.model.use_gatv2, 
    feat_gat_embed_dim=None,
    time_gat_embed_dim=None,
    gru_n_layers=config.model.gru_n_layers,
    gru_hid_dim=config.model.gru_hid_dim,
    forecast_n_layers=config.model.forecast_n_layers, # todo 2
    forecast_hid_dim=config.model.forecast_hid_dim, # todo 2
    recon_n_layers=config.model.recon_n_layers, # todo 2
    recon_hid_dim=config.model.recon_hid_dim, # todo 2
    dropout=config.model.dropout,
    alpha=config.model.alpha
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
