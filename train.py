import os
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

import sys
import os.path as osp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from learning.datasets.imitation_dataset import ImitationDataModule5
from learning.datasets.runtime_dataset_real import RuntimeDataModuleReal
from learning.net.primitive_diffusion import PrimitiveDiffusion

# %%
# main script
@hydra.main(config_path="config/supervised_experiment", config_name="train_supervised_tshirt_short", version_base='1.1')
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    print(os.getcwd())
    os.makedirs("checkpoints", exist_ok=True)
    
    model = PrimitiveDiffusion(**cfg.model)

    if cfg.runtime_datamodule is None:
        datamodule = ImitationDataModule5(**cfg.datamodule)
        datamodule.prepare_data()
    else:
        if cfg.runtime_datamodule.data_type == 'new_pipeline_supervised' and cfg.trainer_adjustment.use_adaptive_episode:
            # more epochs for multiple poses
            cfg.trainer.max_epochs *= cfg.runtime_datamodule.num_multiple_poses
            model.cos_t_max *= cfg.runtime_datamodule.num_multiple_poses
        elif cfg.runtime_datamodule.data_type == 'new_pipeline_finetune':
            # calculate the overall number of rankings per sample
            cfg.runtime_datamodule.num_rankings_per_sample = \
                cfg.runtime_datamodule.manual_num_rankings_per_sample + cfg.num_of_grasp_points // 2
            num_of_grasp_points = cfg.runtime_datamodule.num_rankings_per_sample * 4
            model.diffusion_head.num_of_grasp_points = num_of_grasp_points
            model.state_head.num_pred_candidates = num_of_grasp_points
            
        datamodule = RuntimeDataModuleReal(**cfg.runtime_datamodule)
        # use augmentation to do normalization
        datamodule.prepare_data(use_augmentation_in_val=True)

    logger = MLFlowLogger(**cfg.logger)

    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': os.getcwd(),
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    logger.log_hyperparams(all_config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        monitor='val_loss',
        save_last=True,
        save_top_k=1,
        mode='min', 
        save_weights_only=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        default_root_dir=os.getcwd(),
        enable_checkpointing=True,
        logger=logger,
        check_val_every_n_epoch=1,
        strategy="ddp_find_unused_parameters_true", 
        **cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)

    # log artifacts
    logger.experiment.log_artifact(logger.run_id, os.getcwd())

# %%
# driver
if __name__ == "__main__":
    main()
