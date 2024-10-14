import os
import os.path as osp
import sys

import py_cli_interaction
from typing import Tuple, Optional, Iterable, Dict
import hydra
from omegaconf import DictConfig

sys.path.insert(0, osp.join("..", os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from learning.inference_3d import Inference3D
from manipulation.experiment_real import ExperimentReal

from controller.configs.config import config as config_planner
from controller.configs.config import config_tshirt_short as planning_config_tshirt_short
from controller.configs.config import config_tshirt_long as planning_config_tshirt_long
from controller.configs.config import config_nut as planning_config_nut
from controller.configs.config import config_rope as planning_config_rope
from common.experiment_base import convert_dict
from omegaconf import OmegaConf
# Experiment = None
from common.logging_utils import Logger as ExpLogger

def is_validate_object_id(object_id: str) -> bool:
    if object_id == "":
        return False
    else:
        # TODO: verify object id, check if it is in the database
        return True

def collect_real_data(cfg, exp: ExperimentReal):
    for obj_idx in range(100):
        logger.info("Input object id...")
        object_id = ""
        while not (is_validate_object_id(object_id) and continue_flag):
            object_id = py_cli_interaction.must_parse_cli_string("input object_id")
            continue_flag = py_cli_interaction.must_parse_cli_bool(
                "i have confirmed that the correct object is selected and flattened"
            )

        # create logger
        experiment_logger = ExpLogger(
            namespace=cfg.logging.namespace, config=cfg.logging, tag=cfg.logging.tag
        )
        experiment_logger.init()
        experiment_logger.log_running_config(cfg)
        experiment_logger.log_commit(cfg.experiment.environment.project_root)
        experiment_logger.log_object_id(object_id)

        # take point cloud
        logger.info("stage 3.1: capture pcd!")

        obs, err = exp.get_obs()
        experiment_logger.log_pcd_raw("begin", obs.raw_virtual_pcd)
        experiment_logger.log_rgb("begin", obs.rgb_img)
        experiment_logger.log_mask("begin", obs.mask_img)

        experiment_logger.log_pcd_processed("begin", obs.valid_virtual_pcd)
        experiment_logger.close()


@hydra.main(
    config_path="../config/finetune_experiment", config_name="experiment_finetune_tshirt_short", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    
    cfg.logging.namespace = "capture_gt_pcd"
    continue_flag = False
    while not (continue_flag):
        cfg.logging.tag = py_cli_interaction.must_parse_cli_string("input logging tag")
        continue_flag = py_cli_interaction.must_parse_cli_bool(
            "i have confirmed that the entered logging tag is correct"
        )
    
    if cfg.experiment.compat.object_type == 'tshirt_long':
        planning_config = planning_config_tshirt_long
    elif cfg.experiment.compat.object_type == 'tshirt_short':
        planning_config = planning_config_tshirt_short
    elif cfg.experiment.compat.object_type == 'nut':
        planning_config = planning_config_nut
    elif cfg.experiment.compat.object_type == 'rope':
        planning_config = planning_config_rope
    else:
        raise NotImplementedError
    planning_config.update(config_planner)
    
    cfg.experiment.compat.use_real_robots = True
    cfg.experiment.planning = OmegaConf.create(convert_dict(planning_config))

    # create experiment
    exp = ExperimentReal(config=cfg.experiment)

    # start capturing objects
    collect_real_data(cfg, exp)


if __name__ == "__main__":
    main()
