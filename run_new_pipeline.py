import os
import os.path as osp
import sys

import py_cli_interaction
import hydra
from omegaconf import DictConfig
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.notification import get_bark_notifier
from common.statemachine import ObjectStateDef
from manipulation.statemachine_object import ObjectStateMachine

from loguru import logger
from learning.inference_3d import Inference3D

from manipulation.experiment_real import ExperimentReal
from common.experiment_base import convert_dict

from common.registry import ExperimentRegistry
from tools.debug_controller import Client as DebugClient
from omegaconf import OmegaConf


def is_validate_object_id(object_id: str) -> bool:
    if object_id == "":
        return False
    else:
        # TODO: verify object id, check if it is in the database
        return True


def collect_real_data():
    _r = ExperimentRegistry()
    cfg, exp = _r.cfg, _r.exp
    episode_idx: int = _r.episode_idx

    # create inference class
    inference = Inference3D(experiment=exp, **cfg.inference)
    
    logger.info(f'Starting Episode {episode_idx}!')
    _r.running_inference = inference

    fixed_object_id = cfg.experiment.strategy.fixed_object_id
    for obj_idx in range(cfg.experiment.strategy.instance_num_per_episode):
        logger.info("stage 1: inputs object id")
        if fixed_object_id is None:
            object_id = ""
            while not (is_validate_object_id(object_id) and continue_flag):
                object_id = py_cli_interaction.must_parse_cli_string("input object_id")
                continue_flag = py_cli_interaction.must_parse_cli_bool(
                    "i have confirmed that the correct object is selected and mounted", default_value=True
                )
        else:
            object_id = fixed_object_id

        _r.object_id = object_id
        # TODO: add object state converter
        for trial_idx in range(cfg.experiment.strategy.trial_num_per_instance):
            _r.trial_idx = trial_idx
            logger.info(f"stage 2: inputs action type")
            m = ObjectStateMachine(disp=True,
                                    only_success=cfg.inference.args.only_success,
                                    only_smoothing=cfg.inference.args.only_smoothing,
                                    enable_record=cfg.inference.args.enable_record)
            while True:
                m.loop()
                if m.current_state.name == ObjectStateDef.SUCCESS:
                    print("[result] =", m.current_state.name)
                    break
                elif m.current_state.name == ObjectStateDef.FAILED:
                    print("[result] =", m.current_state.name)
                    break

        _n = get_bark_notifier()
        err = _n.notify("[DeformPAM] Time to change the cloth")
        if err is not None:
            logger.error(f'Failed to connect to notification server!')


@hydra.main(
    config_path="config/supervised_experiment", config_name="experiment_supervised_tshirt_long", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
    logger.add("run_new_pipeline-loguru.log", enqueue=True) # enqueue=True for multi-processing)
    global __DEBUG_CLIENT__
    reg = ExperimentRegistry()
    # hydra creates working directory automatically
    pred_output_dir = os.getcwd()
    logger.info(pred_output_dir)
    _n = get_bark_notifier()
    err = _n.notify("[DeformPAM] Program Starts!!")
    if err is not None:
        logger.error(f'Failed to connect to notification server!')

    if cfg.inference.remote_debug.enable:
        logger.info(f"enable remote debug, url={cfg.inference.remote_debug.endpoint}")
        reg.debug_client = DebugClient(cfg.inference.remote_debug.endpoint)

    if cfg.experiment.compat.object_type in ['tshirt_long', 'tshirt_short', 'nut', 'rope']:
        if cfg.experiment.compat.object_type == 'tshirt_long':
            from controller.configs.config import config_tshirt_long as planning_config_tshirt_long
            planning_config = planning_config_tshirt_long
        elif cfg.experiment.compat.object_type == 'tshirt_short':
            from controller.configs.config import config_tshirt_short as planning_config_tshirt_short
            planning_config = planning_config_tshirt_short
        elif cfg.experiment.compat.object_type == 'nut':
            from controller.configs.config import config_nut as planning_config_nut
            planning_config = planning_config_nut
        elif cfg.experiment.compat.object_type == 'rope':
            from controller.configs.config import config_rope as planning_config_rope
            planning_config = planning_config_rope
        from controller.configs.config import config as config_planner
        config_planner.update(planning_config)
        cfg.experiment.planning = OmegaConf.create(convert_dict(planning_config))
    else:
        raise NotImplementedError    
    # init
    runtime_output_dir = None
    episode_idx = cfg.experiment.strategy.start_episode
    logger.debug(f'start episode_idx: {episode_idx}')
    for episode_idx in range(cfg.experiment.strategy.start_episode,
                             cfg.experiment.strategy.start_episode + cfg.experiment.strategy.episode_num):
        if runtime_output_dir is not None:
            # load newest runtime checkpoint
            cfg.inference.model_path = runtime_output_dir
            cfg.inference.model_name = 'last'

        if cfg.experiment.strategy.skip_data_collection_in_first_episode and \
                episode_idx == cfg.experiment.strategy.start_episode:
            pass
        else:
            try:
                # create experiment
                exp = ExperimentReal(config=cfg.experiment)
                if exp.option.compat.use_real_robots:
                    exp.controller.actuator.open_gripper()
                # collect data
                logger.info(f"Begin to collect data for episode {episode_idx}!")
                reg.cfg = cfg
                reg.is_validate_object_id = is_validate_object_id
                reg.exp = exp
                reg.episode_idx = episode_idx
                collect_real_data()
            finally:
                exp.camera.stop()
                logger.info('Stopping camera now!')
                reg.exp = None
                del exp

if __name__ == "__main__":
    main()
