import time
from typing import Optional, Tuple

import py_cli_interaction
import yaml
from pydantic import BaseModel
from statemachine import State, StateMachine
import random
from loguru import logger
import cv2

from common.datamodels import ActionMessage, ActionTypeDef, AnnotationFlag, ExceptionMessage, ObservationMessage, PredictionMessage, GeneralObjectState, GarmentSmoothingStyle
from common.space_util import transform_point_cloud
from common.registry import ExperimentRegistry
from common.statemachine import ObjectStateDef
from common.logging_utils import Logger as ExpLogger
from common.visualization_util import visualize_pc_and_grasp_points
from controller.configs.error_config import error_code
from tools.debug_controller import get_remote_action_type_str

from multiprocessing import Process, Event
from os import path as osp
from third_party.mvcam.record_worker import record_worker

class ObjectMachineConditions(BaseModel):
    object_operable: Optional[bool] = None
    object_reachable: Optional[bool] = None
    garment_need_drag: Optional[bool] = None
    garment_consecutive_drag_num: Optional[int] = 0
    object_organized_enough: Optional[bool] = None
    garment_smoothing_style: GarmentSmoothingStyle = GarmentSmoothingStyle.UNKNOWN
    garment_rotation_angle: Optional[float] = None # in degree
    garment_keypoint_parallel: Optional[bool] = None
    garment_folded_once_success: Optional[bool] = None
    garment_folded_twice_success: Optional[bool] = None
    garment_last_strip: list = []


class ObjectStateMachine:
    
    def __new__(cls, disp: bool = False, only_success:bool = False, only_smoothing: bool = False, enable_record: bool = False):
        if only_success:
            statemachine_class = ObjectStateMachineOnlySuccess
        else:
            if only_smoothing:
                statemachine_class = ObjectStateMachineOnlySmoothing
            else:
                statemachine_class = ObjectStateMachineDefault
        return statemachine_class(disp=disp, only_success=only_success, only_smoothing=only_smoothing, enable_record=enable_record)


class ObjectStateMachineOnlySuccess(StateMachine):
    """
    Object StateMachine (Only Success)
    This statemachine is used for smoothing style collection
    """
    unknown = State(name=ObjectStateDef.UNKNOWN, initial=True)
    success = State(name=ObjectStateDef.SUCCESS, final=True)

    # transitions
    begin = (
            unknown.to(success)
    )

    loop = (
            begin
    )

    def __init__(
            self,
            disp: bool = False,
            only_success: bool = False,
            only_smoothing: bool = False,
            enable_record: bool = False,
    ):
        self.only_success = only_success
        self.only_smoothing = only_smoothing
        self.enable_record = enable_record
        self.record_worker_process = None
        
        self.step_idx = 0
        self._disp: bool = disp
        _r = ExperimentRegistry()
        self.max_num_steps = _r.cfg.experiment.strategy.step_num_per_trial

        self.condition: ObjectMachineConditions = ObjectMachineConditions()
        self._latest_err: Optional[ExceptionMessage] = None
        self._latest_logger: Optional[ExpLogger] = None
        self._latest_observation: Optional[ObservationMessage] = None
        self._latest_inference: Optional[PredictionMessage] = None
        self._latest_action: Optional[ActionMessage] = None

        self._initialized = False
        super().__init__(allow_event_without_transition=True)
        self._initialized = True

    def observation_is_valid(self) -> bool:
        return self._latest_observation is not None

    def object_operable(self) -> bool:
        return self.condition.object_operable

    def object_reachable(self) -> bool:
        return self.condition.object_reachable

    def object_organized_enough(self) -> bool:
        return self.condition.object_organized_enough
    
    def garment_keypoint_parallel(self) -> bool:
        return self.condition.garment_keypoint_parallel

    def garment_need_drag(self) -> bool:
        return self.condition.garment_need_drag

    def garment_drag_num_exceeded(self) -> bool:
        return self.condition.garment_consecutive_drag_num > 2

    def garment_folded_once_success(self) -> bool:
        return self.condition.garment_folded_once_success

    def garment_folded_twice_success(self) -> bool:
        return self.condition.garment_folded_twice_success
    
    def garment_folded_third_success(self) -> bool:
        return self.condition.garment_folded_third_success

    def garment_folded_third_success(self) -> bool:
        return self.condition.garment_folded_third_success

    def garment_step_threshold_exceeded(self) -> bool:
        return self.step_idx > self.max_num_steps

    def garment_is_strip(self):
        mask = self._latest_observation.mask_img[:, :, 0]

        valid_coord = mask.nonzero()
        x_std = valid_coord[1].std()
        y_std = valid_coord[0].std()

        logger.debug(f'x_std of garment: {x_std}. y_std of garment: {y_std}')

        is_strip = x_std > 3 * y_std
        return is_strip

    def garment_consecutive_is_strip(self) -> bool:
        if len(self.condition.garment_last_strip) == 2:
            if all(self.condition.garment_last_strip):
                logger.warning("The garment is consecutively strip twice!!!")
                self.condition.garment_last_strip = []
                return True
        return False
    
    def garment_is_rectangle(self) -> bool:
        mask = self._latest_observation.mask_img

        contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        contour_area = cv2.contourArea(largest_contour)
        bounding_rect_area = w * h
        
        logger.debug(f'Contour area: {contour_area}. Bounding_rect_area: {bounding_rect_area}.')

        # TODO: change to 0.25
        if abs(contour_area - bounding_rect_area) / bounding_rect_area < 0.4:  # 0.25 by default
            return True
        else:
            logger.warning('Garment is not rectangle!!!')
            return False

    def on_exit_unknown(self):
        _r = ExperimentRegistry()
        logger.info("stage 2: randomize the garment")
        if _r.cfg.experiment.strategy.random_lift_in_each_trial:
            self._try_lift_to_center()
        if getattr(_r.cfg.experiment.strategy, 'grasp_wiper_in_first_trial', False):
            if _r.trial_idx == 0:
                self._grasp_wiper()          

    def on_enter_unknown(self):
        if self.enable_record and self.record_worker_process is not None:
            self.record_worker_process.join()
            
        _r = ExperimentRegistry()
        logger.info("Starting Episode {}, Object {}, Trial {}".format(_r.episode_idx, _r.object_id, _r.trial_idx))
        # reset the robot to the home position
        if not getattr(_r.cfg.experiment.strategy, 'grasp_wiper_in_first_trial', False) or \
            _r.trial_idx == 0:
                _r.exp.controller.actuator.open_gripper()
        if py_cli_interaction.parse_cli_bool('Whether to enter FreeDrive mode?', default_value=False)[0]:
            _r.exp.controller.freeDrive()
            if py_cli_interaction.parse_cli_bool('Whether to exit FreeDrive mode?', default_value=True)[0]:
                _r.exp.controller.stopRobots()
            
        if py_cli_interaction.parse_cli_bool('Whether to MoveUP two robots?', default_value=False)[0]:
            _r.exp.controller.smartMoveUpMovel(up_trans_first=[0, 0, 0.03],
                                               up_trans_second=[0, 0, 0.02])
        _r.exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))
        
        if getattr(_r.cfg.experiment.strategy, 'manually_randomize_in_each_trial', False):
            self._manually_randomize()

        if self.enable_record:
            save_path_base = osp.join(
                _r.cfg.logging['path'],
                _r.cfg.logging.namespace,
                _r.cfg.logging.tag
            )
            self.stop_record_event = Event()
            self.encode_record_event = Event()
            self.delete_record_event = Event()
            self.record_worker_process = Process(
                target=record_worker, args=(save_path_base, self.stop_record_event, self.encode_record_event, self.delete_record_event)
            )
            self.record_worker_process.start()

        # protected section
        self._latest_observation, err = _r.exp.get_obs()
        self.update_condition(ignore_manual_operation=False if self.only_success else True)
        
        # test mode
        _r.running_inference.init_test_info()
    
    def on_enter_success(self):
        if self.enable_record:
            if py_cli_interaction.parse_cli_bool('Whether to encode the recording?',
                                                 default_value=True)[0]:
                self.encode_record_event.set()
            if not py_cli_interaction.parse_cli_bool('Whether to keep the raw recording?',
                                                 default_value=False)[0]:
                self.delete_record_event.set()
            self.stop_record_event.set()

        # log status
        self._log_status("success")
        self._close_logger()
        
        # log final status
        self._init_logger()
        prediction_message = PredictionMessage()
        action_message = ActionMessage(action_type=ActionTypeDef.DONE)
        self._log_after_prediction(prediction_message, action_message)
        self._log_before_action(self._latest_observation)
        self._log_status("success")
        
        _r = ExperimentRegistry()
        if not _r.cfg.experiment.compat.only_capture_pcd_before_action:
            logger.info("stage 3.4: capture pcd after action")
            obs = self._latest_observation
            self._latest_logger.log_pcd_raw("end", obs.raw_virtual_pcd, only_npz=True)
            self._latest_logger.log_rgb("end", obs.rgb_img)
            self._latest_logger.log_mask("end", obs.mask_img)
            self._latest_logger.log_pcd_processed("end", obs.valid_virtual_pcd, only_npz=True)
        
        self._close_logger()
        
        # test mode
        _r = ExperimentRegistry()
        if getattr(_r.cfg.inference.args, 'test_mode', False):
            _r.running_inference.update_test_info(status='success')
    
    def on_enter_failed(self):
        if self.enable_record:
            if not py_cli_interaction.parse_cli_bool('Whether to keep this recording?',
                                                 default_value=True)[0]:
                self.delete_record_event.set()
            self.stop_record_event.set()

        # log status
        self._log_status("failed")
        self._close_logger()
        
        # log final status
        self._init_logger()
        prediction_message = PredictionMessage()
        action_message = ActionMessage(action_type=ActionTypeDef.FAIL)
        self._log_after_prediction(prediction_message, action_message)
        self._log_before_action(self._latest_observation)
        self._log_status("failed")
        
        _r = ExperimentRegistry()
        if not _r.cfg.experiment.compat.only_capture_pcd_before_action:
            logger.info("stage 3.4: capture pcd after action")
            obs = self._latest_observation
            self._latest_logger.log_pcd_raw("end", obs.raw_virtual_pcd, only_npz=True)
            self._latest_logger.log_rgb("end", obs.rgb_img)
            self._latest_logger.log_mask("end", obs.mask_img)
            self._latest_logger.log_pcd_processed("end", obs.valid_virtual_pcd, only_npz=True)
        
        self._close_logger()
        
        # test mode
        _r = ExperimentRegistry()
        if getattr(_r.cfg.inference.args, 'test_mode', False):
            _r.running_inference.update_test_info(status='failed')

    def _capture(self):
        self._latest_observation, err = ExperimentRegistry().exp.get_obs()

    def _init_logger(self):
        _r = ExperimentRegistry()
        cfg = _r.cfg
        exp = _r.exp
        self._latest_logger = ExpLogger(
            namespace=cfg.logging.namespace, config=cfg.logging, tag=cfg.logging.tag
        )
        self._latest_logger.init()
        self._latest_logger.log_running_config(cfg)
        self._latest_logger.log_commit(cfg.experiment.environment.project_root)
        self._latest_logger.log_model(
            cfg.inference.model_path, cfg.inference.model_name
        )
        self._latest_logger.log_calibration(
            exp.transforms.camera_to_world_transform,
            exp.transforms.left_robot_to_world_transform,
            exp.transforms.right_robot_to_world_transform,
        )
        self._latest_logger.log_object_id(_r.object_id)
        self._latest_logger.log_episode_idx(_r.episode_idx)
        self._latest_logger.log_trial_idx(_r.trial_idx)
        self._latest_logger.log_action_step(self.step_idx)

    def _try_lift_to_center(self, n_try: int = 3):
        if self.step_idx == 1:
            self._log_status("begin")
        else:
            self._log_status("action")
        self._close_logger()
        self._init_logger()
        _r = ExperimentRegistry()
        _failed = True
        while _failed:
            action_message, err = _r.running_inference.choose_random_point_for_lift(
                self._latest_observation.valid_virtual_pts)
            
            prediction_message = PredictionMessage()
            self._log_after_prediction(prediction_message, action_message)
            self._log_before_action(self._latest_observation)
            
            logger.info("Garment UNREACHABLE, Implement LIFT action!")
            err = self._execute_action_failsafe(action_message)
            self._latest_observation, self._latest_err = _r.exp.get_obs()

            self._finalize_after_action(action_message, err)

            if self._latest_err is not None:
                self.current_state = ObjectStateDef.FAILED
                self.on_enter_failed()
                return ExceptionMessage("Capture failed")
            elif n_try == 0:
                return ExceptionMessage("Lift retry run out")
            else:
                _failed = not _r.exp.is_object_reachable(self._latest_observation.mask_img)

            n_try -= 1

        return None
    
    def _try_drag(self, mode='crumpled'):
        if self.step_idx == 1:
            self._log_status("begin")
        else:
            self._log_status("action")
        self._close_logger()
        self._init_logger()
        _r = ExperimentRegistry()

        if mode == 'crumpled':
            # use the last action for choosing points
            action_message, err = _r.running_inference.choose_points_for_drag(
                self._latest_observation.valid_virtual_pts, mode=mode, last_action=self._latest_action)
        elif mode == 'folded_once':
            action_message, err = _r.running_inference.choose_points_for_drag(
                self._latest_observation.valid_virtual_pts, mode=mode)
        else:
            raise NotImplementedError
        
        prediction_message = PredictionMessage()
        self._log_after_prediction(prediction_message, action_message)
        self._log_before_action(self._latest_observation)

        if err is None:
            logger.info("Garment NEED DRAG, Implement DRAG action!")
            err = self._execute_action_failsafe(action_message)
        else:
            self.current_state = ObjectStateDef.UNREACHABLE
            logger.warning('Drag action is not executable!!!')

        self._latest_observation, self._latest_err = _r.exp.get_obs()
        
        self._finalize_after_action(action_message, err)
        
        if self._latest_err is not None:
            self.current_state = ObjectStateDef.FAILED
            return ExceptionMessage("Capture failed")

        return err
    
    def _manually_randomize(self):
        _r = ExperimentRegistry()
        action_message = ActionMessage(action_type=ActionTypeDef.MANUALLY_RANDOMIZE)
        _r.exp.execute_action(action=action_message)
        
        return None
    
    def _grasp_wiper(self):
        _r = ExperimentRegistry()
        action_message = ActionMessage(action_type=ActionTypeDef.GRASP_WIPER)
        _r.exp.execute_action(action=action_message)
        
        return None

    def _get_observation(self) -> Tuple[ObservationMessage, Optional[Exception]]:
        _r = ExperimentRegistry()
        if _r.cfg.experiment.strategy.check_grasp_failure_before_action and not _r.exp.is_object_reachable(self._latest_observation.mask_img):
            self._try_lift_to_center()
        else:
            self._latest_observation, self._latest_err = _r.exp.get_obs()
            return self._latest_observation, self._latest_err

    def _log_before_action(self, obs: ObservationMessage):
        self._latest_logger.log_pcd_raw("begin", obs.raw_virtual_pcd, only_npz=True)
        self._latest_logger.log_rgb("begin", obs.rgb_img)
        self._latest_logger.log_mask("begin", obs.mask_img)
        self._latest_logger.log_pcd_processed("begin", obs.valid_virtual_pcd, only_npz=True)

    def _log_after_prediction(self, p: PredictionMessage, a: ActionMessage):
        _r = ExperimentRegistry()
        self._latest_logger.log_pose_prediction_virtual(
            "begin", p.grasp_point_all
        )
        if ('idxs' in a.extra_params) and (a.extra_params['idxs'] is not None):
            self._latest_logger.log_decision("begin", a.extra_params["idxs"])
        self._latest_logger.log_action_type(
            ActionTypeDef.to_string(a.action_type)
        )
        left_pick_point_in_virtual, right_pick_point_in_virtual = _r.exp.get_pick_points_in_virtual(a)
        left_place_point_in_virtual, right_place_point_in_virtual = _r.exp.get_place_points_in_virtual(a)

        self._latest_logger.log_pose_gripper_virtual(
            "begin", left_pick_point_in_virtual, right_pick_point_in_virtual
        )
        # FIXME: there may be bugs
        self._latest_logger.log_pose_gripper_virtual(
            "end", left_place_point_in_virtual, right_place_point_in_virtual
        )
        self._latest_logger.log_predicted_reward(
            "virtual", p.virtual_reward_all
        )
        self._latest_logger.log_predicted_reward(
            "real", p.real_reward_all
        )

    def _get_action_type(self) -> ActionTypeDef:
        _r = ExperimentRegistry()
        if _r.cfg.inference.args.action_type_override.type is not None:
            _action_type = _r.cfg.inference.args.action_type_override.type
        else:
            _action_type = "null"
            if _r.debug_client is not None:
                _action_type = get_remote_action_type_str(_r.debug_client)
            if _action_type == 'null':
                if self.current_state.value == ObjectStateDef.CRUMPLED:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.FLING)
                elif self.current_state.value == ObjectStateDef.SMOOTHED:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.FOLD_1_1)
                elif self.current_state.value == ObjectStateDef.FOLDED_ONCE:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.FOLD_2)
                elif self.current_state.value == ObjectStateDef.FOLDED_TWICE:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.DONE)
                else:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.FAIL)
                    raise Exception("TODO")
        logger.debug(f"action_type: {_action_type}")
        return ActionTypeDef.from_string(_action_type)

    def _abort_with_error(self, err):
        _r = ExperimentRegistry()
        logger.warning(f'{err}')
        
        step_idx = self.step_idx + 1
        if step_idx == 1:
            self._log_status("begin")
        else:
            self._log_status("action")

        if not _r.cfg.experiment.compat.only_capture_pcd_before_action:
            logger.info("stage 3.4: capture pcd after action")
            _r.exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))
            obs, _ = _r.exp.get_obs()
            self._latest_logger.log_pcd_raw("end", obs.raw_virtual_pcd, only_npz=True)
            self._latest_logger.log_rgb("end", obs.rgb_img)
            self._latest_logger.log_mask("end", obs.mask_img)
            self._latest_logger.log_pcd_processed("end", obs.valid_virtual_pcd, only_npz=True)

        self._latest_logger.finalize()
        self._latest_logger.log_processed_file(str(AnnotationFlag.COMPLETED.value))
        self._latest_logger.close()
        self._latest_logger = None
        self._latest_err = err
        if getattr(_r.cfg.experiment.strategy, 'open_gripper_on_abort', True):
            _r.exp.controller.actuator.open_gripper()
        if not _r.cfg.experiment.strategy.skip_all_errors:
            self.current_state = self.failed
            self.on_enter_failed()
            raise err
        else:
            return err

    def _execute_action_failsafe(self, a: ActionMessage) -> Optional[ExceptionMessage]:
        _r = ExperimentRegistry()
        err = None
        if a.action_type not in (ActionTypeDef.DONE, ActionTypeDef.FAIL):
            err = _r.exp.execute_action(a)
        elif a.action_type == ActionTypeDef.DONE:
            logger.warning(f"Task done! Skipping action now...")
        elif a.action_type == ActionTypeDef.FAIL:
            if _r.cfg.experiment.strategy.skip_all_errors:
                logger.warning('Skipping ActionTypeDef.FAIL...')
                err = ExceptionMessage(
                    code=error_code.plan_failed,
                    message='ActionTypeDef.FAIL...',
                )

        # need not to handle grasp failure here
        if err is not None and err.code != error_code.grasp_failed:
            # execution failed
            _r.exp.controller.actuator.switch_mode(mode="primitive",robot="all")
            _r.exp.controller.actuator.open_gripper()
            # TODO: remove this
            if py_cli_interaction.parse_cli_bool('Whether to MoveUP two robots?', default_value=True)[0]:
                _r.exp.controller.smartMoveUpMovel(up_trans_first=[0, 0, 0.06],
                                                   up_trans_second=[0, 0, 0.06])

            # reset the robot to the home position
            _r.exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))

            if not _r.cfg.experiment.strategy.skip_all_errors:
                raise err
        return err

    def _debug_visualize_points(self, a: ActionMessage, o: ObservationMessage):
        _r = ExperimentRegistry()
        left_pick_point_in_virtual, right_pick_point_in_virtual = _r.exp.get_pick_points_in_virtual(a)
        left_place_point_in_virtual, right_place_point_in_virtual = _r.exp.get_place_points_in_virtual(a)

        visualize_pc_and_grasp_points(
            o.raw_virtual_pts,
            left_pick_point=left_pick_point_in_virtual[:3],
            right_pick_point=right_pick_point_in_virtual[:3],
        )

        visualize_pc_and_grasp_points(
            o.raw_virtual_pts,
            left_pick_point=left_place_point_in_virtual[:3],
            right_pick_point=right_place_point_in_virtual[:3],
        )

    def _finalize_after_action(self, a: ActionMessage, err: ExceptionMessage):
        _r = ExperimentRegistry()
        if not _r.cfg.experiment.compat.only_capture_pcd_before_action:
            logger.info("stage 3.4: capture pcd after action")
            _r.exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))
            obs, _ = _r.exp.get_obs()
            self._latest_logger.log_pcd_raw("end", obs.raw_virtual_pcd, only_npz=True)
            self._latest_logger.log_rgb("end", obs.rgb_img)
            self._latest_logger.log_mask("end", obs.mask_img)
            self._latest_logger.log_pcd_processed("end", obs.valid_virtual_pcd, only_npz=True)


        if (a.action_type in (ActionTypeDef.DONE, ActionTypeDef.FAIL)
                or (err is not None and err.code not in [error_code.grasp_failed, error_code.plan_failed])):
            self.current_state = self.failed
            self.on_enter_failed()
            return
 
    def _log_status(self, status: str):
        if self._latest_logger is not None:
            self._latest_logger.log_status(status)

    def _close_logger(self):
        if self._latest_logger is not None:
            try:
                self._latest_logger.finalize()
                self._latest_logger.log_processed_file(str(AnnotationFlag.COMPLETED.value))
                self._latest_logger.close()
                self._latest_logger = None
            except yaml.representer.RepresenterError as e:
                logger.error(e)
                logger.debug(self._latest_logger._metadata)

    def _action_loop(
            self
    ) -> Tuple[
        Optional[ObservationMessage],
        Optional[ActionMessage],
        Optional[PredictionMessage],
        Optional[ExceptionMessage]
    ]:
        _r = ExperimentRegistry()
        cfg, exp = _r.cfg, _r.exp
        logger.info("Starting Episode {}, Object {}, Trial {}, Step {}".format(
            _r.episode_idx, _r.object_id, _r.trial_idx, self.step_idx)
        )

        # reset the robot to the home position
        exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))
        # create logger
        self._init_logger()
        # take point cloud
        logger.info("stage 3.1: capture pcd before action")
        obs, err = self._get_observation()
        self._log_before_action(obs)
        # decision
        logger.info("stage 3.2: model inference")
        prediction_message, action_message, err = _r.running_inference.predict_action(
            obs,
            action_type=self._get_action_type(),
            vis=cfg.inference.args.vis_action,
        )
        # TODO: handle drag action if predicted action is not reachable
        self._latest_inference, self._latest_action = prediction_message, action_message
        self._log_after_prediction(prediction_message, action_message)
        if err is not None:
            return obs, action_message, prediction_message, self._abort_with_error(err)
        # after decision
        if cfg.experiment.compat.debug:
            self._debug_visualize_points(action_message, obs)
        # execute decision
        logger.info(f"stage 3.3: execute action {action_message.action_type}, pick_points: {action_message.pick_points}, "
                    f"place_points: {action_message.place_points}")
        err = self._execute_action_failsafe(action_message)
        if err is None or err.code == error_code.grasp_failed:
            logger.warning('None / grasp failure error is omitted...')
        else:
            logger.debug(f'Execution error: {err}')
            return obs, action_message, prediction_message, self._abort_with_error(err)
        #  after action
        self._finalize_after_action(action_message, err)
        self.condition.garment_consecutive_drag_num = 0

        return obs, action_message, prediction_message, err

    def on_enter_crumpled(self):
        self.current_state = ObjectStateDef.CRUMPLED
        # log status
        if self.step_idx == 1:
            self._log_status("begin")
        else:
            self._log_status("action")
        self._close_logger()
        _r = ExperimentRegistry()
        logger.info("Starting Episode {}, Object {}, Trial {}, Step {}".format(
            _r.episode_idx, _r.object_id, _r.trial_idx, self.step_idx)
        )
        _, _, _, err = self._action_loop()
        if self._latest_action.action_type == ActionTypeDef.FLING and \
                err is not None and err.code != error_code.grasp_failed:
            self.update_need_drag_condition(True)

        # protected section
        self.step_idx += 1
        self._latest_observation, self._latest_err = _r.exp.get_obs()
        self.update_condition()
        
        # test mode
        if getattr(_r.cfg.inference.args, 'test_mode', False):
            _r.running_inference.update_test_info(status='action')
    
    def on_enter_need_drag(self):
        self.condition.garment_consecutive_drag_num += 1
        _r = ExperimentRegistry()

        if self._latest_action.action_type == ActionTypeDef.FLING:
            err = self._try_drag(mode='crumpled')
        elif self._latest_action.action_type == ActionTypeDef.FOLD_1_1:
            err = self._try_drag(mode='folded_once')
        else:
            err = None

        # We only handle plan failure here. The grasp failure is handled inside controller.
        if err is None or err.code == error_code.grasp_failed:
            logger.warning('None / Grasp failure error is discarded...')
            err = None
        else:
            logger.warning('Drag is also failed, switch to Lift action...')
            err = self._try_lift_to_center()

        if err is None:
            self.update_need_drag_condition(False)
        
        # protected section
        self.step_idx += 1
        self.update_condition()
        
        # test mode
        if getattr(_r.cfg.inference.args, 'test_mode', False):
            _r.running_inference.update_test_info(status='action')

    def on_enter_unreachable(self):
        _r = ExperimentRegistry()
        self._try_lift_to_center()

        # protected section
        self.step_idx += 1
        self.update_condition()
        
        # test mode
        if getattr(_r.cfg.inference.args, 'test_mode', False):
            _r.running_inference.update_test_info(status='action')
            
    def folding_mode_template(self):
        # log status
        if self.step_idx == 1:
            self._log_status("begin")
        else:
            self._log_status("action")
        self._close_logger()
        _r = ExperimentRegistry()
        _, action_message, _, err = self._action_loop()

        if err is not None:
            if action_message.action_type == ActionTypeDef.FOLD_1_1:
                logger.debug('Could not execute fold action, need drag now!')
                # We only drag for fold1
                self.update_need_drag_condition(True)
        else:
            self.step_idx += 1
            self._latest_observation, self._latest_err = _r.exp.get_obs()

        self.update_condition()
        logger.debug(self.condition)

        # test mode
        if getattr(_r.cfg.inference.args, 'test_mode', False):
            _r.running_inference.update_test_info(status='action')

    def on_enter_smoothed(self):
        logger.debug('Enter smoothed state!!')
        self.current_state = ObjectStateDef.SMOOTHED
        _r = ExperimentRegistry()
        logger.debug(f'The garment rotation angle is: {self.condition.garment_rotation_angle}.')
        if self.condition.garment_rotation_angle is not None:
            _r.exp.move_rot_table(self.condition.garment_rotation_angle)
        else:
            if self.condition.garment_smoothing_style == GarmentSmoothingStyle.UP:
                _r.exp.move_rot_table(71)
            elif self.condition.garment_smoothing_style == GarmentSmoothingStyle.DOWN:
                _r.exp.move_rot_table(-102)
            elif self.condition.garment_smoothing_style == GarmentSmoothingStyle.LEFT:
                _r.exp.move_rot_table(0)
            elif self.condition.garment_smoothing_style == GarmentSmoothingStyle.RIGHT:
                _r.exp.move_rot_table(143)
        logger.debug('Table rotation done!!')
        self.folding_mode_template()

    def on_enter_folded_once(self):
        self.current_state = ObjectStateDef.FOLDED_ONCE

        _r = ExperimentRegistry()
        _r.exp.move_rot_table(-90)
        self.folding_mode_template()
        
    def on_enter_folded_twice(self):
        self.current_state = ObjectStateDef.FOLDED_TWICE
        logger.info("stage 3.5: the end")
        # reset the robot to the home position
        _r = ExperimentRegistry()
        _r.exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))

    def update_condition(self, ignore_manual_operation: bool = False):
        _r = ExperimentRegistry()
        # self._capture()  # TODO: remove this line in production
        if self.only_success or self.current_state.value == ObjectStateDef.CRUMPLED:
            object_state = _r.running_inference.predict_object_state(self._latest_observation, ignore_manual_operation=ignore_manual_operation)
        else:
            object_state = None
        garment_last_strip = self.condition.garment_last_strip
        if len(garment_last_strip) == 2:
            garment_last_strip.pop(0)
            garment_last_strip.append(self.garment_is_strip())
        else:
            garment_last_strip.append(self.garment_is_strip())
        payload = {
            "object_operable": _r.exp.is_object_on_table(self._latest_observation.mask_img),
            "object_reachable": _r.exp.is_object_reachable(self._latest_observation.mask_img),
            "object_organized_enough": (object_state.general_object_state == GeneralObjectState.ORGANIZED) if object_state is not None else self.condition.object_organized_enough,
            "garment_smoothing_style": object_state.garment_smoothing_style if object_state is not None else self.condition.garment_smoothing_style,
            "garment_rotation_angle": object_state.garment_rotation_angle if object_state is not None else self.condition.garment_rotation_angle,
            "garment_keypoint_parallel": object_state.garment_keypoint_parallel if object_state is not None else self.condition.garment_keypoint_parallel,
            "garment_need_drag": self.condition.garment_need_drag, # keep until operation
            "garment_consecutive_drag_num": self.condition.garment_consecutive_drag_num, # update during operation
            "garment_folded_once_success": self._latest_action is not None and self._latest_action.action_type == ActionTypeDef.FOLD_1_1,
            "garment_folded_twice_success": self._latest_action is not None and self._latest_action.action_type == ActionTypeDef.FOLD_2,
            "garment_last_is_strip": garment_last_strip,
        }
        logger.debug(f"update_condition: {payload}")
        self.condition = ObjectMachineConditions(**payload)
        
    def update_need_drag_condition(self, need_drag: bool):
        self.condition.garment_need_drag = need_drag

    def dump(self, img_path: str):
        self._graph().write_png(img_path)

class ObjectStateMachineOnlySmoothing(ObjectStateMachineOnlySuccess):
    """
    Object StateMachine (Only Smoothing)
    """
    unknown = State(name=ObjectStateDef.UNKNOWN, initial=True)
    crumpled = State(name=ObjectStateDef.CRUMPLED)
    unreachable = State(name=ObjectStateDef.UNREACHABLE)
    need_drag = State(name=ObjectStateDef.NEED_DRAG)
    success = State(name=ObjectStateDef.SUCCESS, final=True)
    failed = State(name=ObjectStateDef.FAILED, final=True)

    # transitions
    begin = (
            unknown.to(crumpled, cond=["observation_is_valid", "object_operable", "object_reachable"])
            | unknown.to(unreachable, unless="object_reachable")
            | unknown.to(failed, cond=["observation_is_valid"], unless="object_operable")
            | unknown.to(unknown)
    )

    fling = (
            crumpled.to(success, cond=["object_organized_enough", "object_operable"])
            | crumpled.to(need_drag, cond=["garment_need_drag", "object_operable"])
            | crumpled.to(unreachable, unless="object_reachable")
            | crumpled.to(unreachable, cond="garment_consecutive_is_strip")
            | crumpled.to(failed, cond="garment_step_threshold_exceeded")
            | crumpled.to(crumpled, unless="garment_consecutive_is_strip")
    )
    
    drag = (
            need_drag.to(crumpled, cond="object_operable", unless="garment_consecutive_is_strip")
            | need_drag.to(unreachable, unless="object_reachable")
            | need_drag.to(unreachable, cond="garment_consecutive_is_strip")
            | need_drag.to(failed, cond="garment_step_threshold_exceeded")
            | need_drag.to(failed, cond="garment_drag_num_exceeded")
            | need_drag.to(need_drag, unless="garment_consecutive_is_strip")
    )

    lift = (
        unreachable.to(crumpled, cond="object_reachable")
        | unreachable.to(unreachable, unless="object_reachable")
    )


    loop = (
            begin
            | lift
            | fling
            | drag
    )

class ObjectStateMachineDefault(ObjectStateMachineOnlySmoothing):
    """
    Object StateMachine
    """
    unknown = State(name=ObjectStateDef.UNKNOWN, initial=True)
    crumpled = State(name=ObjectStateDef.CRUMPLED)
    unreachable = State(name=ObjectStateDef.UNREACHABLE)
    need_drag = State(name=ObjectStateDef.NEED_DRAG)
    smoothed = State(name=ObjectStateDef.SMOOTHED)
    folded_once = State(name=ObjectStateDef.FOLDED_ONCE)
    folded_twice = State(name=ObjectStateDef.FOLDED_TWICE)
    success = State(name=ObjectStateDef.SUCCESS, final=True)
    failed = State(name=ObjectStateDef.FAILED, final=True)

    # transitions
    begin = (
            unknown.to(crumpled, cond=["observation_is_valid", "object_operable", "object_reachable"])
            | unknown.to(unreachable, unless="object_reachable")
            | unknown.to(failed, cond=["observation_is_valid"], unless="object_operable")
            | unknown.to(unknown)
    )

    fling = (
            crumpled.to(smoothed, cond=["object_organized_enough", "garment_keypoint_parallel", "object_operable"], unless="garment_consecutive_is_strip")
            | crumpled.to(need_drag, cond=["garment_need_drag", "object_operable"])
            | crumpled.to(unreachable, unless="object_reachable")
            | crumpled.to(unreachable, cond=["garment_consecutive_is_strip"])
            | crumpled.to(failed, cond="garment_step_threshold_exceeded")
            | crumpled.to(crumpled, unless="garment_consecutive_is_strip")
    )
    
    drag = (
            need_drag.to(smoothed, cond=["object_organized_enough", "garment_keypoint_parallel", "object_operable"], unless="garment_consecutive_is_strip")
            | need_drag.to(crumpled, cond="object_operable", unless="object_organized_enough")
            | need_drag.to(unreachable, unless="object_reachable")
            | need_drag.to(unreachable, cond=["garment_consecutive_is_strip"])
            | need_drag.to(failed, cond="garment_step_threshold_exceeded")
            | need_drag.to(failed, cond="garment_drag_num_exceeded")
            | need_drag.to(need_drag, unless="garment_consecutive_is_strip")
    )

    lift = (
        unreachable.to(crumpled, cond="object_reachable")
        | unreachable.to(unreachable, unless="object_reachable")
    )

    fold_once = (
            smoothed.to(folded_once, cond=["garment_is_rectangle", "garment_folded_once_success", "object_operable"])
            | smoothed.to(unreachable, unless="object_reachable")
            | smoothed.to(failed, cond="garment_step_threshold_exceeded", unless="object_operable")
            | smoothed.to(failed, unless="garment_is_rectangle")
            | smoothed.to(failed, unless="garment_folded_once_success")
            | smoothed.to(crumpled, unless="object_organized_enough")
            | smoothed.to(smoothed)
    )

    fold_twice = (
            folded_once.to(folded_twice, cond=["garment_folded_twice_success", "object_operable"])
            | folded_once.to(unreachable, unless="object_reachable")
            | folded_once.to(failed, cond="garment_step_threshold_exceeded", unless="object_operable")
            | folded_once.to(failed, unless="garment_folded_twice_success")
            | folded_once.to(crumpled, unless="object_organized_enough")
            | folded_once.to(folded_once)
    )

    end = (
            folded_twice.to(unreachable, unless="object_reachable")
            | folded_twice.to(failed, cond="garment_step_threshold_exceeded", unless="object_operable")
            | folded_twice.to(success)
    )

    loop = (
            begin
            | lift
            | fling
            | drag
            | fold_once
            | fold_twice
            | end
    )
    

def _test():
    seed = time.time() * 1e6  # 1695502691057083.8, 1695503526499119.0
    logger.info(f"seed={seed}")
    random.seed(seed)

    total_success = 0
    total_failed = 0
    num_steps_arr = []

    ObjectStateMachine(disp=False).dump('statemachine_object.png')
    for idx in range(100):
        m = ObjectStateMachine(disp=True)
        print(f">============== begin {idx + 1} trial ==============")
        while True:
            m.loop()
            if m.current_state.name == ObjectStateDef.SUCCESS:
                print("[result] =", m.current_state.name)
                break
            elif m.current_state.name == ObjectStateDef.FAILED:
                print("[result] =", m.current_state.name)
                break

        print(f">============== end {idx + 1} trial ==============")
        print("\n")
        if m.current_state.name == ObjectStateDef.SUCCESS:
            total_success += 1
            num_steps_arr.append(m.step_idx)
        else:
            total_failed += 1

    print(f"total success: {total_success}, total failed: {total_failed}")
    print(
        f"avg={sum(num_steps_arr) / (1e-8 + len(num_steps_arr))}, "
        f"max={max(num_steps_arr) if num_steps_arr else 'N/A'}, "
        f"min={min(num_steps_arr) if num_steps_arr else 'N/A'}"
    )


if __name__ == '__main__':
    _test()
