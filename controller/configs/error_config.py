# Error Config for Grasping Toolbox
# Author: Guangfei Zhu, Han Xue, Yutong Li

# from lib_py.base.error_base import ErrorBase
from easydict import EasyDict as edict

# error_code = ErrorBase("cloth_folding")
error_code = edict()
error_code.ok = 0
error_code.empty_results = 101
error_code.wrong_data_shape = 201
error_code.wrong_data_type = 202
error_code.wrong_data_number = 203
error_code.wrong_path = 203

error_code.robot_failed = 301
error_code.robot_emergency = 302
error_code.grasp_failed = 303
error_code.ik_failed = 304
error_code.plan_failed = 305

error_code.camera_failed = 401
error_code.camera_control_path_failed = 402
error_code.segmentation_failed = 403