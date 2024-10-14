from typing import Optional, Callable, Any, OrderedDict, Union

from omegaconf import DictConfig

from common.utils import singleton
from learning.inference_3d import Inference3D
from manipulation.experiment_real import ExperimentReal
from tools.debug_controller import Client as DebugClient


@singleton
class ExperimentRegistry:
    def __init__(self):
        self._cfg: Optional[DictConfig] = None
        self._debug_client: Optional[DebugClient] = None
        self._runtime_output_dir: Optional[str] = None
        self._inference: Optional[Inference3D] = None
        self._running_inference: Optional[Inference3D] = None
        self._model_class: Optional[Any] = None
        self._inference_model_state_dict: Optional[OrderedDict] = None
        self._pred_output_dir: Optional[str] = None
        self._exp: Optional[ExperimentReal] = None
        self._episode_idx: Optional[int] = None
        self._fixed_object_id: Optional[str] = None
        self._is_validate_object_id: Optional[Callable] = None
        self._object_id: Optional[str] = None
        self._trial_idx: Optional[int] = None
        pass

    @staticmethod
    def get_instance() -> 'ExperimentRegistry':
        return ExperimentRegistry()

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, v: DictConfig):
        self._cfg = v

    @property
    def debug_client(self):
        return self._debug_client

    @debug_client.setter
    def debug_client(self, v):
        self._debug_client = v

    @property
    def runtime_output_dir(self):
        return self._runtime_output_dir

    @runtime_output_dir.setter
    def runtime_output_dir(self, v):
        self._runtime_output_dir = v

    @property
    def inference(self):
        return self._inference

    @inference.setter
    def inference(self, v):
        self._inference = v

    @property
    def running_inference(self):
        return self._running_inference

    @running_inference.setter
    def running_inference(self, v):
        self._running_inference = v

    @property
    def model_class(self):
        return self._model_class

    @model_class.setter
    def model_class(self, v):
        self._model_class = v

    @property
    def inference_model_state_dict(self):
        return self._inference_model_state_dict

    @inference_model_state_dict.setter
    def inference_model_state_dict(self, v):
        self._inference_model_state_dict = v

    @property
    def pred_output_dir(self):
        return self._pred_output_dir

    @pred_output_dir.setter
    def pred_output_dir(self, v):
        self._pred_output_dir = v

    @property
    def exp(self):
        return self._exp

    @exp.setter
    def exp(self, v):
        self._exp = v

    def reset_exp(self):
        del self._exp

    @property
    def episode_idx(self):
        return self._episode_idx

    @episode_idx.setter
    def episode_idx(self, v):
        self._episode_idx = v

    @property
    def fixed_object_id(self):
        return self._fixed_object_id

    @fixed_object_id.setter
    def fixed_object_id(self, v):
        self._fixed_object_id = v

    @property
    def is_validate_object_id(self):
        return self._is_validate_object_id

    @is_validate_object_id.setter
    def is_validate_object_id(self, v):
        self._is_validate_object_id = v

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, v):
        self._object_id = v

    @property
    def trial_idx(self):
        return self._trial_idx

    @trial_idx.setter
    def trial_idx(self, v):
        self._trial_idx = v