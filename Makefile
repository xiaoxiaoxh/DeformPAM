SHELL = /bin/bash

TASK_TYPE = tshirt_short
# TASK_TYPE = nut
# TASK_TYPE = rope
TASK_VERSION = debug

EXP_NAME_SUPERVISED = ${TASK_TYPE}-supervised_${TASK_VERSION}
EXP_NAME_SUPERVISED_CLASSIFICATION_DETECTION = ${TASK_TYPE}-supervised_classification_detection_${TASK_VERSION}
EXP_NAME_SUPERVISED_VR = Tshirt-short-action14-dataset-v7_real-hybrid_zero_center_supervised_vr
EXP_NAME_SUPERVISED_DEBUG = ${TASK_TYPE}-supervised_debug
EXP_NAME_FINETUNE = ${TASK_TYPE}-finetune_${TASK_VERSION}
EXP_NAME_FINETUNE_REWARD_PREDICTION = ${TASK_TYPE}-finetune_reward_prediction_${TASK_VERSION}
EXP_NAME_FINETUNE_DEBUG = ${TASK_TYPE}-finetune_debug
EXP_NAME_TEST = ${TASK_TYPE}-action14_test

LOGGING_TAG_SUPERVISED = ${TASK_TYPE}_supervised_${TASK_VERSION}
LOGGING_TAG_SUPERVISED_CLASSIFICATION_DETECTION = ${TASK_TYPE}_supervised_classification_detection_${TASK_VERSION}
LOGGING_TAG_SUPERVISED_DEBUG = ${TASK_TYPE}_supervised_debug
LOGGING_TAG_FINETUNE = ${TASK_TYPE}_finetune_${TASK_VERSION}
LOGGING_TAG_FINETUNE_DEBUG = ${TASK_TYPE}_finetune_debug
LOGGING_TAG_TEST= ${TASK_TYPE}_test
RAW_LOG_NAMESPACE_SUPERVISED = experiment_supervised
RAW_LOG_NAMESPACE_FINETUNE = experiment_finetune
RAW_LOG_NAMESPACE_TEST = experiment_test
START_EPISODE = 0

CONFIG_MVCAM_DEV = $(shell lsusb | grep MindVision | awk 'END { if (NR==0 || $$2=="") print "--"; else print "/dev/bus/usb/"$$2"/"$$4;}' | head -c -2)  #/dev/bus/usb/002/002
SECOND_ROBOT_LEFT_CLIENT_PATH = /home/xuehan/2.10.9_flexiv_elements/FlexivElements_v2.10.9_left_v
SECOND_ROBOT_RIGHT_CLIENT_PATH = /home/xuehan/2.10.9_flexiv_elements/FlexivElements_v2.10.9_right_v

ROBOT_LEFT_CLIENT_PATH = /home/xuehan/2.10.9_flexiv_elements/FlexivElements_v2.10.9_left_w
ROBOT_RIGHT_CLIENT_PATH = /home/xuehan/2.10.9_flexiv_elements/FlexivElements_v2.10.9_right_w

DISPLAY_PORT = 12000
ANNO_PORT = 13240
MULTI_POSE_NUM = 10
COMPARE_K = 16
INFERENCE_POINT_NUM = 16
KEYPOINT_NUM = 4

CALIBRATION_PATH = /path/to/calibration/folder
# select the camera to use
CAMERA_PARAM = phoxi_camera_with_rgb

SUPERVISED_MODEL_CKPT_PATH = /path/to/supervised_model/folder
TEST_MODEL_CKPT_PATH = /path/to/test_model/folder

.PHONY: all
all:
	@echo "Please specify a target"

manipulation.prerun:
	@perm=$$(ls -ld ${CONFIG_MVCAM_DEV} | cut -b 1-10); \
	if [ $$perm != "crwxrwxrwx" ]; then \
		echo "Changing permissions of ${CONFIG_MVCAM_DEV} to 777"; \
		sudo chmod 777 ${CONFIG_MVCAM_DEV}; \
	else \
  		echo "${CONFIG_MVCAM_DEV} permissions have been set!"; \
  	fi
	umask 002

test_experiment: manipulation.prerun
	export PYTHONPATH=$$PYTHONPATH:$(shell pwd); \
        python \
			manipulation/experiment_real.py \
			experiment.compat.calibration_path=${CALIBRATION_PATH} \
			camera_param=${CAMERA_PARAM} \

# only real data collection (no training)
supervised.run_real: manipulation.prerun
	python \
        run.py --config-name experiment_supervised_${TASK_TYPE}.yaml \
        hydra.job.chdir=True \
        experiment.runtime_training_config_override.logger.experiment_name=${EXP_NAME_SUPERVISED} \
        experiment.strategy.skip_all_errors=True \
        experiment.strategy.start_episode=0 \
        experiment.strategy.random_exploration.enable=False \
        experiment.strategy.random_lift_in_each_trial=False \
        experiment.compat.calibration_path=${CALIBRATION_PATH} \
        camera_param=${CAMERA_PARAM} \
        logging.tag=${LOGGING_TAG_SUPERVISED} \
        inference.model_path=${SUPERVISED_MODEL_CKPT_PATH} \
        inference.args.vis_action=False \
        inference.args.vis_all_fling_pred=False \
        inference.args.manual_operation.remote_args.enable=False \
        inference.args.manual_operation.remote_args.display_port=${DISPLAY_PORT} \
        inference.args.manual_operation.remote_args.anno_port=${ANNO_PORT} \
        inference.args.only_success=False \
        inference.args.only_smoothing=True \
        experiment.compat.use_real_robots=True

scripts.remote_operation:
	export PYTHONPATH=$$PYTHONPATH:$(shell pwd); \
        python ./tools/remote_operation/client.py --host 192.168.2.223 \
                --display_port ${DISPLAY_PORT} \
                --anno_port ${ANNO_PORT}

scripts.run_supervised_annotation:
	export PYTHONPATH=$$PYTHONPATH:$(shell pwd); \
        python ./tools/run_annotation --raw_log_namespace ${RAW_LOG_NAMESPACE_SUPERVISED} \
                --root_dir ${LOGGING_TAG_SUPERVISED} \
                --object_type ${TASK_TYPE} \
                --annotation_type new_pipeline_supervised \
                --multi_pose_num ${MULTI_POSE_NUM} \
                --keypoint_num ${KEYPOINT_NUM}

# supervised training using VR data
supervised.train_vr:
	python train_supervised.py --config-name train_supervised_tshirt_short_vr.yaml \
		hydra.job.chdir=True \
		logger.experiment_name=${EXP_NAME_SUPERVISED_VR} \
		logger.run_name=action14_supervised_vr \

# supervised training using real data
supervised.train_real:
	python train.py --config-name train_supervised_${TASK_TYPE}_real.yaml \
		hydra.job.chdir=True \
		logger.experiment_name=${EXP_NAME_SUPERVISED} \
		logger.run_name=action14_supervised_real \
                runtime_datamodule.namespace=${RAW_LOG_NAMESPACE_SUPERVISED} \
                runtime_datamodule.tag=${LOGGING_TAG_SUPERVISED} \
                runtime_datamodule.num_multiple_poses=${MULTI_POSE_NUM}

#  test sampling
test_model:
	python test.py \
		--config-path outputs/2024-03-23/23-17-41 \
		hydra.job.chdir=True \
		+save_bson=True \
                +scheduler_type=ddpm \
                +num_inference_steps=10 \
                +ddim_eta=0.0

# only real data collection (no training)
finetune.run_real: manipulation.prerun
	python \
        run.py --config-path config/finetune_experiment \
        --config-name experiment_finetune_${TASK_TYPE}.yaml \
        hydra.job.chdir=True \
        experiment.runtime_training_config_override.logger.experiment_name=${EXP_NAME_FINETUNE} \
        experiment.strategy.skip_all_errors=True \
        experiment.strategy.start_episode=0 \
        experiment.strategy.random_exploration.enable=False \
        experiment.strategy.random_lift_in_each_trial=False \
        experiment.compat.calibration_path=${CALIBRATION_PATH} \
        camera_param=${CAMERA_PARAM} \
        logging.namespace=${RAW_LOG_NAMESPACE_FINETUNE} \
        logging.tag=${LOGGING_TAG_FINETUNE} \
        inference.model_path=${SUPERVISED_MODEL_CKPT_PATH} \
        inference.args.vis_action=True \
        inference.args.vis_all_fling_pred=True \
        inference.args.vis_pred_order=False \
        inference.model_name=last\
        inference.args.manual_operation.remote_args.enable=False \
        inference.args.manual_operation.remote_args.display_port=${DISPLAY_PORT} \
        inference.args.manual_operation.remote_args.anno_port=${ANNO_PORT} \
        inference.args.model.diffusion_head_params.scheduler_type=ddim \
        inference.args.model.diffusion_head_params.num_inference_steps=100 \
        inference.args.model.diffusion_head_params.ddim_eta=1.0 \
        +inference.args.model.use_dpo_reward_for_inference=False \
        +inference.args.model.random_select_diffusion_action_pair_for_inference=True \
		+inference.args.model.inference_point_num=${INFERENCE_POINT_NUM} \
        inference.args.only_success=False \
        inference.args.only_smoothing=True \
        experiment.compat.use_real_robots=True

scripts.run_finetune_sort_annotation:
	export PYTHONPATH=$$PYTHONPATH:$(shell pwd); \
        python ./tools/run_annotation --raw_log_namespace ${RAW_LOG_NAMESPACE_FINETUNE} \
                --root_dir ${LOGGING_TAG_FINETUNE} \
                --object_type ${TASK_TYPE} \
                --annotation_type new_pipeline_finetune_sort \
                --multi_pose_num ${MULTI_POSE_NUM} \
                --K ${COMPARE_K} \
                --keypoint_num ${KEYPOINT_NUM}

# finetuning using real data
finetune.train_real:
	python train.py --config-path config/finetune_experiment \
                --config-name train_finetune_${TASK_TYPE}_real.yaml \
		hydra.job.chdir=True \
		logger.experiment_name=${EXP_NAME_FINETUNE} \
		logger.run_name=action14_finetune_real \
                runtime_datamodule.namespace=${RAW_LOG_NAMESPACE_FINETUNE} \
                runtime_datamodule.tag=${LOGGING_TAG_FINETUNE} \
                +runtime_datamodule.manual_num_rankings_per_sample=${COMPARE_K} \
                +model.reference_model_path=${SUPERVISED_MODEL_CKPT_PATH} \

# finetuning using real data with reward prediction
finetune.train_real_reward_prediction:
	python train.py --config-path config/finetune_experiment \
                --config-name train_finetune_reward_prediction_${TASK_TYPE}_real.yaml \
		hydra.job.chdir=True \
		logger.experiment_name=${EXP_NAME_FINETUNE_REWARD_PREDICTION} \
		logger.run_name=action14_finetune_real \
                runtime_datamodule.namespace=${RAW_LOG_NAMESPACE_FINETUNE} \
                runtime_datamodule.tag=${LOGGING_TAG_FINETUNE} \
                +runtime_datamodule.manual_num_rankings_per_sample=${COMPARE_K} \
                +model.reference_model_path=${SUPERVISED_MODEL_CKPT_PATH} \

STEP_NUM_PER_TRIAL = $(shell if [ ${TASK_TYPE} = "tshirt_short" ]; then echo 10; \
				elif [ ${TASK_TYPE} = "tshirt_long" ]; then echo 10; \
                elif [ ${TASK_TYPE} = "nut" ]; then echo 15; \
                elif [ ${TASK_TYPE} = "rope" ]; then echo 20; \
                else echo "unknown_task_type"; fi)
ACTION_TYPE_OVERRIDE = $(shell if [ ${TASK_TYPE} = "tshirt_short" ]; then echo "null"; \
				elif [ ${TASK_TYPE} = "tshirt_long" ]; then echo "null"; \
                elif [ ${TASK_TYPE} = "nut" ]; then echo "sweep"; \
                elif [ ${TASK_TYPE} = "rope" ]; then echo "single_pick_and_place"; \
                else echo "unknown_task_type"; fi)
DDIM_ETA = $(shell if [ ${TASK_TYPE} = "tshirt_short" ]; then echo 0.0; \
                elif [ ${TASK_TYPE} = "tshirt_long" ]; then echo 0.0; \
                elif [ ${TASK_TYPE} = "nut" ]; then echo 0.0; \
                elif [ ${TASK_TYPE} = "rope" ]; then echo 1.0; \
                else echo "unknown_task_type"; fi)
# only real data collection (no training)
test_real: manipulation.prerun
	python \
        run.py --config-path config/finetune_experiment \
        --config-name experiment_finetune_${TASK_TYPE}.yaml \
        hydra.job.chdir=True \
        experiment.runtime_training_config_override.logger.experiment_name=${EXP_NAME_TEST} \
        experiment.strategy.skip_all_errors=True \
        experiment.strategy.start_episode=0 \
		experiment.strategy.step_num_per_trial=${STEP_NUM_PER_TRIAL} \
        experiment.strategy.random_exploration.enable=False \
        experiment.strategy.random_lift_in_each_trial=False \
        experiment.compat.calibration_path=${CALIBRATION_PATH} \
        experiment.compat.only_capture_pcd_before_action=False \
        camera_param=${CAMERA_PARAM} \
        logging.namespace=${RAW_LOG_NAMESPACE_TEST} \
        logging.tag=${LOGGING_TAG_TEST} \
        inference.model_path=${TEST_MODEL_CKPT_PATH} \
        inference.args.vis_action=True \
        inference.args.vis_all_fling_pred=True \
        inference.args.vis_pred_order=False \
        inference.args.vis_pred_order_num=4 \
        inference.model_name=last\
        +inference.args.test_mode=True \
        inference.args.manual_operation.enable=True \
        inference.args.manual_operation.remote_args.enable=False \
        inference.args.manual_operation.remote_args.display_port=${DISPLAY_PORT} \
        inference.args.manual_operation.remote_args.anno_port=${ANNO_PORT} \
        inference.args.model.diffusion_head_params.scheduler_type=ddim \
        inference.args.model.diffusion_head_params.num_inference_steps=100 \
        inference.args.model.diffusion_head_params.ddim_eta=${DDIM_ETA} \
        +inference.args.model.use_dpo_reward_for_inference=True \
        +inference.args.model.dpo_reward_sample_num=100 \
        +inference.args.model.random_select_diffusion_action_pair_for_inference=False \
        inference.args.model.manually_select_diffusion_action_pair_for_inference=False \
		+inference.args.model.inference_point_num=${INFERENCE_POINT_NUM} \
        inference.args.action_type_override.type=${ACTION_TYPE_OVERRIDE} \
        inference.args.only_success=False \
        inference.args.only_smoothing=True \
        inference.args.enable_record=False \
        experiment.compat.use_real_robots=True \

tools.capture_gt_pcd: manipulation.prerun
	python \
        tools/capture_canonical_general.py --config-name experiment_finetune_${TASK_TYPE}.yaml \
        hydra.job.chdir=False \
        camera_param=${CAMERA_PARAM} \
        experiment.compat.segmentation.grounding_dino_config_path=data/checkpoints/GroundingDINO_SwinT_OGC.cfg.py \
        experiment.compat.segmentation.grounding_dino_checkpoint_path=data/checkpoints/groundingdino_swint_ogc.pth \
        experiment.compat.segmentation.sam_checkpoint_path=data/checkpoints/sam_vit_b_01ec64.pth

debug.controller:
	python tools/debug_controller.py

scripts.check_annotation:
	export PYTHONPATH=$$PYTHONPATH:$(shell pwd); \
        python ./tools/run_annotation --exam_mode \
                --multi_pose_num ${MULTI_POSE_NUM} \
                --K ${COMPARE_K} \
                --keypoint_num ${KEYPOINT_NUM}

.PHONY: clean
clean:
	@echo "Cleaning up..."
