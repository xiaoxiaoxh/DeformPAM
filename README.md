<h3 align="center">
    🦾DeformPAM: Data-Efficient Learning for
    <br>Long-horizon <u>Deform</u>able Object Manipulation via
    <br><u>P</u>reference-based <u>A</u>ction Align<u>m</u>ent
</h3>

<p align="center">
    <a href="https://github.com/ChenWendi2001">Wendi Chen</a><sup>1*</sup>,
    <a href="https://hanxue.me">Han Xue</a><sup>1*</sup>,
    Fangyuan Zhou<sup>1</sup>,
    Yuan Fang<sup>1</sup>,
    <a href="https://www.mvig.org/">Cewu Lu</a><sup>1</sup>
    <br>
    <sup>1</sup>Shanghai Jiao Tong University,
    <sup>*</sup> indicates equal contribution
    <br>
</p>

<div align="center">
<a href='https://arxiv.org/abs/2410.11584'><img alt='arXiv' src='https://img.shields.io/badge/arXiv-2410.11584-red.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://deform-pam.robotflow.ai'><img alt='project website' src='https://img.shields.io/website-up-down-green-red/http/cv.lbesson.qc.to.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<img alt='powered by Pytorch' src='https://img.shields.io/badge/PyTorch-❤️-F8C6B5?logo=pytorch&logoColor=white'> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

## Contents

- [Contents](#contents)
- [📄 Introduction](#-introduction)
  - [Motivation](#motivation)
  - [Method](#method)
  - [Tasks and Hardware Setup](#tasks-and-hardware-setup)
- [⚙️ Environment Setup](#️-environment-setup)
  - [🧠 Learning Environment](#-learning-environment)
    - [🔽 Pre-trained Models](#-pre-trained-models)
  - [🤖 Real Environment](#-real-environment)
    - [📷 Camera](#-camera)
    - [🦾 Robot](#-robot)
  - [📦 Miscs](#-miscs)
    - [🔧 Tools](#-tools)
- [📕 Usage](#-usage)
  - [🔍 Inference](#-inference)
  - [📚 Train Your Own Model](#-train-your-own-model)
    - [Stage 1 (Supervised Learning)](#stage-1-supervised-learning)
    - [Stage 2 (Preference Learning)](#stage-2-preference-learning)

## 📄 Introduction

### Motivation

In long-horizon manipulation tasks, a probabilistic policy may encounter distribution shifts when imperfect policy fitting leads to unseen states. As time progresses, the deviation from the expert policy becomes more significant. Our framework employs Reward-guided Action Selection (RAS) to reassess sampled actions from the generative policy model, thereby improving overall performance.

<p align="center">
<img src="assets/teaser.png" alt="teaser" style="width:600px;" />
</p>

### Method

![pipeline](assets/pipeline.png)

- In stage ①, we assign actions for execution and annotate auxiliary actions for supervised learning in a real-world environment and train a supervised primitive model based on Diffusion.
- In stage ②, we deploy this model in the environment to collect preference data composed of annotated and predicted actions. These data are used to train a DPO-finetuned model.
- During stage ③ (inference), we utilize the supervised model to predict actions and employ an implicit reward model derived from two models for Reward-guided Action Selection (RAS). The action with the highest reward is regarded as the final prediction.

### Tasks and Hardware Setup

The following figure illustrates the object states and primitives of each task. Beginning with a random complex state of an object, multiple steps of action primitives are performed to gradually achieve the target state.
<p align="center">
<img src="assets/tasks_primitives.png" alt="Tasks and Primitives" style="width:660px;" />
</p>

Here is the hardware setup and tools used in our real-world experiments. Devices and tools marked with DP are not used in primitive-based methods.
<p align="center">
<img src="assets/hardware_setup.png" alt="Hardware Setup" style="width:400px;" />
</p>

## ⚙️ Environment Setup

### 🧠 Learning Environment

The learning code should work on environments that meet the following requirements:

- Modern Linux Distributions that are not in EOL.
- Python >= 3.8
- Pytorch >= 1.11.0
- CUDA >= 11.3

We recommend these combinations:

- Ubuntu 20.04
- Python = 3.8
- Pytorch = 1.11.0
- CUDA = 11.3

To setup the learning environment, you need to download and install CUDA from [here](https://developer.nvidia.com/cuda-downloads) in advance. Then, you should run the `setup-env.sh` script to setup all basic requirments except for GroundedSAM.


```bash
bash setup-env.sh
```

This script will automatically create an conda environment named `DeformPAM` and install dependent packages in it. You can modify this script to make it behaves differently.

Finally, see [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) for installation of Grounded-DINO and Segment-Anything.

#### 🔽 Pre-trained Models
We will release the pre-trained models soon. 

### 🤖 Real Environment

#### 📷 Camera

Our project should work on any commercial 3D cameras systems that produce colorful point cloud and RGB images. However, for the best performance, we recommend high-precision and high-resolution 3D cameras. In our experiment, we adopt [Photoneo MotionCam3D M+](https://www.photoneo.com/products/motioncam-3d-m-plus/) adn [Mech-Mind Mech-Eye LSR L](https://community.mech-mind.com/t/topic/2112) as the main 3D camera. However, if you are using custom cameras, please re-implement `get_obs()` method in [manipulation.experiment_real::ExperimentReal](manipulation/experiment_real.py#L408).

Please generate the calibration files and set the `CALIBRATION_PATH` in [Makefile](Makefile). You can take [tools/handeye_cali.py](handeye_cali.py) and [tools/find_world_transform_from_robot_cali.py](find_world_transform_from_robot_cali.py) for reference.

#### 🦾 Robot

Our experiments are conducted using two [Flexiv Rizon 4](https://www.flexiv.cn/en/product/rizon) robot arms through [Flexiv RDK](https://rdk.flexiv.com/en/). Please re-implement [controller.robot_actuator::RobotActuator](controller/robot_actuator.py#L1), [controller.atom_controller::AtomController](controller/atom_controller.py#L8), and [controller.controller::Controller](controller/controller.py#L6), whether you are using Flexiv or custom robot arms .

### 📦 Miscs

#### 🔧 Tools

Please refer to [tools/data_management/README.md](tools/data_management/README.md) for setting up the data management tools.

## 📕 Usage

### 🔍 Inference

You can modify the [`TASK_TYPE`](Makefile#L3), [`SUPERVISED_MODEL_CKPT_PATH`](Makefile#L47), and [`TEST_MODEL_CKPT_PATH`](Makefile#L48) in [Makefile](Makefile) and run the following command to conduct inference on the real-world environment.

```bash
make test_real
```

### 📚 Train Your Own Model

The training pipeline includes 2 stages, all wrapped as Makefie targets.

#### Stage 1 (Supervised Learning)

Run the following commands:
```bash
# Stage 1.1: collect supervised data
make supervised.run_real
# Stage 1.2: annotate supervised data
make scripts.run_supervised_annotation
# Stage 1.3: train supervised model
make supervised.train_real
```

#### Stage 2 (Preference Learning)
Settup the `SUPERVISED_MODEL_CKPT_PATH` in [Makefile](Makefile) to the path of the trained model in stage 1. Then run the following commands:
```bash
# Stage 2.1: collect on-policy data
make finetune.run_real
# Stage 2.2: train preference model
make scripts.run_finetune_sort_annotation
# Stage 2.3: train DPO-finetuned model
make finetune.train_real
```

## 🙏 Acknowlegement

- The data annotation tool and motion primitives used in our code is adapted from [UniFolding](https://github.com/xiaoxiaoxh/UniFolding).
