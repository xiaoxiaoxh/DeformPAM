import os
import numpy as np
from easydict import EasyDict as edict
import copy

"""
This config file is not complete, you have to add more settings if needed.
"""
config = edict()

# Motion config.
config.motion = edict()
config.motion.dual_arm_dist_thresh = 0.3
config.motion.max_v = 1
config.motion.max_a = 1.0
config.motion.max_jerk = 100
config.motion.level = 2

config.motion.grasp_waypts_z_offset = 0.08
config.motion.rotation_out = 40

config.motion.robot_l_ref_pos_list = np.array([
[ 0.43837932, -1.02951956, -1.68733442,  1.06657863,  1.36298525,  0.22826762, -0.32809168],
[ 0.12733249, -1.14912379 ,-1.48523974 , 1.0273844 ,  1.94765341,  0.64911348,  1.28888762],
[ 0.2132819,  -0.99996173, -1.57893276,  1.21044612,  1.49251235, -0.49963018,  1.14422297],
[ 1.10960054, -0.78696096, -1.89713085,  2.19092131,  1.1996913,   0.15369458,  0.65326101],
[ 1.34325838, -0.90656817, -2.17600942,  1.68509507,  1.44083893,  0.19079618,  0.58354735]])

config.motion.robot_r_ref_pos_list = np.array([
[ 0.18713713, -1.3737731,   1.51345432,  0.93545473, -2.2431531,   0.39889285, -0.89678961],
[-0.02665293, -1.10325444,  1.36245263,  1.38844883, -1.8139075,  -0.3023797,  -0.96229738],
[-0.85440415, -1.01831317,  1.55035615,  2.30124211, -1.58312726, -0.1233367,  -0.68905169],
[-0.20115604, -1.27033174,  1.21532595,  2.1806035,  -1.49549878,  0.0924382,  -0.06044285],
[ 0.1021216,  -1.10653412,  1.03909075,  1.51350105, -1.96533644,  0.18870603,-0.17781955],
[ 0.35787621, -1.48659515,  1.24066424,  0.8087666,  -2.55008602,  0.32375631, -0.58702588],
[-0.42251569, -0.86446238,  1.27819788,  2.0286994,  -1.89430869, -0.49906728,  -1.56493604],
[-0.87051785, -0.92639488,  1.36287344,  2.20197392, -1.89532375,  0.79010463,  -1.49049997],
[ 0.39776942, -1.1565783,   0.57648182,  1.05942452, -1.75674462,  0.85980558,  -1.50049579],
[-0.76231903, -1.1543299,   1.72470474,  2.04636383, -1.77626216, -0.23733471, -0.05820146]])


config.motion.robot_l_home = np.array([0.66, 0.27, -0.22, 1.82, 0.14, 0.83, 0.57])
config.motion.robot_r_home = np.array([-0.85, 0.09, 0.73, 1.75, -0.44, 0.84, -0.11])

config.motion.robot_l_ready = np.array([0.5430026021596961, -0.3554782800554828, -1.351822170991266, 1.4603189087794535, 1.1970456404196057, 0.10354042431913917, -0.6040010476333671])
config.motion.robot_r_ready = np.array([-0.06530063831398528, -0.47668352226505195, 0.9104904546522099, 1.592879170054106, -1.2694959952333644, 0.26092488781793594, 0.9537960901044955])

config.motion.before_fling_pose_l = np.array([14.23, 32.61, -60.15, 90.51, 21.46, -4.17, -28.54])*np.pi/180
config.motion.before_fling_pose_r = np.array([32.35, -21.83, 65.06, 87.43, -56.95,-6.87,  78.9])*np.pi/180
'''
plan velocity and acceleration factor
'''
config.motion.default_velocity_factor = 1.0
config.motion.default_acceleration_factor = 0.7

'''
fixed velocity for robot!6
Not recommended for change
'''
config.motion.max_jnt_vel = [2.5]*7 # [3.0]*7
config.motion.max_jnt_acc = [2.5]*7 # [3.0]*7
config.motion.time_interval = 0.001

'''
IK joints limit
Modify joint 7 to [-pi,pi]
'''
config.motion.left_joint_limit_lowers = [-2.7925, -2.2689, -2.9671, -1.8675, -2.9671, -1.3963, -1.5707]
config.motion.right_joint_limit_lowers = [-2.7925, -2.2689, -2.9671, -1.8675, -2.9671, -1.3963, -1.5707]
config.motion.left_joint_limit_uppers = [2.7925, 2.2689, 2.9671, 2.6878, 2.9671, 4.5379, 1.5707]
config.motion.right_joint_limit_uppers = [2.7925, 2.2689, 2.9671, 2.6878, 2.9671, 4.5379, 1.5707]

config.motion.ref_ik_mode = "default_ik_ref"
config.motion.q_ref_pose= np.array([40,-68,-138,73,104,-21,121,44,-65,-119,128,100,-26,20])*np.pi/180

'''
Fold one action config
'''
config.fold_one_action = edict()
config.fold_one_action.l_home_pose = np.array([82.30, -86.37, -155.21, 106.61, 114.81, -22.92, 3.82])*np.pi/180
config.fold_one_action.r_home_pose = np.array([-51.31, -87.22, 135.49, 101.81, -136.01, -21.51, 6.92])*np.pi/180

'''
Fold two action config 
'''
config.fold_two_action = edict()
config.fold_two_action.l_home_pose = np.array([82.30, -86.37, -155.21, 106.61, 114.81, -22.92, 3.82])*np.pi/180
config.fold_two_action.r_home_pose = np.array([-51.31, -87.22, 135.49, 101.81, -136.01, -21.51, 6.92])*np.pi/180

"------------------------------"

'''
config for Short-sleeve Shirts
'''
config_tshirt_short = edict(copy.deepcopy(config))

'''
config for Nuts
'''
config_nut = edict(copy.deepcopy(config))

'''
config for Rope
'''
config_rope = edict(copy.deepcopy(config))