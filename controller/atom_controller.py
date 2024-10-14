"""Controller class.

Author: Dan Qing Lee
"""
import numpy as np
from controller.robot_actuator import RobotActuator

class AtomController():
    def __init__(self, config, check_grasp):
        """
        You have to implement this function
        """
        self.init_parser = InitParser(config)
        self.check_grasp = check_grasp
        self.actuator = RobotActuator(self.init_parser)
    
    def getRandomRefInitPose(self):
        """
        You have to implement this function
        """
        pass
        
    def singleIKWithQuatFromRef(self,
                                robot_type,
                                tcp_goal_quat = np.array([0.5,0.2,0.2,0,0,1,0]),  #x,y,z,w,rx,ry,rz
                                ref_pose_left = None,
                                ref_pose_right = None):
        """
        You have to implement this function
        """
        pass
    
    def checkCollisionState(self, left_joints_value, right_joints_value):
        """
        You have to implement this function
        """
        pass
    
    def getDualRobotMinDistance(self, left_joints_value, right_joints_value, 
                                obj_names_vec_left=[ "leftlink4","leftlink5","leftlink6",
                                                     "leftlink7","leftgripper"],
                                obj_names_vec_right=["rightlink4","rightlink5","rightlink6",
                                                     "rightlink7","rightgripper"],
                                output=False):
        """
        You have to implement this function
        """
        pass

    def getRobotDesktopMinDistance(self, left_joints_value, right_joints_value, output=False):
        """
        You have to implement this function
        """
        pass
    
    def runDualRobotJPoseForPlan(self, start_point, left_robot_jp, right_robot_jp):
        """
        You have to implement this function
        """
        pass
    
    def calCartesianPath(self, waypoints):
        """
        You have to implement this function
        """
        pass
    
    def checkJacobiSvd(self, robot_type="left_robot", left_joints_position_np=None, right_joints_position_np=None):
        """
        You have to implement this function
        """
        pass
        
    def freeDrive(self):
        """
        You have to implement this function
        """
        pass
   
    def stopRobots(self):
        """
        You have to implement this function
        """
        pass

    def goToHome(self):
        """
        You have to implement this function
        """
        pass
    
    def smartMoveUpMovel(self,
                        up_trans_first=[0.0,0.0,0.3],
                        up_trans_second=[0.0,0.0,0.2], 
                        obj_names_vec_left=[ "leftlink4","leftlink5","leftlink6",
                                                "leftlink7","leftgripper"],
                        obj_names_vec_right=["rightlink4","rightlink5","rightlink6",
                                                "rightlink7","rightgripper"],
                        ):
        """
        You have to implement this function
        """
        pass

    def move_rot_table(self, degrees=None):
        """
        You have to implement this function
        """
        pass