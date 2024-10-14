class RobotActuator:
    def __init__(self,
                 init_parser,
                 refPose,
                 fcl_planner,
                 left_end_link=28,
                 right_end_link=9) -> None:
        """
        You have to implement this function
        """
        self.init_parser = init_parser
        
    def switch_mode(self):
        """
        You have to implement this function
        """
        pass
            
    def op_gripper(self,robot=None,width:float=None):
        """
        You have to implement this function
        """
        pass
    
    def getJointSpaceState(self):
        """
        You have to implement this function
        """
        pass