"""This calss builds a mutiprocessing pool for the Flexiv robot.

Author: Danqing Li, Yongyuan Wang

"""

from multiprocessing import Pool
import time

class FlexivMutiprocessing():
    """This class keep synchronise of dual robot.
    """
    def __init__(self, pool_size=2):
        """
        Args:
            pool_size (int, optional): Pool maxnetwork is 2.
        """
        self.pool_size = pool_size

    def run(self, func, args):
        """
        Args:
            func (function): the function to run
            args (list): the arguments of the function
        Return:
            None
        """
        func(*args)

    def multicore(self, func1, args1, func2, args2, timeout=2, flag=True):
        """
        Args:
            func1 (function): the function to run
            args1 (list): the arguments of the function
            func2 (function): the function to run
            args2 (list): the arguments of the function
            timeout (int): the timeout in seconds
            flag (bool): the flag
        Return:
            None
        """
        with Pool(self.pool_size) as pool:
            result1 = pool.apply_async(self.run, (func1, args1))
            result2 = pool.apply_async(self.run, (func2, args2))

            if flag:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if result1.ready() and result2.ready():
                        break
                    time.sleep(0.1)
            else:
                try:
                    result1.get(timeout=timeout)
                except TimeoutError:
                    print("Function 1 timed out")

                try:
                    result2.get(timeout=timeout)
                except TimeoutError:
                    print("Function 2 timed out")

class testClass:
    def __init__(self):
        self.mp = FlexivMutiprocessing()
        
    def function1(self, val1, val2, val3):
        print("val1:", val1)
        print("func1:", val2 + val3)

    def function2(self, val1, val2):
        print("val1:", val1)
        print("func2:", val2)      

    def execute_functions(self, func1, args1, func2, args2):
        self.mp.multicore(func1, args1, func2, args2)  
        

if __name__ == "__main__":
    OtherClass = testClass()

    OtherClass.execute_functions(OtherClass.function1, ("question1", 0, 1), OtherClass.function2, ("answer1", 1))
    time.sleep(0.1)
    OtherClass.execute_functions(OtherClass.function1, ("question2", 2, 3), OtherClass.function2, ("answer2", 5))


