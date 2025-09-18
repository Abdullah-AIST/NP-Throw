import rtde_control
import rtde_receive
from rtde_control import RTDEControlInterface as RTDEControl

import numpy as np
import time


class ur5e_controller:
    def __init__(self, ip_address = "192.168.56.101",
                rtde_frequency = 480,
                control_freq = 60): 
        
        self.rtde_c = rtde_control.RTDEControlInterface(ip_address, rtde_frequency)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip_address, rtde_frequency)

        self.rtde_frequency = rtde_frequency
        self.control_freq = control_freq
        self.dt = 1.0/control_freq  # 2ms
        self.n_steps = int(rtde_frequency/control_freq)

        self.max_velocity = 3.14
        self.max_acceleration = 30
        self.lookahead_time = 0.03
        self.gain = 1000

        self.joint_q_home = np.array([0,-np.pi/4,np.pi/2,-np.pi/4,np.pi/2,0])
        self.joint_q = self.getActualQ()


        T = self.dt
        nsteps = self.n_steps
        dt = 1/nsteps
        t = np.linspace(0, T, nsteps+1).reshape(-1, 1)[1:]
        t_mat = np.concatenate([t**3, t**2, t], axis=-1).T
        A_mat = np.array([[1*T**3, 1*T**2, 1*T],
                    [1/4*T**3, 1/3*T**2, 1/2*T],
                    [1/10*T**3, 1/6*T**2, 1/3*T]])
        coeff = np.linalg.solve(A_mat, np.array([1, 1, 1]))
        coeff = coeff @ t_mat
        self.coeff = coeff - coeff.sum()*dt + 1



    def reset_traj(self, initJointPos=None):

        if initJointPos is None:
            initJointPos = self.joint_q_home
        diff = np.abs(self.joint_q[0] - initJointPos[0])
        #print(self.joint_q[0], initJointPos[0], diff)
        self.moveJ(initJointPos)

        #if diff > np.pi/2:
        #    print("MoveJ")
        #    self.moveJ(initJointPos)
        #else:
        #    print("MoveL")
        #    self.moveL(initJointPos)
        self.joint_q =  initJointPos.copy()
        self.joint_qd = np.zeros_like(self.joint_q)
        self.joint_qdd = np.zeros_like(self.joint_q)

        return initJointPos

    def moveJ(self, joint_pos):
        self.rtde_c.moveJ(joint_pos, speed=0.5, acceleration=0.5)
    def moveL(self, joint_pos):
        self.rtde_c.moveL_FK(joint_pos, speed=0.25, acceleration=0.25)

    def getActualQ(self):
        return np.array(self.rtde_r.getActualQ())

    def getActualQd(self):
        return np.array(self.rtde_r.getActualQd())
    
    def servoJ_accelaration(self, joint_qdd, smooth=True):
        """ move to joint position with servoJ """
        """ Implicit integration of joint acceleration to joint position and velocity """
        for step in range(self.n_steps):
            if smooth:
                joint_qdd_sm = self.joint_qdd + self.coeff[step]*(joint_qdd.copy() - self.joint_qdd)
            else:
                joint_qdd_sm = joint_qdd.copy()
            self.joint_q  = self.joint_q  + self.joint_qd  * self.dt/self.n_steps + 0.5*joint_qdd_sm * (self.dt/self.n_steps)**2
            self.joint_qd = self.joint_qd + joint_qdd_sm * self.dt/self.n_steps

            t_start = self.rtde_c.initPeriod()
            #self.rtde_c.servoJ(self.joint_q, self.max_velocity, self.max_acceleration, self.dt/self.n_steps, self.lookahead_time, self.gain)
            self.rtde_c.speedJ(self.joint_qd, self.max_acceleration, self.dt/self.n_steps)
            self.rtde_c.waitPeriod(t_start)
        
        self.joint_qdd = joint_qdd.copy()

    def servoJ_position(self, joint_q):
        """ move to joint position with servoJ """
        t_start = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(joint_q, self.max_velocity, self.max_acceleration, self.dt/self.n_steps, self.lookahead_time, self.gain)
        self.rtde_c.waitPeriod(t_start)
        self.joint_q = joint_q.copy()   

    def servostop(self):
        #self.rtde_c.servoStop()
        self.rtde_c.speedStop()
    def stopScript(self):
        self.rtde_c.stopScript()


if __name__ == "__main__":
    controller = ur5e_controller()
    controller.init_traj(controller.joint_q_home)
    controller.servoJ_accelaration(np.zeros_like(controller.joint_q_home))
    controller.stopScript() 