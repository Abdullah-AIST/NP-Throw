
from controller import ur5e_controller
import numpy as np
import time

control_freq = 20
controller = ur5e_controller(control_freq=control_freq)


#load trajs
runTime = 3.2
trajLen = int(control_freq*runTime) # 2 seconds
posTraj = np.load('scripts/jointPosAll.npy').reshape(-1,trajLen,6)
velTraj = np.load('scripts/jointVelAll.npy').reshape(-1,trajLen,6)
accTraj = np.load('scripts/jointAccAll.npy').reshape(-1,trajLen,6)

n = posTraj.shape[0]
for traj_id in range(10):
    
    print(f"Trajectory {traj_id}")
    # Move to initial joint position with a regular moveJ
    initJointPos = posTraj[traj_id,0,:]
    initJointPos = controller.reset_traj(initJointPos)
    #if traj_id == 0:
    #    time.sleep(5)
    time.sleep(5)
    arm_qpos = controller.getActualQ()
    arm_qvel = controller.getActualQd()

    Arm_qpos = []
    Arm_qvel = []
    Time = []
    elapsed_time_accum = 0.0

    # Execute 500Hz control loop for 2 seconds, each cycle is 2ms
    for i in range(trajLen):
        start_time = time.time() 
        joint_qdd = accTraj[traj_id,i,:]
        controller.servoJ_accelaration(joint_qdd,smooth=True)
        arm_qpos = controller.getActualQ()
        arm_qvel = controller.getActualQd()
        elapsed_time  = time.time() - start_time
        elapsed_time_accum += elapsed_time
        Time.append(elapsed_time_accum)
        Arm_qpos.append(arm_qpos)
        Arm_qvel.append(arm_qvel)


    controller.servostop()
    


    import matplotlib.pyplot as plt
    Arm_qpos = np.array(Arm_qpos)
    Arm_qvel = np.array(Arm_qvel)
    Time = np.array(Time)


    Data = np.hstack((Time.reshape(-1,1), Arm_qpos, Arm_qvel, posTraj[traj_id,:,:], velTraj[traj_id,:,:], accTraj[traj_id,:,:]))

    np.savetxt(f'./data/real/trajID{traj_id}_position_velocity_real_Throw_{control_freq}Hz.csv', Data, delimiter=',', header='Time,Actual Position,Desired Position,Actual Velocity', comments='')


#rtde_c.moveJ(joint_q_home)
controller.stopScript()