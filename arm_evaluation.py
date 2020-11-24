import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
import jmpc
import time
import sys
import settings
import shutil
#from PolicyNet import ExpertMixturePolicy as PolicyNet
#from PolicyNet_2Layer import ExpertMixturePolicy as PolicyNet_2Layer

if settings.enablePybulletTraining == True:
   print("enablePybulletTraining = True, close evaluation app")
   sys.exit(0)

mpc = jmpc.Jmpc(port=1234)
pybulletClient = jmpc.Jmpc(port=1235)

STATE_DIM = 12
learning_iterations = settings.learning_iterations

mpc.resetTrajectory()
getTrajectoryResponse = mpc.getTrajectory()
mpcTrajectoryTimes = getTrajectoryResponse.get("result").get("times")
trajectoryLen = len(mpcTrajectoryTimes)
trajectoryLastTime = mpcTrajectoryTimes[trajectoryLen-1] # length of trajectories to generate with MPC

mpcTrajectoryStates = getTrajectoryResponse.get("result").get("trajectory")

'''
def mseVerification(policy):
    getTrajectoryResponse = mpc.getTrajectory()
    trajectoryLen = len(getTrajectoryResponse.get("result").get("times"))

    # prepare saving of MPC solution trajectory (always add first point of a slq run)
    trajectoryMaxTime = getTrajectoryResponse.get("result").get("times")[trajectoryLen-1] # length of trajectories to generate with MPC
    dt = round(getTrajectoryResponse.get("result").get("times")[1] - getTrajectoryResponse.get("result").get("times")[0], 2) # 0.03s control duration
    #trajectoryTime = np.linspace(0.0, trajectoryMaxTime, trajectoryLen)
    trajectoryTime = np.arange(0.0, trajectoryMaxTime, dt)

    initState = getTrajectoryResponse.get("result").get("trajectory")[0]
    initState.extend(np.zeros(int(STATE_DIM/2)))

    mpc.resetTrajectory()
    x0 = initState
    #x0[0] = np.random.uniform(-0.5, 0.5) # base x
    #x0[1] = np.random.uniform(-0.5, 0.5) # base y
    MSELoss = 0.0
    for mpc_time in trajectoryTime: # mpc dummy loop

        ttx_net = torch.tensor(np.concatenate((mpc_time, x0), axis=None), dtype=torch.float, device=torch.device("cpu"), requires_grad=False)
        p, u_pred = policy(ttx_net)
        
        if len(p) > 1:
            u_net = torch.matmul(p, u_pred)
        else:
            u_net = u_pred[0]

        u_np = u_net.detach().numpy().astype('float64')

        #print("start policyReqResp, ", np.transpose(tx[1:])[0].tolist(), " ", tx[0][0])
        computePolicyResponse = mpc.computePolicy(x0, mpc_time)
        if computePolicyResponse.get("result") == False :
           print("Compute Policy Error!")
           sys.exit(0)
        
        jsonControl = mpc.getControl(dt, x0, mpc_time)
        MSELoss += np.square(np.subtract(u_np.tolist(), jsonControl.get("result"))).sum() 
        #print("index ", index,"time ",mpc_time, " ,control net ", u_np.tolist(), "control mpc ", jsonControl.get("result"))
        #nextStateList = mpc.getNextState(jsonControl.get("result"), dt, x0) #for mpc get next state
        nextStateList = mpc.getNextState(u_np.tolist(), dt, x0) #for mpc-net get next state
        
        x0 = nextStateList.get("result")

    return MSELoss
'''

initState = mpcTrajectoryStates[0].copy()
initState.extend(np.zeros(int(STATE_DIM/2)))
dt = 7./240.

setInitStateResponse = pybulletClient.setInitState(1./240., initState, trajectoryLen, learning_iterations)
print("setInitStateResponse ", setInitStateResponse)
if setInitStateResponse == False:
   print("set Initial State Response Error!")
   sys.exit(0)

def plot(save_path, policy):
    policy = torch.load(save_path)
    pybullet_mpcnet_position_history = np.zeros((trajectoryLen, STATE_DIM + 1))
    pybullet_mpcnet_velocity_history = np.zeros((trajectoryLen, STATE_DIM + 1))
    mpc_control_history = np.zeros((trajectoryLen, int(STATE_DIM/2)+1))
    mpcnet_control_history = np.zeros((trajectoryLen, int(STATE_DIM/2)+1))
    tx_history = np.zeros((trajectoryLen, STATE_DIM + 1))
    trajectory_history = np.zeros((trajectoryLen, STATE_DIM + 1))

    
    ############## MPC Rollout #####################
    # Set pubullet robot arm initial position, control period and total control times
    MSE = 0
    currentStateList = initState.copy()
    trajectory_history[0, 1:] = np.array(mpcTrajectoryStates[0] + [0]*int(STATE_DIM/2))
    
    for timeIndex in range(trajectoryLen-1):
        currentTime = dt * timeIndex
        trajectory_history[timeIndex, 0] = currentTime
        trajectory_history[timeIndex, 1:] = currentStateList
        computePolicyResponse = mpc.computePolicy(currentStateList, currentTime)
        if computePolicyResponse == False :
           print("Compute Policy Error!")
           sys.exit(0)
        
        jsonControl = mpc.getControl(dt, currentStateList, currentTime)
        mpc_control_history[timeIndex, 0] = currentTime
        mpc_control_history[timeIndex, 1:] = jsonControl

        #nextStateList = mpc.getNextState(jsonControl, dt, currentStateList) #for mpc-net get next state
        nextStateList = pybulletClient.getNextState(jsonControl, dt, currentStateList) #for pyBullet get next state
        currentStateList = nextStateList
    #print("MPC Final State Diff ", np.subtract(currentStateList[:int(STATE_DIM/2)], mpcTrajectoryStates[-1]))
    
    time.sleep(1)   
    ############## MPC-Net Rollout #####################
    MSE = 0.0
    currentStateList = initState.copy()
    mpc.resetTrajectory()
    #only have to calculate trajectoryLen-1 control

    for timeIndex in range(trajectoryLen-1):
        currentTime = dt*timeIndex
        tx_history[timeIndex, 0] = currentTime
        tx_history[timeIndex, 1:] = currentStateList
        tx_torch = torch.tensor(np.concatenate((currentTime, currentStateList.copy()), axis=None), dtype=torch.float, requires_grad=False)
        #tx_torch[0][0] = 0.0 #optionally run it in MPC style

        p, u_pred = policy(tx_torch)
        if len(p) > 1:
            u = torch.matmul(p, u_pred)
        else:
            u = u_pred[0]

        u_np = u.detach().numpy().astype('float64')
        mpcnet_control_history[timeIndex, 0] = currentTime
        mpcnet_control_history[timeIndex, 1:] = u_np.tolist()
        
        computePolicyResponse = mpc.computePolicy(currentStateList.copy(), currentTime)
        if computePolicyResponse == False :
           print("Compute Policy Error!")
           sys.exit(0)

        jsonControl = mpc.getControl(dt, currentStateList.copy(), currentTime)
        MSE += np.square(np.subtract(currentStateList[:6], mpcTrajectoryStates[timeIndex])).sum()

        #print("MPC-Net index ", timeIndex,", Time ",currentTime," ,MSE loss",MSE, "\n")
        pybullet_mpcnet_position_history[timeIndex, 0] = currentTime
        pybullet_mpcnet_velocity_history[timeIndex, 0] = currentTime
        if settings.currentRendering == "enableRendering":
           nextStateList = pybulletClient.getNextState(u_np.tolist(), dt, currentStateList.copy()) #for pyBullet get next state
           pybullet_mpcnet_position_history[timeIndex, int(STATE_DIM/2)+1:] = nextStateList[:int(STATE_DIM/2)]
           pybullet_mpcnet_velocity_history[timeIndex, int(STATE_DIM/2)+1:] = nextStateList[int(STATE_DIM/2):]
        elif settings.currentRendering == "enablePureRendering":
           nextStateList = mpc.getNextState(u_np.tolist(), dt, currentStateList.copy()) #for mpc-net get next state
           pybullet_mpcnet_position_history[timeIndex, 1:int(STATE_DIM/2)+1] = nextStateList[:int(STATE_DIM/2)]
           pybullet_mpcnet_velocity_history[timeIndex, 1:int(STATE_DIM/2)+1] = nextStateList[int(STATE_DIM/2):]

           pybulletClient.setState(currentStateList.copy(), timeIndex)
        elif settings.currentRendering == "enableResetStateRendering":
           pybulletClient.setState(currentStateList.copy(), timeIndex)
           nextStateList = pybulletClient.getNextState(u_np.tolist(), dt, currentStateList.copy()) #for pyBullet get next state
           #print("pybullet nextStateList ", nextStateList[5])
           pybullet_mpcnet_position_history[timeIndex, int(STATE_DIM/2)+1:] = nextStateList[:int(STATE_DIM/2)]
           pybullet_mpcnet_velocity_history[timeIndex, int(STATE_DIM/2)+1:] = nextStateList[int(STATE_DIM/2):]

           nextStateList = mpc.getNextState(u_np.tolist(), dt, currentStateList.copy())
           print("index ", timeIndex, " ,mpc nextStateList ", currentStateList)
           pybullet_mpcnet_position_history[timeIndex, 1:int(STATE_DIM/2)+1] = nextStateList[:int(STATE_DIM/2)]
           pybullet_mpcnet_velocity_history[timeIndex, 1:int(STATE_DIM/2)+1] = nextStateList[int(STATE_DIM/2):]

        currentStateList = nextStateList.copy()

    print("MPC-Net Final State Diff ", np.subtract(currentStateList[:int(STATE_DIM/2)], mpcTrajectoryStates[-1]), 'MSE Loss ', MSE)
    
    f, axarr = plt.subplots(4,2)
    
    lineObjects = axarr[0][0].plot(trajectory_history[:trajectoryLen-1, 0], trajectory_history[:trajectoryLen-1, 1:int(STATE_DIM/2)+1])   #plot velocity
    axarr[0][0].legend(iter(lineObjects), ('q1', 'q2','q3','q4','q5','q6'))
    axarr[0][0].set_ylim(-2, 2)
    axarr[0][0].grid(True)
    axarr[0][0].set_title("MPC State")

    lineObjects = axarr[0][1].plot(tx_history[:trajectoryLen-1, 0], tx_history[:trajectoryLen-1, 1:int(STATE_DIM/2)+1])   #plot velocity
    axarr[0][1].set_ylim(-2, 2)
    axarr[0][1].grid(True)
    axarr[0][1].set_title("MPC-Net State")


    lineObjects = axarr[1][0].plot(trajectory_history[:trajectoryLen-1, 0], trajectory_history[:trajectoryLen-1, int(STATE_DIM/2)+1:])   #plot velocity
    axarr[1][0].legend(iter(lineObjects), ('q7', 'q8','q9','q10','q11','q12'))
    axarr[1][0].set_ylim(-2, 2)
    axarr[1][0].grid(True)

    lineObjects = axarr[1][1].plot(tx_history[:trajectoryLen-1, 0], tx_history[:trajectoryLen-1, int(STATE_DIM/2)+1:])   #plot velocity
    axarr[1][1].set_ylim(-2, 2)
    axarr[1][1].grid(True)

    lineObjects = axarr[2][0].plot(trajectory_history[:trajectoryLen-1, 0], trajectory_history[:trajectoryLen-1, int(STATE_DIM/2)+1:-1])   #plot velocity
    axarr[2][0].legend(iter(lineObjects), ('q7', 'q8','q9','q10','q11','q12'))
    axarr[2][0].set_ylim(-2, 2)
    axarr[2][0].grid(True)

    lineObjects = axarr[2][1].plot(tx_history[:trajectoryLen-1, 0], tx_history[:trajectoryLen-1, int(STATE_DIM/2)+1:-1])   #plot velocity
    axarr[2][1].set_ylim(-2, 2)
    axarr[2][1].grid(True)

    lineObjects = axarr[3][0].plot(mpc_control_history[:trajectoryLen-1, 0], mpc_control_history[:trajectoryLen-1, 1:int(STATE_DIM/2)+1])   #plot velocity
    axarr[3][0].legend(iter(lineObjects), ('c1', 'c2','c3','c4','c5','c6'))
    axarr[3][0].set_ylim(-50, 50)
    axarr[3][0].grid(True)

    lineObjects = axarr[3][1].plot(mpcnet_control_history[:trajectoryLen-1, 0], mpcnet_control_history[:trajectoryLen-1, 1:int(STATE_DIM/2)+1])   #plot velocity
    axarr[3][1].set_ylim(-50, 50)
    axarr[3][1].grid(True)


if settings.currentRendering == "enablePureRendering":
   shutil.copy("PolicyNet_2Layer.py","PolicyNet.py")
   #the only two layer policy, we train it from mpc dynamic environment. Hard code path now.
   policy = torch.load("armPolicy/pyBullet/1014/mpcPolicy_2020-10-28_025339.pt")
   plot(save_path="armPolicy/pyBullet/1014/mpcPolicy_2020-10-28_025339.pt", policy=policy)
else:
   shutil.copy("PolicyNet_3Layer.py","PolicyNet.py")
   #Formally we use three layer policy now, we train it from pybullet dynamic environment. Hard code path now.
   policy = torch.load(settings.loadPolicyPath)
   plot(save_path=settings.loadPolicyPath, policy=policy)
#plot(save_path="armPolicy/pyBullet/1115/161926/233420/004124/mpcPolicy_2020-11-16_015155.pt", t_end=trajectoryLastTime)
#plot(save_path="armPolicy/pyBullet/1115/161926/233420/004124/mpcPolicy_2020-11-16_011155.pt", t_end=trajectoryLastTime)
#plot(save_path="armPolicy/pyBullet/1115/161926/233420/004124/mpcPolicy_2020-11-16_010655.pt", t_end=trajectoryLastTime)
#plot(save_path="armPolicy/pyBullet/1115/161926/233420/mpcPolicy_2020-11-16_004124.pt", t_end=trajectoryLastTime)
#plot(save_path="armPolicy/pyBullet/1105/094457/112213/mpcPolicy_2020-11-06_143418.pt", t_end=trajectoryLastTime)
#plot(save_path="armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/175636/020101/mpcPolicy_2020-10-28_025049.pt", t_end=trajectoryLastTime)
#plot(save_path="armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/mpcPolicy_2020-10-20_175406.pt", t_end=trajectoryMaxTime) #very good
#plot(save_path="armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/mpcPolicy_2020-10-20_175906.pt", t_end=trajectoryMaxTime)
#plot(save_path="armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/mpcPolicy_2020-10-20_174636.pt", t_end=trajectoryMaxTime)
#plot(save_path="armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/mpcPolicy_2020-10-20_173406.pt", t_end=trajectoryMaxTime)
#plot(save_path="armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/mpcPolicy_2020-10-20_172136.pt", t_end=trajectoryMaxTime)
#plot(save_path="armPolicy/alphaMix_1014/single_state/mpcPolicy_2020-10-20_144840.pt", t_end=trajectoryMaxTime)
#plot(save_path="armPolicy/next_sate_without_alpha_mixing/1016/mpcPolicy_2020-10-19_192314.pt", t_end=trajectoryMaxTime)
#plot(save_path="armPolicy/alphaMix_1014/1020/mpcPolicy_2020-10-20_023059.pt", t_end=trajectoryMaxTime)
#plot(save_path="armPolicy/next_sate_without_alpha_mixing/mpcPolicy_2020-10-12_201140.pt", t_end=trajectoryMaxTime)
#plot(save_path="armPolicy/mpcPolicy_2020-10-07_020028.pt", t_end=trajectoryMaxTime)
#plot(save_path="armPolicy/mpcPolicy_2020-10-07_020028.pt", t_end=trajectoryMaxTime)

plt.show()