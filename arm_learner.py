import numpy as np
import torch
from tensorboardX import SummaryWriter
import datetime
import time
import pickle
from replay_memory import ReplayMemory
import jmpc
import settings
import shutil

from PolicyNet import ExpertMixturePolicy as PolicyNet

#workaround to support two layer and three layer policy mpc-net rendering
shutil.copy("PolicyNet_3Layer.py","PolicyNet.py")

STATE_DIM = 12
INPUT_DIM = 6
mpc = jmpc.Jmpc(port=1234)
pybulletClient = jmpc.Jmpc(port=1235)

jointTorqueConstraint = np.array([104, 104, 69, 69, 34, 34])
jointRadianLimit = np.array([3.14, 2.35, 2.61, 3.14, 2.56, 3.14])
jointVelocityLimit = np.array([1.57, 1.57, 1.57, 1.57, 1.57, 1.57])
learning_rate = 1e-3
batch_size = 2**5
learning_iterations = settings.learning_iterations
dt_control = 7./240.

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU

def mseLoss_function(u_pred, u0):
    mseloss = torch.nn.MSELoss(reduction='sum')
    return mseloss(u_pred, u0)

writer = SummaryWriter()

if settings.loadPolicy:
    policy = torch.load(settings.loadPolicyPath)
    policy.eval()
else:
    policy = PolicyNet(STATE_DIM+1, INPUT_DIM)

policy.to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

if settings.loadMemory:
    settings.enableSampling = False
    with open(settings.loadMemoryPath, 'rb') as memFile:
        mem = pickle.load(memFile)
else:
    mem_capacity = 100000
    mem = ReplayMemory(mem_capacity)

getTrajectoryResponse = mpc.getTrajectory()
trajectoryTimes = getTrajectoryResponse.get("result").get("times")
trajectoryLen = len(trajectoryTimes)

# prepare saving of MPC solution trajectory (always add first point of a slq run)
trajectoryLastTime = trajectoryTimes[trajectoryLen-1] # length of trajectories to generate with MPC
#mpc_traj_t = np.linspace(0.0, trajectoryLastTime, trajectoryLen)
lastPolicySaveTime = time.time()


trajectoryStates = getTrajectoryResponse.get("result").get("trajectory")
initState = trajectoryStates[0].copy()
initState.extend(np.zeros(int(STATE_DIM/2)))

if settings.enablePybulletTraining == True:
   setInitStateResponse = pybulletClient.setInitState(1./240., initState, trajectoryLen, learning_iterations)
   if setInitStateResponse == False:
      print("set Initial State Response Error!")
      sys.exit(0)

def sampling(it):
    alpha_mix = np.clip(1.0 - 1.0 * it / learning_iterations, 0.2, 1.0)

    # run data collection (=MPC) less frequently than the policy updates
    mpc_decimation = 1 if len(mem) < 15000 else 500
    x0 = []
    if it % mpc_decimation == 0:

        mpc.resetTrajectory()
        mpc.getTrajectory()
        
        if settings.enablePybulletTraining == True:
           x0 = pybulletClient.setInitTrainingState(it)
           print("x0 ", x0)
        else:
           x0 = initState.copy()

        print("learning_iterations ", it, ", proportion of MPC policy is", alpha_mix)

        for timeIndex in range(trajectoryLen-1): # mpc dummy loop
            
            currentTime = dt_control*timeIndex
            computePolicyResponse = mpc.computePolicy(x0, currentTime)

            if computePolicyResponse == False :
               print("Compute Policy Error!")
               break

            u_result = mpc.getControl(dt_control, x0, currentTime)
            
            trajectoryNextVelocity = (np.subtract(trajectoryStates[timeIndex+1], trajectoryStates[timeIndex])/dt_control).tolist()
            trajectoryNextState = trajectoryStates[timeIndex+1] + trajectoryNextVelocity
            mem.push(currentTime, x0, None, None, None, None, u_result, trajectoryNextState)

            # increment state for next time step
            ttx_torch = torch.tensor(np.concatenate((currentTime, x0), axis=None), dtype=torch.float, requires_grad=False)
            #ttx_torch = torch.tensor(np.concatenate((currentTime, trajectoryNextState, x0), axis=None), dtype=torch.float, requires_grad=False)
            p, u_net = policy(ttx_torch)
            if len(p) > 1:
                u_net = torch.matmul(p, u_net)
            else:
                u_net = u_net[0]

            uNetControl = u_net.detach().numpy().astype('float64')
            u_mixed = alpha_mix * np.array(u_result) + (1.0 - alpha_mix) * uNetControl

            #currently turning off Torque Limit whiling training because it will interrupt almost all rollout
            """
            if np.any(np.greater(np.abs(u_mixed), jointTorqueConstraint)) == True:
               print("############ Alpha Mixing Torque over Limit ", u_mixed,  " at time ", currentTime, " with state ", x0)
               if settings.enablePybulletTraining == True:
                  pybulletClient.getNextState(u_mixed.tolist(), dt_control, x0, True)
               break
            """
            if settings.enablePybulletTraining == True:
               x0 = pybulletClient.getNextState(u_mixed.tolist(), dt_control, x0) #for pyBullet get next state
            else:
               x0 = mpc.getNextState(u_mixed.tolist(), dt_control, x0)

            #check next state can't be over limit
            if np.any(np.greater(np.abs(np.array(x0[:int(STATE_DIM/2)])), jointRadianLimit)) == True:
               print("############### mpc calculate nextState over Limit ",x0,  " at time ", currentTime)
               if settings.enablePybulletTraining == True:
                  pybulletClient.getNextState(u_mixed.tolist(), dt_control, x0, True)
               break

            if np.any(np.greater(np.abs(x0[int(STATE_DIM/2):]), jointVelocityLimit)) == True:
               print("############### mpc calculate velocity over limit ",x0,  ", at time ", currentTime)
               if settings.enablePybulletTraining == True:
                  pybulletClient.getNextState(u_mixed.tolist(), dt_control, x0, True)
               break
        print("mpc ended up at ", x0)
        
print("==============\nStarting training\n==============")
try:
    for it in range(learning_iterations):
        if settings.enableSampling == True:
           sampling(it)

        samples = mem.sample(batch_size)
        """
        samples = []
        while(len(samples)<batch_size):
            temps = mem.sample(batch_size)
            for sample in temps:
                if sample.t >= 2.8:
                   samples.append(sample)
                if len(samples)==batch_size:
                   break
        """
        writeLogThisIteration = True

        def solver_step_closure():
            loss = torch.zeros([1], dtype=dtype, device=device)  # running sum over samples
            #MSE = 0.0

            for sample in samples:
                ttx_net = torch.tensor(np.concatenate((sample.t, sample.x), axis=None), dtype=dtype, device=device, requires_grad=False)
                p, u_pred = policy(ttx_net)
                
                if len(p) > 1:
                    u_net = torch.matmul(p, u_pred)
                else:
                    u_net = u_pred[0]
                loss += mseLoss_function(u_net, torch.FloatTensor(sample.u0).to(device))
                #MSE += np.square(np.subtract(u_net.detach().numpy().astype('float64'), np.array(sample.u0))).sum()
            
            print("iter ", it, " ,loss ", loss, "mem ", len(mem))
            
            optimizer.zero_grad()
            loss.backward()

            global writeLogThisIteration
            if writeLogThisIteration:
                writer.add_scalar('loss', loss.item(), it)
                writer.add_scalar('mem', len(mem), it)
                writeLogThisIteration = False
            return loss

        if time.time() - lastPolicySaveTime > 300 * 5:
            lastPolicySaveTime = time.time()
            now = datetime.datetime.now()
            save_path = settings.save_path + now.strftime("%Y-%m-%d_%H%M%S")
            print("Iteration ", it, " saving policy to", save_path + ".pt")
            torch.save(policy, save_path + ".pt")

        optimizer.step(solver_step_closure)
        for param in policy.parameters():
            if(torch.isnan(param).any()):
                print("nan in policy!")


    print("==============\nTraining completed.\n==============")
except KeyboardInterrupt:
    print("==============\nTraining interrupted after iteration", it, ".\n==============")
    pass


now = datetime.datetime.now()
save_path = settings.save_path + now.strftime("%Y-%m-%d_%H%M%S")
print("saving policy to", save_path + ".pt")

print("Saving data to", save_path+"_memory.pkl")
with open(save_path+"_memory.pkl", 'wb') as outputFile:
     pickle.dump(mem, outputFile)


writer.close()

print("Done. Exiting now.")