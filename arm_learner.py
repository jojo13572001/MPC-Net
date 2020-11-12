import numpy as np
import torch
from tensorboardX import SummaryWriter
import datetime
import time
import pickle
from replay_memory import ReplayMemory
# ugly workaround until shared library can be discovered properly with python3
import jmpc
import settings

from PolicyNet import ExpertMixturePolicy as PolicyNet


STATE_DIM = 12
INPUT_DIM = 6
mpc = jmpc.Jmpc(port=1234)
pybulletClient = jmpc.Jmpc(port=1235)

systemHasConstraints = False
jointTorqueConstraint = np.array([104, 104, 69, 69, 34, 34])
jointRadianLimit = np.array([3.14, 2.35, 2.61, 3.14, 2.56, 3.14])
jointVelocityLimit = np.array([1.57, 1.57, 1.57, 1.57, 1.57, 1.57])
learning_rate = 1e-3
learning_iterations = 100000
dt_control = 7./240.

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU
"""
def mseVerification(policy):
    mpc.resetTrajectory()
    getTrajectoryResponse = mpc.getTrajectory()
    trajectoryLen = len(getTrajectoryResponse.get("result").get("times"))

    # prepare saving of MPC solution trajectory (always add first point of a slq run)
    trajectoryLastTime = getTrajectoryResponse.get("result").get("times")[trajectoryLen-1] # length of trajectories to generate with MPC
    dt = round(getTrajectoryResponse.get("result").get("times")[1] - getTrajectoryResponse.get("result").get("times")[0], 2) # 0.03s control duration
    #trajectoryTime = np.linspace(0.0, trajectoryLastTime, trajectoryLen)
    trajectoryTime = np.arange(0.0, trajectoryLastTime, dt)
    last_policy_save_time = time.time()

    initState = getTrajectoryResponse.get("result").get("trajectory")[0]
    initState.extend(np.zeros(int(STATE_DIM/2)))

    x0 = initState
    #x0[0] = np.random.uniform(-0.5, 0.5) # base x
    #x0[1] = np.random.uniform(-0.5, 0.5) # base y
    MSELoss = 0.0
    for timeIndex in trajectoryTime: # mpc dummy loop

        ttx_net = torch.tensor(np.concatenate((timeIndex, x0), axis=None), dtype=dtype, device=device, requires_grad=False)
        p, u_pred = policy(ttx_net)
        
        if len(p) > 1:
            u_net = torch.matmul(p, u_pred)
        else:
            u_net = u_pred[0]

        u_np = u_net.detach().numpy().astype('float64')

        #print("start policyReqResp, ", np.transpose(tx[1:])[0].tolist(), " ", tx[0][0])
        computePolicyResponse = mpc.computePolicy(x0, timeIndex)
        if computePolicyResponse == False :
           print("Compute Policy Error!")
           sys.exit(0)
        
        jsonControl = mpc.getControl(dt, x0, timeIndex)
        MSELoss += np.square(np.subtract(u_np.tolist(), jsonControl)).sum() 
        #print("index ", index,"time ",timeIndex, " ,control net ", u_np.tolist(), "control mpc ", jsonControl.get("result"))
        jsonNextState = mpc.getNextState(jsonControl, dt, x0) #for mpc get next state
        #jsonNextState = mpc.getNextState(u_np.tolist(), dt, x0) #for mpc-net get next state
        
        x0 = jsonNextState

    return MSELoss
"""
def mseLoss_function(u_pred, u0):
    mseloss = torch.nn.MSELoss(reduction='sum')
    return mseloss(u_pred, u0)

def hamiltonian_loss_function(tx, u_pred, dVdx, nu):
    f = FlowMap.apply(tx[0], tx[1:], u_pred)
    L = IntermediateCost.apply(tx[0], tx[1:], u_pred)
    loss = L + dVdx.dot(f)
    if systemHasConstraints:
        g1 = StateInputConstraint.apply(tx[0], tx[1:], u_pred)
        loss += g1.dot(nu)
    return loss


writer = SummaryWriter()

if settings.loadPolicy:
    #save_path = "armPolicy/alphaMix_1014/single_state_2_layers/mpcPolicy_2020-10-20_164518.pt"
    #save_path = "armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/mpcPolicy_2020-10-20_175406.pt"
    #save_path = "armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/175636/mpcPolicy_2020-10-28_020101.pt"
    #save_path = "armPolicy/pyBullet/1105/mpcPolicy_2020-11-06_094457.pt"
    save_path = "armPolicy/pyBullet/1105/094457/mpcPolicy_2020-11-06_112213.pt"
    policy = torch.load(save_path)
    policy.eval()
else:
    policy = PolicyNet(STATE_DIM+1, INPUT_DIM)

policy.to(device)

#print("Initial policy parameters:")
#print(list(policy.named_parameters()))


optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)


if settings.loadMemory:
    settings.enableSampling = False
    with open("armPolicy/pyBullet/1105/094457/mpcPolicy_2020-11-06_112443_memory.pkl", 'rb') as memFile:
        mem = pickle.load(memFile)
else:
    mem_capacity = 100000
    mem = ReplayMemory(mem_capacity)

"""
print("start sampling ")
while True:
    temps = mem.sample(32)
    for sample in temps:
        if sample.t < 0.05 and sample.t > 0.0:
           print("sample.t ",sample.t, " ,sample.x ", sample.x)
           print("sample.u0 ", sample.u0, "\n")
           break
print("finish sampling ")
"""
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
   setInitStateResponse = pybulletClient.setInitState(1./240., initState, trajectoryLen)
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
           x0 = pybulletClient.setInitTrainingState(it, learning_iterations)
           print("x0 ", x0)
        else:
           x0 = initState.copy()
        #x0[0] = np.random.uniform(-0.5, 0.5) # base x
        #x0[1] = np.random.uniform(-0.5, 0.5) # base y

        print("learning_iterations ", it, ", proportion of MPC policy is", alpha_mix)
        #print("starting from", x0)

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
            #jsonStateFunctionValueDerivative = mpc.getValueFunctionStateDerivative(x0, timeIndex+1e-4)

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

            #print("############ MPC suggenst control ", u_result,  " at time ", currentTime, " with index ", timeIndex)
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

        # extract batch of samples from replay memory
        batch_size = 2**5

        
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
                #ttx_net = torch.tensor(np.concatenate((sample.nx, sample.x), axis=None), dtype=dtype, device=device, requires_grad=False)
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
                #writer.add_scalar('mseVerification', mseVerify, it)
                #print('loss/allSamples', loss.item(), " ,MSE ", MSE)
                #print("sampleU ", sampleU[batch_size-1])
                #print("predictU ", predictU[batch_size-1])
                #writer.add_scalar('loss/perSample', loss.item() / batch_size, it)
                #writer.add_scalar('loss/constraintViolation', g1_norm / batch_size, it)
                writeLogThisIteration = False
            return loss

        """
        if it % 200 == 0:
            oc_cost, survival_time = trajectoryCost(policy=policy, duration=mpc_traj_len_sec, dt_control=dt_control)
            writer.add_scalar('metric/oc_cost', oc_cost, it)
            writer.add_scalar('metric/survival_time', survival_time, it)
            print("iteration", it, "oc_cost", oc_cost)
        """
        #finalDiff = np.abs(np.subtract(currentStateList[:int(STATE_DIM/2)], mpcTrajectoryStates[-1]))
        #print("Final Diff ", np.subtract(currentStateList[:int(STATE_DIM/2)], mpcTrajectoryStates[-1]))
        #if time.time() - lastPolicySaveTime > 5.0:
        if time.time() - lastPolicySaveTime > 5.0 * 30.0:
            lastPolicySaveTime = time.time()
            now = datetime.datetime.now()
            #save_path = "armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/175636/020101/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
            #save_path = "armPolicy/pyBullet/1105/094457/112213/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
            save_path = "armPolicy/pyBullet/1113/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
            print("Iteration ", it, " saving policy to", save_path + ".pt")
            torch.save(policy, save_path + ".pt")

        #print("Iteration", it, "mseVerification ", mseVerification(policy))
        optimizer.step(solver_step_closure)
        for param in policy.parameters():
            if(torch.isnan(param).any()):
                print("nan in policy!")


    print("==============\nTraining completed.\n==============")
except KeyboardInterrupt:
    print("==============\nTraining interrupted after iteration", it, ".\n==============")
    pass


now = datetime.datetime.now()
save_path = "armPolicy/pyBullet/1113/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
#save_path = "armPolicy/pyBullet/1105/094457/112213/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
#save_path = "armPolicy/alphaMix_1014/single_state_2_layers/keepTrainingWithoutSampling/175636/020101/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
print("saving policy to", save_path + ".pt")
#torch.save(policy, save_path + ".pt")

print("Saving data to", save_path+"_memory.pkl")
with open(save_path+"_memory.pkl", 'wb') as outputFile:
     pickle.dump(mem, outputFile)


writer.close()

print("Done. Exiting now.")