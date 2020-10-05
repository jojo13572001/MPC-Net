import numpy as np
import torch
from tensorboardX import SummaryWriter
import datetime
import time
import pickle
from replay_memory import ReplayMemory
# ugly workaround until shared library can be discovered properly with python3
import jmpc

from PolicyNet import ExpertMixturePolicy as PolicyNet


STATE_DIM = 12
INPUT_DIM = 6
mpc = jmpc.Jmpc(port=5678)
systemHasConstraints = False

mpc.resetTrajectory()


dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU


def mseLoss_function(u0, u_pred):
    mseloss = torch.nn.MSELoss(reduction='mean')
    return mseloss(u0, u_pred)

def loss_function(tx, u_pred, dVdx, nu):
    f = FlowMap.apply(tx[0], tx[1:], u_pred)
    L = IntermediateCost.apply(tx[0], tx[1:], u_pred)
    loss = L + dVdx.dot(f)
    if systemHasConstraints:
        g1 = StateInputConstraint.apply(tx[0], tx[1:], u_pred)
        loss += g1.dot(nu)
    return loss


writer = SummaryWriter()

load_policy = False
if load_policy:
    save_path = "tmp2/mpcPolicy_2020-07-18_122836.pt"
    policy = torch.load(save_path)
    policy.eval()
else:
    policy = PolicyNet(STATE_DIM+1, INPUT_DIM)

policy.to(device)

#print("Initial policy parameters:")
#print(list(policy.named_parameters()))

learning_rate = 1e-3
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)


load_memory = False
if load_memory:
    with open("armPolicy/memory.pkl", 'rb') as memFile:
        mem = pickle.load(memFile)
else:
    mem_capacity = 1000000
    mem = ReplayMemory(mem_capacity)


getTrajectoryResponse = mpc.getTrajectory()
trajectoryLen = len(getTrajectoryResponse.get("result").get("times"))

# prepare saving of MPC solution trajectory (always add first point of a slq run)
mpc_traj_len_sec = getTrajectoryResponse.get("result").get("times")[trajectoryLen-1] # length of trajectories to generate with MPC
dt_control = round(getTrajectoryResponse.get("result").get("times")[1] - getTrajectoryResponse.get("result").get("times")[0], 2) # 0.03s control duration
mpc_traj_t = np.linspace(0.0, mpc_traj_len_sec, trajectoryLen)
last_policy_save_time = time.time()

learning_iterations = 100000

initState = getTrajectoryResponse.get("result").get("trajectory")[0]
initState.extend(np.zeros(int(STATE_DIM/2)))

print("==============\nStarting training\n==============")
try:
    for it in range(learning_iterations):
        alpha_mix = np.clip(1.0 - 1.0 * it / learning_iterations, 0.2, 1.0)

        # run data collection (=MPC) less frequently than the policy updates
        mpc_decimation = 1 if len(mem) < 15000 else 500
        if it % mpc_decimation == 0:

            mpc.resetTrajectory()
            x0 = initState
            #x0[0] = np.random.uniform(-0.5, 0.5) # base x
            #x0[1] = np.random.uniform(-0.5, 0.5) # base y

            print("learning_iterations ", it, ", proportion of MPC policy is", alpha_mix)
            #print("starting from", x0)

            for mpc_time in mpc_traj_t: # mpc dummy loop

                computePolicyResponse = mpc.computePolicy(x0, mpc_time)

                if computePolicyResponse.get("result") == False :
                   print("Compute Policy Error!")
                   break

                jsonControl = mpc.getControl(dt_control, x0, mpc_time)
                u_result = jsonControl.get("result")
                jsonStateFunctionValueDerivative = mpc.getValueFunctionStateDerivative(x0, mpc_time+1e-4)

                mem.push(mpc_time, x0, None, None, None, None, u_result)

                # increment state for next time step
                ttx_torch = torch.tensor(np.concatenate((mpc_time, x0), axis=None),
                                         dtype=torch.float, requires_grad=False)
                p, u_net = policy(ttx_torch)
                if len(p) > 1:
                    u_net = torch.matmul(p, u_net)
                else:
                    u_net = u_net[0]

                u_mixed = alpha_mix * np.array(u_result) + (1.0 - alpha_mix) * u_net.detach().numpy().astype('float64')
                jsonNextState = mpc.getNextState(u_mixed.tolist(), dt_control, x0)
                x0 = jsonNextState.get("result")

            print("mpc ended up at ", x0)

        # extract batch of samples from replay memory
        batch_size = 2**5
        samples = mem.sample(batch_size)

        writeLogThisIteration = True

        def solver_step_closure():
            loss = torch.zeros([1], dtype=dtype, device=device)  # running sum over samples
            sampleU = torch.zeros([batch_size, INPUT_DIM], dtype=dtype, device=device)
            predictU = torch.zeros([batch_size, INPUT_DIM], dtype=dtype, device=device)

            index = 0
            for sample in samples:
                sum_u = 0.0
                tx = torch.tensor(np.concatenate((sample.t, sample.x), axis=None), dtype=dtype, device=device, requires_grad=False)
                ttx_net = torch.tensor(np.concatenate((sample.t, sample.x), axis=None), dtype=dtype, device=device, requires_grad=False)
                p, u_pred = policy(ttx_net)
                
                if len(p) > 1:
                    u_net = torch.matmul(p, u_pred)
                else:
                    u_net = u_pred[0]

                predictU[index] = u_net
                sampleU[index] = torch.FloatTensor(sample.u0).to(device)
                index = index + 1

            loss = mseLoss_function(sampleU, predictU)
            optimizer.zero_grad()
            loss.backward()

            global writeLogThisIteration
            if writeLogThisIteration:
                writer.add_scalar('loss/perSample', loss.item(), it)
                print('loss/perSample', loss.item())
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
        if time.time() - last_policy_save_time > 5.0 * 60.0:
            last_policy_save_time = time.time()
            now = datetime.datetime.now()
            save_path = "armPolicy/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
            print("Iteration", it, "saving policy to", save_path + ".pt")
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
save_path = "armPolicy/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
print("saving policy to", save_path + ".pt")
torch.save(policy, save_path + ".pt")

print("Saving data to", save_path+"_memory.pkl")
with open(save_path+"_memory.pkl", 'wb') as outputFile:
     pickle.dump(mem, outputFile)


writer.close()

print("Done. Exiting now.")