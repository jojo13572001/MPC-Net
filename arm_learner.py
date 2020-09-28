import numpy as np
import torch
from tensorboardX import SummaryWriter
import datetime
import time
import pickle
from replay_memory import ReplayMemory
import os
import json

# ugly workaround until shared library can be discovered properly with python3
#import sys
#sys.path.append(os.environ["HOME"]+"/catkin_ws/devel/lib/python3.6/dist-packages/ocs2_ballbot_example")

#from BallbotPyBindings import mpc_interface, scalar_array, state_vector_array, input_vector_array, dynamic_vector_array, cost_desired_trajectories

from PolicyNet import ExpertMixturePolicy as PolicyNet

#mpc = mpc_interface("mpc", False)
systemHasConstraints = False
STATE_DIM = 14
INPUT_DIM = 7
"""
def getTargetTrajectories():
    desiredTimeTraj = scalar_array()
    desiredTimeTraj.resize(1)
    desiredTimeTraj[0] = 2.0

    desiredInputTraj = dynamic_vector_array()
    desiredInputTraj.resize(1)
    desiredInputTraj[0] = np.zeros((mpc.INPUT_DIM, 1))

    desiredStateTraj = dynamic_vector_array()
    desiredStateTraj.resize(1)
    desiredStateTraj[0] = np.zeros((mpc.STATE_DIM, 1))

    return cost_desired_trajectories(desiredTimeTraj, desiredStateTraj, desiredInputTraj)


targetTrajectories = getTargetTrajectories()

mpc.reset(targetTrajectories)
"""


dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU


"""
class FlowMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, x, u):
        #In the forward pass we receive a Tensor containing the input and return
        #a Tensor containing the output. ctx is a context object that can be used
        #to stash information for backward computation. You can cache arbitrary
        #objects for use in the backward pass using the ctx.save_for_backward method.
        x_cpu = x.cpu()
        u_cpu = u.cpu()
        ctx.save_for_backward(t, x_cpu, u_cpu)
        x_np = x_cpu.t().detach().numpy().astype('float64')
        u_np = u_cpu.t().detach().numpy().astype('float64')
        xDot = torch.tensor(mpc.computeFlowMap(t, x_np, u_np), device=device, dtype=dtype)
        return xDot

    @staticmethod
    def backward(ctx, grad_output):

        #In the backward pass we receive a Tensor containing the gradient of the loss
        #with respect to the output, and we need to compute the gradient of the loss
        #with respect to the input.
 
        grad_t = grad_x = grad_u = None
        t, x, u = ctx.saved_tensors
        x_np = x.t().detach().numpy().astype('float64')
        u_np = u.t().detach().numpy().astype('float64')

        if ctx.needs_input_grad[0]:
            raise NotImplementedError("Derivative of dynamics w.r.t. time not available")
        if ctx.needs_input_grad[1]:
            mpc.setFlowMapDerivativeStateAndControl(t, x_np, u_np)
            dfdx = torch.tensor(mpc.computeFlowMapDerivativeState(), device=device, dtype=dtype)
            grad_x = torch.matmul(grad_output, dfdx).reshape((-1, x_np.size))
        if ctx.needs_input_grad[2]:
            mpc.setFlowMapDerivativeStateAndControl(t, x_np, u_np)
            dfdu = torch.tensor(mpc.computeFlowMapDerivativeInput(), device=device, dtype=dtype)
            grad_u = torch.matmul(grad_output, dfdu).reshape((-1, u_np.size))
        return grad_t, grad_x, grad_u


class IntermediateCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, x, u):
        #In the forward pass we receive a Tensor containing the input and return
        #a Tensor containing the output. ctx is a context object that can be used
        #to stash information for backward computation. You can cache arbitrary
        #objects for use in the backward pass using the ctx.save_for_backward method.
        x_cpu = x.cpu()
        u_cpu = u.cpu()
        ctx.save_for_backward(t, x_cpu, u_cpu)
        x_np = x_cpu.t().detach().numpy().astype('float64')
        u_np = u_cpu.t().detach().numpy().astype('float64')
        L = torch.tensor(mpc.getIntermediateCost(t, x_np, u_np), device=device, dtype=dtype)
        return L

    @staticmethod
    def backward(ctx, grad_output):
        #In the backward pass we receive a Tensor containing the gradient of the loss
        #with respect to the output, and we need to compute the gradient of the loss
        #with respect to the input.
        grad_t = grad_x = grad_u = None
        t, x, u = ctx.saved_tensors
        x_np = x.t().detach().numpy().astype('float64')
        u_np = u.t().detach().numpy().astype('float64')

        if ctx.needs_input_grad[0]:
            raise NotImplementedError("Derivative of RunningCost w.r.t. time not available")
        if ctx.needs_input_grad[1]:
            dLdx = torch.tensor([[mpc.getIntermediateCostDerivativeState(t, x_np, u_np)]], device=device, dtype=dtype)
            grad_x = grad_output * dLdx
        if ctx.needs_input_grad[2]:
            dLdu = torch.tensor([[mpc.getIntermediateCostDerivativeInput(t, x_np, u_np)]], device=device, dtype=dtype)
            grad_u = grad_output * dLdu
        return grad_t, grad_x, grad_u


class StateInputConstraint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, x, u):
        #In the forward pass we receive a Tensor containing the input and return
        #a Tensor containing the output. ctx is a context object that can be used
        #to stash information for backward computation. You can cache arbitrary
        #objects for use in the backward pass using the ctx.save_for_backward method.
        x_cpu = x.cpu()
        u_cpu = u.cpu()
        ctx.save_for_backward(t, x_cpu, u_cpu)
        x_np = x_cpu.t().detach().numpy().astype('float64')
        u_np = u_cpu.t().detach().numpy().astype('float64')
        g1 = torch.tensor(mpc.getStateInputConstraint(t, x_np, u_np), device=device, dtype=dtype)
        return g1

    @staticmethod
    def backward(ctx, grad_output):
        #In the backward pass we receive a Tensor containing the gradient of the loss
        #with respect to the output, and we need to compute the gradient of the loss
        #with respect to the input.
        grad_t = grad_x = grad_u = None
        t, x, u = ctx.saved_tensors
        x_np = x.t().detach().numpy().astype('float64')
        u_np = u.t().detach().numpy().astype('float64')

        if ctx.needs_input_grad[0]:
            raise NotImplementedError("Derivative of StateInputConstraint w.r.t. time not available")
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("Derivative of StateInputConstraint w.r.t. state not available")
        if ctx.needs_input_grad[2]:
            dg1du = torch.tensor(mpc.getStateInputConstraintDerivativeControl(t, x_np, u_np), device=device, dtype=dtype)
            grad_u = torch.matmul(grad_output, dg1du).reshape((-1, u_np.size))
        return grad_t, grad_x, grad_u


def control_Hamiltonian(tx, u_pred, dVdx, nu):
    f = FlowMap.apply(tx[0], tx[1:], u_pred)
    L = IntermediateCost.apply(tx[0], tx[1:], u_pred)
    hamiltonian = L + dVdx.dot(f)
    if systemHasConstraints:
        g1 = StateInputConstraint.apply(tx[0], tx[1:], u_pred)
        hamiltonian += g1.dot(nu)
    return hamiltonian


def num_samples_per_trajectory_point(t, max_num_points, half_value_decay_t):
    #Calculates number of samples drawn for each nominal state point in trajectory
    #:param t: Query time along trajectory
    #:param max_num_points:
    #:param half_value_decay_t: time into trajectory after which number of sampled point is halfed
    #:return: Number of samples to be drawn
    return max_num_points * np.exp(-np.log(2) * t / half_value_decay_t)


def trajectoryCost(policy, duration, dt_control):
    cost = 0.0  # running sum
    numStartingPoints = 1
    for _ in range(numStartingPoints):
        startPos = np.zeros([mpc.STATE_DIM, 1])
        tx = np.concatenate(([[0.0]], startPos))
        for it in range(int(duration / dt_control)):
            ttx_torch = torch.tensor(np.concatenate((tx[0, 0], tx[1:]), axis=None), dtype=dtype,
                                   device=device, requires_grad=False)
            p, u_pred = policy(ttx_torch)
            if len(p) > 1:
                u = torch.matmul(p, u_pred)
            else:
                u = u_pred[0]

            u_np = u.t().detach().numpy().astype('float64')
            cost += torch.tensor(mpc.getIntermediateCost(tx[0], tx[1:], u_np), device=device, dtype=dtype)
            if np.isnan(cost):
                return np.nan, tx[0]
            dx = mpc.computeFlowMap(tx[0], tx[1:], u_np)

            tx[1:] += dx.reshape(mpc.STATE_DIM, 1) * dt_control
            tx[0, 0] += dt_control
    return cost, duration
"""

def MSE_Loss(u0, u_pred):
    mseloss = torch.nn.MSELoss()
    return mseloss(u0, u_pred)

writer = SummaryWriter()


load_policy = False
if load_policy:
    save_path = "data/policy.pt"
    policy = torch.load(save_path)
    policy.eval()
else:
    policy = PolicyNet(STATE_DIM+1, INPUT_DIM)

policy.to(device)

print("Initial policy parameters:")
#print(list(policy.named_parameters()))

learning_rate = 1e-2
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)


load_memory = False
if load_memory:
    with open("data/memory.pkl", 'rb') as memFile:
        mem = pickle.load(memFile)
else:
    mem_capacity = 2000
    mem = ReplayMemory(mem_capacity)

#read string line by line and convert to float

f = open("mpcData.txt", "r")
#data = json.load(f)
lines = [line for line in f.readlines()]
f.close()


data = [[0.0]*(STATE_DIM+INPUT_DIM+1)  for i in range(len(lines))]
for i in range(len(lines)):
    data[i] = [float(val) for val in lines[i].split()]
    mem.push(data[i][0], data[i][1:STATE_DIM+1], None, None, None, None, data[i][STATE_DIM+1:STATE_DIM+INPUT_DIM+1])


# prepare saving of MPC solution trajectory (always add first point of a slq run)
#mpc_traj_len_sec = 3.0 # length of trajectories to generate with MPC
#dt_control = 1.0/400. # 400 Hz control frequency
#mpc_traj_t = np.linspace(0.0, mpc_traj_len_sec, int(mpc_traj_len_sec/dt_control))


last_policy_save_time = time.time()

learning_iterations = 10000

print("==============\nStarting training\n==============")
try:
    for it in range(learning_iterations):

        # extract batch of samples from replay memory
        batch_size = 2**5
        samples = mem.sample(batch_size)

        writeLogThisIteration = True

        def solver_step_closure():
            loss = torch.zeros([1], dtype=dtype, device=device)  # running sum over samples
            #mpc_H = torch.zeros([1], dtype=dtype, device=device)  # running sum over samples
            #g1_norm = 0.0  # running sum over samples
            for sample in samples:
                sum_u = torch.zeros([INPUT_DIM], dtype=dtype, device=device, requires_grad=False)
                tx = torch.tensor(np.concatenate((sample.t, sample.x), axis=None), dtype=dtype, device=device, requires_grad=False)
                ttx_net = torch.tensor(np.concatenate((sample.t, sample.x), axis=None), dtype=dtype, device=device, requires_grad=False)
                p, u_pred = policy(ttx_net)
                #dVdx = torch.tensor(sample.dVdx, dtype=dtype, device=device, requires_grad=False)
                #if systemHasConstraints:
                    #nu = torch.tensor(sample.nu, dtype=dtype, device=device, requires_grad=False)
                #else:
                    #nu = None
                for pi, u_pred_i in zip(p, u_pred): # loop through experts
                    sum_u += torch.mul(u_pred_i, pi)
                loss +=  MSE_Loss(torch.tensor(sample.u0), sum_u)
                #mpc_H += control_Hamiltonian(tx, torch.tensor(sample.u0), dVdx, nu)

                #if len(p) > 1:
                u_net = torch.matmul(p, u_pred)
                #else:
                #    u_net = u_pred[0]

                #if systemHasConstraints:
                    #g1_norm += np.linalg.norm(mpc.getStateInputConstraint(sample.t, sample.x, u_net.detach().numpy().astype('float64')))

            optimizer.zero_grad()
            loss.backward()

            global writeLogThisIteration
            if writeLogThisIteration:
                writer.add_scalar('loss/perSample', loss.item() / batch_size, it)
                print('loss/perSample ', loss.item() / batch_size, it)
                #writer.add_scalar('loss/constraintViolation', g1_norm / batch_size, it)
                writeLogThisIteration = False

            return loss


        #if it % 200 == 0:
        #    oc_cost, survival_time = trajectoryCost(policy=policy, duration=mpc_traj_len_sec, dt_control=dt_control)
        #    writer.add_scalar('metric/oc_cost', oc_cost, it)
        #    writer.add_scalar('metric/survival_time', survival_time, it)
        #    print("iteration", it, "oc_cost", oc_cost)

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


print("optimized policy parameters:")
#print(list(policy.named_parameters()))

now = datetime.datetime.now()
save_path = "armPolicy/mpcPolicy_" + now.strftime("%Y-%m-%d_%H%M%S")
print("saving policy to", save_path + ".pt")
torch.save(policy, save_path + ".pt")

# print("Saving data to", save_path+"_memory.pkl")
# with open(save_path+"_memory.pkl", 'wb') as outputFile:
#     pickle.dump(mem, outputFile)


writer.close()

print("Done. Exiting now.")
