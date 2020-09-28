import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit

import sys
#sys.path.append(os.environ["HOME"]+"/catkin_ws/devel/lib/python3.6/dist-packages/ocs2_ballbot_example")

#####wired workaround that reading file should before calling BallbotPyBindings
"""
f = open("mpcData.txt", "r")
lines=f.readlines()
print(lines)
mpc_history = np.zeros((4000, 6))
index = 0
for line in lines:
   mpcData = np.fromstring(line, dtype=float, sep=' ')
   mpc_history[index] = mpcData
   index=index+1
f.close()
"""
#####

#from BallbotPyBindings import mpc_interface, scalar_array, state_vector_array, input_vector_array, dynamic_vector_array, cost_desired_trajectories
#mpc = mpc_interface("mpc", False)

STATE_DIM=14
def computeFlowMap(x0, u_np):
    variableDim = int(STATE_DIM/2);
    qDot = x0[variableDim:]
    qDDot = np.reshape(u_np, (variableDim,1))
    return np.vstack((qDot, qDDot))

def plot(save_path, t_end=10.0):
    policy = torch.load(save_path)

    #dt = 1./400.
    dt = 1.0
    tx0 = np.zeros((STATE_DIM + 1, 1))
    tx0[1, 0] = -3.879
    tx0[2, 0] = -0.991
    tx0[3, 0] = -4.015
    tx0[4, 0] = 0.761
    tx0[5, 0] = -2.440
    tx0[6, 0] = -1.569
    tx0[7, 0] = 1.119
    tx0[8, 0] = -0.014
    tx0[9, 0] = -0.002
    tx0[10, 0] = 0.026
    tx0[11, 0] = -0.022
    tx0[12, 0] = -0.016
    tx0[13, 0] = -0.023
    tx0[14, 0] = 0.000

    tx_history = np.zeros((int(t_end/dt)+1, STATE_DIM+1))

    tx = tx0
    average_constraint_violation = 0
    steps = int(t_end/dt)
    #print("tx0 steps", tx0, steps)
    count=0
    result=0.0
    for it in range(steps):
        tx_history[it, :] = np.transpose(tx)
        tx_torch = torch.tensor(np.transpose(tx), dtype=torch.float, requires_grad=False)
        tx_torch[0][0] = 0.0 #optionally run it in MPC style

        start = timeit.default_timer()
        p, u_pred = policy(tx_torch)
        if len(p) > 1:
            u = torch.mm(p.t(), u_pred)
        else:
            u = u_pred[0]

        u_np = u.t().detach().numpy().astype('float64')

        dx = computeFlowMap(tx[1:], u_np)
        stop = timeit.default_timer()
        result=result+(stop-start)
        count=count+1
        np.transpose(tx[1:])[0] += np.transpose(dx*dt)[0]
        tx[0] += dt

    tx_history[it+1,:] = np.transpose(tx)

    plt.figure(figsize=(14, 14))
    print('MPC inference time: ', result/count)  
    #print(tx_history[:,1:6])
    """
    f = open("moveData.txt", "w")
    for j in range(steps):
        for i in range(6):
            f.write(str(tx_history[j,i])+" ")
        f.write("\n")
    f.close()
    """
    lineObjects = plt.plot(tx_history[:, 0], tx_history[:, 1:STATE_DIM+1])
    #lineObjects = plt.plot(tx_history[:, 0:6])
    plt.legend(iter(lineObjects), ('q1', 'q2','q3','q4','q5','q6','q7','qD1','qD2','qD3','qD4','qD5','qD6','qD7'))

plot(save_path="armPolicy/mpcPolicy_2020-09-22_162433.pt", t_end=10.0)
#plot(save_path="tmp/mpcPolicy_2020-08-04_193847.pt", t_end=10.0)
plt.show()
