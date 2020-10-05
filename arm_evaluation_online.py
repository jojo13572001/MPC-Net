
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
import jmpc

mpc = jmpc.Jmpc(port=5678)
mpc.resetTrajectory()
STATE_DIM = 12

getTrajectoryResponse = mpc.getTrajectory()
trajectoryLen = len(getTrajectoryResponse.get("result").get("times"))
finalTime = getTrajectoryResponse.get("result").get("times")[trajectoryLen-1] # length of trajectories to generate with MPC
dt = round(getTrajectoryResponse.get("result").get("times")[1] - getTrajectoryResponse.get("result").get("times")[0], 2) # 0.03s control duration

initState = getTrajectoryResponse.get("result").get("trajectory")[0]
initState.extend(np.zeros(int(STATE_DIM/2)))


def plot(save_path, t_end=10.0):
    policy = torch.load(save_path)

    tx = np.zeros((STATE_DIM + 1, 1))
    tx_history = np.zeros((trajectoryLen, STATE_DIM + 1))
    tx[1:, 0] = initState

    consumedTime=0.0
    for index in range(trajectoryLen):
        
        tx_history[index, :] = np.transpose(tx)
        print("index ", index, ", ", tx_history[index, :])
        tx_torch = torch.tensor(np.transpose(tx), dtype=torch.float, requires_grad=False)
        tx_torch[0][0] = 0.0 #optionally run it in MPC style

        start = timeit.default_timer()
        p, u_pred = policy(tx_torch)
        if len(p) > 1:
            u = torch.mm(p.t(), u_pred)
        else:
            u = u_pred[0]

        u_np = u.t().detach().numpy().astype('float64')
        stop = timeit.default_timer()
        consumedTime=consumedTime+(stop-start)

        #print("start policyReqResp, ", np.transpose(tx[1:])[0].tolist(), " ", tx[0][0])
        computePolicyResponse = mpc.computePolicy(np.transpose(tx[1:])[0].tolist(), tx[0][0])
        if computePolicyResponse.get("result") == False :
           print("Compute Policy Error!")
           sys.exit(0)

        jsonControl = mpc.getControl(dt, np.transpose(tx[1:])[0].tolist(), tx[0][0])
        
        jsonNextState = mpc.getNextState(u_np.tolist(), dt, np.transpose(tx[1:])[0].tolist()) #for mpc-net
        
        #jsonNextState = mpc.getNextState(jsonControl.get("result"), dt, np.transpose(tx[1:])[0].tolist()) #for mpc
        nextState = jsonNextState.get("result")
        #update timestamp and state
        np.transpose(tx[1:])[0] = nextState
        tx[0] = (index+1)*dt

    plt.figure(figsize=(STATE_DIM, STATE_DIM))
    print('MPC inference time: ', consumedTime/trajectoryLen)  
    print(tx_history[:,1:STATE_DIM])
    lineObjects = plt.plot(tx_history[:, 0], tx_history[:, 1:int(STATE_DIM/2)])
    plt.legend(iter(lineObjects), ('q1', 'q2','q3','q4','q5','q6'))

plot(save_path="armPolicy/mpcPolicy_2020-10-05_183930.pt", t_end=finalTime)
#plot(save_path="ballbot/mpcPolicy_2020-09-30_022409.pt", t_end=10.0)
plt.show()
