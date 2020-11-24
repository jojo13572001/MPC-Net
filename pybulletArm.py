import pybullet
import time
import jmpc
import json
import zmq
import settings

pybulletServer = jmpc.JmpcServer(port=1235)

#pybullet.connect(p.GUI, options="--minGraphicsUpdateTimeMs=16000")
pybullet.connect(pybullet.GUI)

import pybullet_data
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

#pybullet.resetSimulation()
#pybullet.setPhysicsEngineParameter(numSolverIterations=150)
jointTorqueLimit = [104, 104, 69, 69, 34, 34]
jointPositionLimit = [3.14, 2.35, 2.61, 3.14, 2.56, 3.14]
jointVelocityLimit = [1.57, 1.57, 1.57, 1.57, 1.57, 1.57]

robotPosition = [0, 0, 0]
pybullet.resetDebugVisualizerCamera(cameraDistance=1.7, cameraYaw=-110, cameraPitch=-20, cameraTargetPosition=robotPosition)
plane = pybullet.loadURDF("plane.urdf")
robot = pybullet.loadURDF("elfin3_urdf/elfin3_orig.urdf", robotPosition, useFixedBase=1)  # use a fixed base!
#position, orientation = pybullet.getBasePositionAndOrientation(robot)
#print("Base postion ", position, "Base orientation ", orientation)
totoalJointNum = pybullet.getNumJoints(robot)
jointStartIndex = 2
jointNum = 6

# Set Environment
pybullet.setGravity(0, 0, -9.81)
pybullet.setTimeStep(1./240.)
pybullet.setRealTimeSimulation(0)
for i in range(totoalJointNum):
    pybullet.enableJointForceTorqueSensor(robot, i, enableSensor=1)

pybullet.changeDynamics(robot, -1, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0)
for i in range(jointNum):
    pybullet.changeDynamics(robot, i+jointStartIndex, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0,
                                                      maxJointVelocity=jointVelocityLimit[i],
                                                      jointLimitForce=jointTorqueLimit[i],
                                                      jointLowerLimit=-jointPositionLimit[i],
                                                      jointUpperLimit=jointPositionLimit[i])

def resetState(resetState):
    #reset arm to specified state and get current torque

    #reset directly to specified state and forward simulation (this way can't get torque)
    for i in range(jointNum):
           pybullet.resetJointState(robot, i+jointStartIndex, resetState[i])
    pybullet.stepSimulation()


    #reset again directly to specified state. Using POSITION_CONTROL, we can get torque
    for i in range(totoalJointNum):
        pybullet.setJointMotorControlArray(robot, range(jointStartIndex, jointStartIndex+jointNum), pybullet.POSITION_CONTROL,
                                           targetPositions=resetState[:jointNum])

    #Using POSITION_CONTROL, we have to wait a moment to get into stable state
    for _ in range(240):
        pybullet.stepSimulation()

    joint_positions = [state[0] for state in pybullet.getJointStates(robot, range(pybullet.getNumJoints(robot)))]
    
    #Turn off automatical motor control
    pybullet.setJointMotorControlArray(robot, range(totoalJointNum), pybullet.VELOCITY_CONTROL, forces=[0]*totoalJointNum)

    #Return current torque used in 'resetState'
    joint_velocities = [state[1] for state in pybullet.getJointStates(robot, range(pybullet.getNumJoints(robot)))]
    joint_torques = [state[3] for state in pybullet.getJointStates(robot, range(pybullet.getNumJoints(robot)))]
    return joint_torques, joint_positions[jointStartIndex:jointStartIndex+jointNum]+joint_velocities[jointStartIndex:jointStartIndex+jointNum]

def rendering(initTorques, color):
    duration = 0
    end_effector_xyz_start = []
    end_effector_xyz_end = []
    lineLength = 2
    while duration<7*(trajectoryLen-1):
        if (duration%7) == 0:
           #print(duration/7, " Wait to Receive MPC Control")
           joint_torques, stop = pybulletServer.recvControl()
           if stop == True:
              pybulletServer.responseCurrentState([])
              break
           joint_positions = [state[0] for state in pybullet.getJointStates(robot, range(pybullet.getNumJoints(robot)))]
           joint_velocities = [state[1] for state in pybullet.getJointStates(robot, range(pybullet.getNumJoints(robot)))]
           if duration%(7*lineLength) == 0:
              end_effector_xyz_start = pybullet.getLinkState(robot,7)[4]

        pybullet.setJointMotorControlArray(robot, range(totoalJointNum), pybullet.TORQUE_CONTROL, forces=[0,0]+
                                                                                                         joint_torques + 
                                                                                                         [0,0])
        pybullet.stepSimulation()

        if (duration%7) == 6:
           joint_positions = [state[0] for state in pybullet.getJointStates(robot, range(pybullet.getNumJoints(robot)))]
           joint_velocities = [state[1] for state in pybullet.getJointStates(robot, range(pybullet.getNumJoints(robot)))]
           pybulletServer.responseCurrentState(joint_positions[jointStartIndex:jointStartIndex+jointNum]+joint_velocities[jointStartIndex:jointStartIndex+jointNum])
           if duration%(7*lineLength) == 6:
              end_effector_xyz_end = pybullet.getLinkState(robot,7)[4]
              pybullet.addUserDebugLine(end_effector_xyz_start, end_effector_xyz_end, color, lineWidth=2)
        duration+=1

def pureRendering(color, lineLength, jointStartIndex ,jointNum, robotID, trajectoryLen):
    index = 0
    trajectoryLen = 100000
    end_effector_xyz_start = []
    end_effector_xyz_end = []
    lineLength = 2
    while index+1 < trajectoryLen:
        state, index= pybulletServer.recvState()
        for i in range(jointNum):
           pybullet.resetJointState(robotID, i+jointStartIndex, state[i])

        if index%lineLength == 0:
           end_effector_xyz_start = pybullet.getLinkState(robot,7)[4]

        if index%lineLength == 1:
           end_effector_xyz_end = pybullet.getLinkState(robot,7)[4]
           pybullet.addUserDebugLine(end_effector_xyz_start, end_effector_xyz_end, color, lineWidth=2)


def resetStateRendering(color, lineLength, jointStartIndex ,jointNum, robot, trajectoryLen):
    index = 0
    end_effector_xyz_start = []
    end_effector_xyz_end = []
    totoalJointNum = pybullet.getNumJoints(robot)
    while index+1 < trajectoryLen:
        state, index= pybulletServer.recvState()
        joint_torques, stop = pybulletServer.recvControl()
        if stop == True:
           pybulletServer.responseCurrentState([])
           break
        
        resetState(state)
        pybullet.setJointMotorControlArray(robot, range(totoalJointNum), pybullet.TORQUE_CONTROL, forces=[0,0]+
                                                                                                         joint_torques + 
                                                                                                         [0,0])
        pybullet.stepSimulation()

        joint_positions = [state[0] for state in pybullet.getJointStates(robot, range(pybullet.getNumJoints(robot)))]
        joint_velocities = [state[1] for state in pybullet.getJointStates(robot, range(pybullet.getNumJoints(robot)))]
        #print(count/7, " Returning Final State ", joint_positions[2:2+jointNum]+joint_velocities[2:2+jointNum])
        pybulletServer.responseCurrentState(joint_positions[jointStartIndex:jointStartIndex+jointNum]+joint_velocities[jointStartIndex:jointStartIndex+jointNum])
        
        if index%lineLength == 0:
           end_effector_xyz_start = pybullet.getLinkState(robot,7)[4]

        if index%lineLength == 1:
           end_effector_xyz_end = pybullet.getLinkState(robot,7)[4]
           pybullet.addUserDebugLine(end_effector_xyz_start, end_effector_xyz_end, color, lineWidth=2)

def showJointInfo():
    joint_info = pybullet.getJointInfo(robot, jointStartIndex)
    print("################################ Robot Arm Info ##########################################")
    print("joint_index ", joint_info[0], "\nname ", joint_info[1]," \njoint_type ", joint_info[2]," \nqIndex ", joint_info[3], 
        "\nuIndex ", joint_info[4], "\nflags ",joint_info[5], "\njointDamping ", joint_info[6], "\njointFriction ", joint_info[7], 
        "\nlower_limit ", joint_info[8], " \nupper_limit ", joint_info[9], "\njointMaxForce ", joint_info[10],
        "\njointMaxVelocity ", joint_info[11], "\nlinkName ", joint_info[12], "\njointAxis ", joint_info[13],
        "\nparentFramePos ", joint_info[14], "\nparentFrameOn ", joint_info[15], "\nparentIndex ", joint_info[16])
    print("###########################################################################################")


#showJointInfo()

lineLength = 2
initState, dt, trajectoryLen, learningIterations = pybulletServer.recvInitState()

######################Training#########################
learningIterationIndex = 0
if settings.enablePybulletTraining == True:
   while (learningIterationIndex+1 < learningIterations):
     initTorques, firstState = resetState(initState.copy())
     learningIterationIndex = pybulletServer.recvInitTrainingState(firstState)
     rendering(initTorques, [1,0,0])

######################MPC-Net Rendering###########################
#Ploct MPC
initTorques, _= resetState(initState.copy())
rendering(initTorques, [1,0,0])
if settings.currentRendering == "enableRendering":
   #Plot MPC-NET
   initTorques, _ = resetState(initState.copy())
   rendering(initTorques, [0,0,1])
elif settings.currentRendering == "enablePureRendering":
   pureRendering([0,0,1], lineLength, jointStartIndex, jointNum, robot, trajectoryLen)
elif settings.currentRendering == "enableResetStateRendering":
   resetStateRendering([0,0,1], lineLength, jointStartIndex ,jointNum, robot, trajectoryLen)

while True:
    pass

pybullet.disconnect()