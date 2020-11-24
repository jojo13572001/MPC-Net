renderMethods = ["enablePureRendering", "enableRendering", "enableResetStateRendering"]
currentRendering = renderMethods[1]
enablePybulletTraining = False
enableSampling = True
loadMemory = not enableSampling
loadPolicy = not enableSampling
loadPolicyPath = 'armPolicy/pyBullet/1115/mpcPolicy_2020-11-15_225649.pt'
loadMemoryPath = "armPolicy/pyBullet/1115/mpcPolicy_2020-11-15_233515_memory.pkl"

learning_iterations = 100000
save_path = "armPolicy/pyBullet/1124/mpcPolicy_"