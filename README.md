## Environment
Ubuntu 18

## Requirement
pip3 install -r requirements.txt

## Running the mpc-net Training
Modify settings.py, set enablePybulletTraining = True<br>
`bash runTraining.sh`

## Running the mpc-net Rendering
Modify settings.py, set enablePybulletTraining = False<br>
set enablePybulletTraining = False<br>
set currentRendering = one of ["enableMpcRendering", "enablePybulletRendering", "enableResetStateRendering"]<br>
`bash runRendering.sh`


## To monitor progress
execute tensorboard<br>
`tensorboard --logdir runs`