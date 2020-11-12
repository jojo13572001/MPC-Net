#/bin/bash
docker image build -t mpc:1.0 .
docker rm -f mpcTraining
docker run -i -v C:/Users/Administrator/Documents/git/MPC-Net/pick_place_mpc_app:/pick_place_mpc_app --workdir /pick_place_mpc_app -p 1234:1234 --name=mpcTraining mpc:1.0 bash run.sh &
sleep 1
python3 pybulletArm.py &
sleep 1
python3 arm_learner.py