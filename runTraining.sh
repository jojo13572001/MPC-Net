#/bin/bash
cd pick_place_mpc_app
sudo chmod 777 pick_place_mpc_app
sudo docker image build -t mpc:1.0 .
sudo docker rm -f mpcTraining
sudo docker run -i -v /home/bean/catkin_ws/src/MPC-Net/pick_place_mpc_app:/pick_place_mpc_app --workdir /pick_place_mpc_app -p 1234:1234 --name=mpcTraining mpc:1.0 bash run.sh &
sleep 1
cd ..
python3 pybulletArm.py &
pip3 install torch tensorboardX matplotlib
python3 arm_learner.py