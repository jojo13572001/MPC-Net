#/bin/bash
cd pick_place_mpc_app
sudo docker build -t mpc:1.0 .
sudo docker rm -f mpcRendering
sudo docker run -i -v /home/bean/catkin_ws/src/MPC-Net/pick_place_mpc_app:/pick_place_mpc_app --workdir /pick_place_mpc_app -p 1234:1234 --name=mpcRendering mpc:1.0 bash run.sh &
cd ..
#./pick_place_mpc_app --port=1234 --server --config=elfin3.info --wait 5 &
pip3 install torch tensorboardX matplotlib
sleep 1
python3 pybulletArm.py &
sleep 1
python3 arm_evaluation.py