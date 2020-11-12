#/bin/bash
docker rm -f mpcTraining
docker run -i -v C:/Users/Administrator/Documents/git/MPC-Net/pick_place_mpc_app:/pick_place_mpc_app --workdir /pick_place_mpc_app -p 1234:1234 --name=mpcTraining mpc:1.0 bash run.sh
#cd pick_place_mpc_app
#./pick_place_mpc_app --port=1234 --server --config=elfin3.info --wait 5 &
#cd ..
sleep 1
python pybulletArm.py &
sleep 1
python arm_learner.py