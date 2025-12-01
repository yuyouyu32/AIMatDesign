

echo "Start to train Cls..."
#!/bin/bash
cd /data/home/yeyongyu/SHU/ReinforceMatDesign/src
nohup python -u -m MLs.cls_trainer > ../logs/cls_trainer.log 2>&1 & 
# 获取第一个命令的进程 ID
pid=$!

# 等待第一个命令完成
wait $pid

sleep 20

echo "Cls Train finished, start to train ML..."
#!/bin/bash
cd /data/home/yeyongyu/SHU/ReinforceMatDesign/src
# 第一个命令立即执行
nohup python -u -m MLs.reg_trainer > ../logs/MLs.reg_trainer.log 2>&1 &

# 获取第一个命令的进程 ID
pid=$!

# 等待第一个命令完成
wait $pid

# 等待5分钟（300秒）
sleep 20

echo "ML Train finished, start to train RL..."

cd /data/home/yeyongyu/SHU/ReinforceMatDesign/src
# 第一个命令完成并等待5分钟后执行第二个命令
nohup python -m train --model td3 --batch_size 512 --total_steps 100000 --save_path ../ckpts/td3_seed21/ --start_timesteps 1000 --log_episodes 10 --eval_steps 512 --use_per --use_trust > ../logs/td3_train_seed21.log 2>&1 &


echo "RL Train finished"
