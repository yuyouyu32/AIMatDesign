python -u -m MLs.reg_trainer
nohup python -u -m MLs.reg_trainer > ../logs/reg_trainer.log 2>&1 &
nohup python -u -m MLs.cls_trainer > ../logs/cls_trainer.log 2>&1 & 

# pkill -f "python -u -m MLs.reg_trainer"