python3 federated_main.py --model=mlp --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.01 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3




python3 -u lora_central.py --model=mlp_lora --dataset=cifar10 --gpu=1 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 > central_lora.log 2>&1 &
python3 -u lora_central.py --model=mlp --dataset=cifar10 --gpu=2 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.01 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 > central.log 2>&1 &


python3 -u lora_central_reset.py --model=mlp_lora --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 > central_reset1.log 2>&1 &
python3 -u lora_central_reset.py --model=mlp_lora --dataset=cifar10 --gpu=3 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 --reset 10 > central_reset10.log 2>&1 &




nohup python3 -u lora_central_reset.py --model=mlp_lora --dataset=cifar10 --gpu=3 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 --reset 10 > central_reset10.log 2>&1 &


nohup python3 -u federated_main.py --model=mlp --dataset=cifar10 --gpu=1 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.01 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 > fedavg.log 2>&1 &

nohup python3 -u lora_split.py --model=mlp --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 > fedlora_split.log 2>&1 &



python3 -u lora_split.py --model=mlp --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3




nohup python3 -u lora.py --model=mlp_lora --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 > fedlora_0.1.log 2>&1 &

nohup python3 -u lora.py --model=mlp_lora --dataset=cifar10 --gpu=3 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 > fedlora_1.log 2>&1 &

nohup python3 -u lora_reset.py --model=mlp_lora --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 --reset 30 > fedlora_reset30_0.1.log 2>&1 &



python3 -u lora_central_reset.py --model=mlp_lora --dataset=cifar10 --gpu=3 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 --reset 200 --r 8


nohup python3 -u lora_central.py --model=mlp --dataset=cifar10 --gpu=3 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.01 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 > cent_mlp.log 2>&1 &

nohup python3 -u lora_central_reset.py --model=mlp_lora --dataset=cifar10 --gpu=2 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 --reset 200 --r 8 > cent_reset200.log 2>&1 &



nohup python3 -u lora_central.py --model=mlp --dataset=cifar10 --gpu=3 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.01 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 > cent_mlp.log 2>&1 &

nohup python3 -u lora_central_reset.py --model=mlp_lora --dataset=cifar10 --gpu=2 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 --reset 200 --r 8 > cent_reset200.log 2>&1 &

nohup python3 -u lora_central_reset.py --model=mlp_lora --dataset=cifar10 --gpu=1 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 --reset 10 --r 8 > cent_reset10.log 2>&1 &

nohup python3 -u lora_central_reset.py --model=mlp_lora --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=150 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1 --local_ep=3 --eps=0 --max_init=1e-3 --reset 1 --r 8 > cent_reset1.log 2>&1 &