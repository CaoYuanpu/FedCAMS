python3 -u lora_reset.py --model=mlp_lora --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 --reset 1

python3 -u lora_reset.py --model=mlp_lora --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 --reset 10

python3 -u lora_reset.py --model=mlp_lora --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 --reset 50

python3 -u lora_reset.py --model=mlp_lora --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 --reset 70

python3 -u lora_reset.py --model=mlp_lora --dataset=cifar10 --gpu=0 --local_bs=20 --epochs=500 --iid=1 --optimizer=fedavg --local_lr=0.1 --lr=1.0 --local_ep=3 --eps=0 --max_init=1e-3 --reset 90