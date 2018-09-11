CUDA_VISIBLE_DEVICES=1,2 python main.py ./data/cifar.python --dataset cifar10 --arch resnet_fr --save_path ./snapshots/cifar10_resnet20_split_2 --epochs 300 --learning_rate 0.01 --schedule 150 225 --gammas 0.1 0.1 --batch_size 128 --manualSeed 2 --depth 20 --splits 2 

