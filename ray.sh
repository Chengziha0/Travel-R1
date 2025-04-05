export CUDA_VISIBLE_DEVICES=0,1,2,3
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4 --ray-debugger-external