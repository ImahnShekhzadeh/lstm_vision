# lstm-vision
Repository containing PyTorch code to train an LSTM on a CV dataset, e.g. MNIST. In principle, any image dataset can be used, for this change [these lines](https://github.com/ImahnShekhzadeh/lstm-vision/blob/main/mnist-lstm/functions.py#L73-L95) and don't forget to adjust the flag `channels_img`, which is by default `1`.

The code can be run both with [AMP](https://pytorch.org/docs/stable/amp.html) (automatic mixed precision) enabled and `torch.compile()`.

Note that this repository is more for showing that LSTMs can also be used to do image classification. To use the right inductive bias, a CNN/ResNet/DenseNet/etc. should be preferred, since an LSTM treats the image sequentially, i.e. pixel by pixel.

## Run

### Single-GPU
On a single-GPU machine, I ran the script `run.py` as follows:
```
docker build -f Dockerfile -t lstm-vision:1.4.0 .
docker run --shm-size 512m --rm -v $(pwd):/app --gpus all -it lstm-vision:1.4.0 python -B /app/lstm_vision/run.py training.saving_path=[...]
```
To check all available config keys, check out the file `configs/conf.yaml`.

If you only want to evaluate the model from a pre-existing checkpoint, add
```
model.loading_path=[...] training.num_epochs=0
```

### Multiple GPUs
You can easily specify to use [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) during training, which uses several GPUs if available:
```
docker run --shm-size 512m --rm -v $(pwd):/app --gpus all -it lstm-vision:1.4.0 torchrun --nproc_per_node=NUM_GPUS_YOU_HAVE /app/lstm_vision/run.py training.use_ddp=true training.master_addr='"<ip-address>"'
- training.master_port='"<port>"' training.saving_path=[...]
```
where the master address is the IP address that can be obtained via `hostname -I`. If you only want to evaluate the model from a pre-existing checkpoint, add
```
model.loading_path=[...] training.num_epochs=0
```

### W&B
If you want to log some metrics to [Weights & Biases](https://wandb.ai/), append the following to the `docker run` command:
```
training.wandb__api_key=<your_key>
# training.wandb__api_key=2fru...
```

## Results

All results were obtained on a single GPU. For this small model, I do not recommend a DDP setup.

Training a bidirectional LSTM with roughly `3.9`M params for `50` epochs results in,
```
Train data: Got 49839/50000 with accuracy 99.68 %
Test data: Got 9906/10000 with accuracy 99.06 %
```
On a machine with an NVIDIA RTX 4090 with an Intel i5-10400, training for `50` epochs takes about `232` s, and in total about `12.61` GB of GPU memory are required. Note that without the `--use_amp` flag, which is specified in `configs/conf.json`, about double the memory will be required. If you have a GPU with less than already `12.61` GB VRAM, decrease the batch size.

I also tried a compilation mode (`training.compile_mode`) with all modes "default", "reduce-overhead" & "max-autotune", and noticed that the runtime slightly _increases_ when using the MNIST dataset. This happens, since the warmup phase, cf. [here](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), takes a long time, and after the warmup phase, the runtime epoch is comparable to no compilation. However, for other CV datasets (e.g. CIFAR100) and other model architectures, this might change! Also, please note that `torch.compile(..., full_graph=False)` has to be used, since `TorchDynamo` does not allow `full_graph=True` for RNNs/GRUs/LSTMs.

The above results were obtained with $10 \%$ label smoothing. I varied the label smoothing between $0 \%$ and $10 \%$ in steps of $2 \%$ and noticed that the greater the label smoothing, the higher the train and validation losses per epoch.
