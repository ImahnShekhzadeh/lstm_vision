# lstm-vision
Repository containing PyTorch code to train an LSTM on a CV dataset, e.g. MNIST. In principle, any image dataset can be used, for this change [these lines](https://github.com/ImahnShekhzadeh/lstm-vision/blob/main/mnist-lstm/functions.py#L73-L95) and don't forget to adjust the flag `channels_img`, which is by default `1`.

The code can be run both with [AMP](https://pytorch.org/docs/stable/amp.html) (automatic mixed precision) enabled and `torch.compile()`.

Note that this repository is more for showing that LSTMs can also be used to do image classification. To use the right inductive bias, a CNN/ResNet/DenseNet/etc. should be preferred, since an LSTM treats the image sequentially, i.e. pixel by pixel.

## Options

The main script `run.py` can be run with different options,

```
options:
  -h, --help            show this help message and exit
  --compile_mode COMPILE_MODE
                        Mode for compilation of the model when using `torch.compile`.
  --dropout_rate DROPOUT_RATE
                        Dropout rate for the dropout layer.
  --freq_output__train FREQ_OUTPUT__TRAIN
                        Frequency of outputting the training loss and accuracy.
  --freq_output__val FREQ_OUTPUT__VAL
                        Frequency of outputting the validation loss and accuracy.
  --max_norm MAX_NORM
                        Max norm for gradient clipping.
  --num_workers NUM_WORKERS
                        Number of subprocesses used in the dataloaders.
  --pin_memory PIN_MEMORY
                        Whether tensors are copied into CUDA pinned memory.
  --saving_path SAVING_PATH
                        Saving path for the files (loss plot, accuracy plot, etc.)
  --seed_number SEED_NUMBER
                        If specified, seed number is used for RNG.
  --sequence_length SEQUENCE_LENGTH
                        Sequence length for the RNN, input: (batch_size, sequence_length, input_size)
  --input_size INPUT_SIZE
                        Input size for the RNN, input: (batch_size, sequence_length, input_size)
  --hidden_size HIDDEN_SIZE
                        Hidden size for the first LSTM layer.
  --num_layers NUM_LAYERS
                        Number of stacked LSTM layers.
  --channels_img CHANNELS_IMG
                        Number of channels in the MNIST input images.
  --learning_rate LEARNING_RATE
                        Learning rate for the training of the NN.
  --num_epochs NUM_EPOCHS
                        Number of epochs used for training of the NN.
  --batch_size BATCH_SIZE
                        Number of batches that are used for one ADAM update rule.
  --load_cp             Whether to load preexisting checkpoint(s) of the model.
  --bidirectional       Whether to use bidirectional LSTM.
  --train_split TRAIN_SPLIT
                        Split ratio of train and validation set.
  --use_amp             Whether to use automatic mixed precision (AMP).
```

## Run

### Basics
I ran the script `run.py` as follows:
```
docker build -f Dockerfile -t lstm-vision:1.2.0 .
docker run --shm-size 512m --rm -v $(pwd):/app --gpus all -it lstm-vision:1.2.0 --config configs/conf.json
```

The options for training I used are in `configs/conf.json`.

### W&B
If you want to log some metrics to [Weights & Biases](https://wandb.ai/), append the following to the `docker run` command:
```
--wandb__api_key <api_key>
# --wandb__api_key 2fru...
```

### Distributed Data Parallel (DDP)
You can also easily specify a [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) setup by passing the flag `--use_ddp` in the `configs/conf.json` file (for this, ensure you have more than one NVIDIA GPU). Note that if running this on the [runpod.io](https://www.runpod.io/), you need to change the line
```python
os.environ["MASTER_ADDR"] = "localhost"
```
in the `setup()` function `functions.py` to
```python
os.environ["MASTER_ADDR"] = "<ip_address>"
# os.environ["MASTER_ADDR"] = "192.xxx.xx.x"
```
where `ip_address` can be obtained via `hostname -I`.

## Results

All results were obtained on a single GPU. For this small model, I do not recommend a DDP setup.

Training a bidirectional LSTM with roughly `3.9`M params for `50` epochs results in,
```
Train data: Got 49824/50000 with accuracy 99.65 %
Test data: Got 9899/10000 with accuracy 98.99 %
```
On a machine with an NVIDIA RTX 4090 with an Intel i5-10400, training for `50` epochs takes about `232` s, and in total about `12.56` GB of GPU memory are required. Note that without the `--use_amp` flag, which is specified in `configs/conf.json`, about double the memory will be required. If you have a GPU with less than already `12.56` GB VRAM, decrease the batch size.

I also tried the flag `--compile_mode` (with all modes "default", "reduce-overhead" & "max-autotune"), and noticed that the runtime slightly _increases_ when using the MNIST dataset. This happens, since the warmup phase, cf. [here](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), takes a long time, and after the warmup phase, the runtime epoch is comparable to no compilation. However, for other CV datasets (e.g. CIFAR100) and other model architectures, this might change! Also, please note that `torch.compile(..., full_graph=False)` has to be used, since `TorchDynamo` does not allow `full_graph=True` for RNNs/GRUs/LSTMs.

The above results were obtained with $10 \%$ label smoothing. I varied the label smoothing between $0 \%$ and $10 \%$ in steps of $2 \%$ and noticed that the greater the label smoothing, the higher the train and validation losses per epoch.

# TODO
- log git branch and git commit hash
- use torchvision.transforms.v2 instead of torchvision.transforms
- when calculating accuracy, use `torch.autocast()`, and use `@torch.no_grad()` decorator
- set shuffle to `False` for val loader
