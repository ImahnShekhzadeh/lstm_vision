# mnist-lstm
Repository containing code to train an LSTM, e.g. on MNIST. Note that this repository is more for showing that LSTMs can also be used to do image classification. To use the right inductive bias, a CNN/ResNet/DenseNet/etc. should be preferred, since an LSTM treats the image sequentially, i.e. pixel by pixel. 

## Options

The main script `run.py` can be run with different options,

```
options:
  -h, --help            show this help message and exit
  --dropout_rate DROPOUT_RATE
                        Dropout rate for the dropout layer.
  --freq_output__train FREQ_OUTPUT__TRAIN
                        Frequency of outputting the training loss and accuracy.
  --freq_output__val FREQ_OUTPUT__VAL
                        Frequency of outputting the validation loss and accuracy.
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

I ran the script `run.py` as follows:
```
docker build -f Dockerfile -t mnist-lstm:1.0.0 .
docker run --rm -v $(pwd)/MNIST:/app/MNIST -v $(pwd)/mnist-lstm:/app/scripts --gpus all -it mnist-lstm:1.0.0
```
where I assume that the `MNIST` folder already exists locally. If not, please download it manually first by extracting [`mnist.zip`](mnist.zip).

The options I used are under `run_scripts.sh`.

## Results

Training a bidirectional LSTM for `10` epochs results in,
```
Train data: Got 48974/50000 with accuracy 97.95 %
Test data: Got 9764/10000 with accuracy 97.64 %
```
On a machine with an NVIDIA RTX 4090 with an Intel i5-10400, training for `10` epochs takes about `44` s, and in total about `1.7` GB of memory is required.

With the flag `--use_amp`, training for `10` epochs takes about `43` s on the same hardware, i.e. there is not a huge runtime gain, 
but the memory consumption is only about `950` MB. The final accuracy on the train data remains the same, and the 
accuracy on the test data is about `97.63 %`, i.e. the performance basically remains the same with dynamic 
casting to `torch.float16` enabled!

## TODO
[ ] Write github action workflow for `isort` & `black` and pin the versions.
[ ] Rename the git repo to `lstm_vision` and put the dataset loading, i.e. L50-L83, into `run.py`, into a separate function `get_datasets()`.