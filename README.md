# infra
Repository containing code to train an LSTM, e.g. on MNIST. Note that this repository is more for showing that LSTMs can also be used to do image classification. To use the right inductive bias, a CNN/ResNet/DenseNet/etc. should be preferred, since an LSTM treats the image sequentially, i.e. pixel by pixel. 

## Options

The main script `run.py` can be run with different options,

```
options:
  -h, --help            show this help message and exit
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
  --num_classes NUM_CLASSES
                        Number of classes NN has to predict at the end.
  --num_epochs NUM_EPOCHS
                        Number of epochs used for training of the NN.
  --batch_size BATCH_SIZE
                        Number of batches that are used for one ADAM update rule.
  --load_cp             Whether to load preexisting checkpoint(s) of the model.
  --bidirectional       Whether to use bidirectional LSTM (True) or not (False). Default: True.
  --test TEST           If the random variables for the noise are drawn from a Gaussian distribution, then this argument determines the standard deviation of the Gaussian distribution.
```