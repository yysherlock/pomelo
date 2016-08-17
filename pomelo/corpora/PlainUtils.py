import numpy as np
from six.moves import xrange

def get_dataset(fname):
    for line in open(fname):
        for word in line.split():
            yield word
        # Add token to the end of the line
        # Equivalent to <eos> in:
        # https://github.com/wojzaremba/lstm/blob/master/data.lua#L32
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L31
        yield '<eos>'

def rnn_data_iterator(raw_data, batch_size, num_steps):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  for i in range(epoch_size):
    x = data[:, i * num_steps:(i + 1) * num_steps]
    y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
    yield (x, y)

def epoch_data_iterator(data,maxepoch,splits=1):
    """
    Generator a splited `epoch data` of `data`. This is because sometimes the whole data is too big,
    it takes too much time to train and the parameters may even achieve the optimal half of the training.
    So we split data to small parts, provide each part as an epoch data for the `run_epoch` function.
    Args:
        data: whole data
        maxepoch: maximum epochs for training
        splits: split data into how many parts
    """
    data_len = len(data)
    split_len = data_len // splits
    splits_starts = [ i*split_len for i in range(splits) if (i+1)*split_len <= data_len ]
    splits_ends = [start + split_len for start in splits_starts]
    if splits_ends[-1] < data_len: splits_ends[-1] = data_len

    for epoch in xrange(maxepoch):
        cur_split = epoch % splits
        yield data[splits_starts[cur_split] : splits_ends[cur_split]]

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def data_iterator(orig_X, orig_y=None, batch_size=32, label_size=2, shuffle=False):
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(orig_X))
    data_X = orig_X[indices]
    data_y = orig_y[indices] if np.any(orig_y) else None
  else:
    data_X = orig_X
    data_y = orig_y
  ###
  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    x = data_X[batch_start:batch_start + batch_size]
    # Convert our target from the class index to a one hot vector
    y = None
    if np.any(data_y):
      y_indices = data_y[batch_start:batch_start + batch_size]
      y = np.zeros((len(x), label_size), dtype=np.int32)
      y[np.arange(len(y_indices)), y_indices] = 1
    ###
    yield x, y
    total_processed_examples += len(x)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)
