# TODO: rewrite this
import numpy as np
import random

class DataReader:
    def __init__(self, data_path, batch_size):
        self._batch_size = batch_size
        with open(data_path, encoding = 'ISO-8859-1') as f:
            d_lines = f.read().splitlines()

        self._data = []
        self._labels = []
        self._sequence_lengths = []
        for data_id, line in enumerate(d_lines):
            features = line.split('<fff>')
            label, doc_id, sequence_length = int(features[0]), int(features[1]), int(features[2])
            tokens = features[3].split()
            self._data.append(tokens)
            self._labels.append(label)
            self._sequence_lengths.append(sequence_length)

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)
        self._sequence_lengths = np.array(self._sequence_lengths)
        self._num_epoch = 0
        self._batch_id = 0

    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end > len(self._data):
            end = len(self._data)
            start = end - self._batch_size
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            random.seed(2018)
            random.shuffle(indices)
            self._data, self._labels, self._sequence_lengths = self._data[indices], self._labels[indices], self._sequence_lengths[indices]
        final_tokens = np.array([sequence[-1] for sequence in self._data[start:end]])
        return self._data[start:end], self._labels[start:end], self._sequence_lengths[start:end], final_tokens