import torch.utils.data as data
import numpy as np


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class XORDataset(data.Dataset):

    def __init__(self, size, seed, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            seed - Seed used to create the PRNG state with which we want to generate the data points
            std - Standard deviation of the noise
        """
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed=seed)
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        """
        Each data point in the XOR dataset consists of two binary values, (x, y).
        The corresponding label to the datapoint is:
        (1, 1) = 0
        (1, 0) = 1
        (0, 1) = 1
        (0, 0) = 0
        """

        data = self.np_rng.randint(low=0, high=2, size=(self.size, 2)).astype(np.float32)
        label = (data.sum(axis=1) == 1).astype(np.int32)
        # We add some noise to the datapoints (after generating the labels) to
        # artificially make the dataset more complex
        data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)

        self.data = data
        self.label = label

    def __len__(self):

        return self.size

    def __getitem__(self, idx):

        data_point = self.data(idx)
        data_label = self.label(idx)
        return data_point, data_label
