from flax import linen as nn


class Classifier(nn.Module):
    num_hidden: int
    num_outputs: int

    def setup(self):

        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):

        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        return x
