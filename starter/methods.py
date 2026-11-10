"""TTA methods. Add your own here and register it in METHODS.

A method is a callable: method(x) -> LongTensor of labels in {0..19} for the normalized batch x.
It may update its own state (BN statistics, parameters, memory, ...) as the stream goes on.
"""
import torch
import torch.nn as nn


class Source:
    """No adaptation."""

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x).argmax(1)


METHODS = {"source": Source}
