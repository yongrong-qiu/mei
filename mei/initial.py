from abc import ABC, abstractmethod

import torch
from torch import Tensor, randn


class InitialGuessCreator(ABC):
    """Implements the interface used to create an initial guess."""

    @abstractmethod
    def __call__(self, *shape) -> Tensor:
        """Creates an initial guess from which to start the MEI optimization process given a shape."""


class RandomNormal(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return self._create_random_tensor(*shape)

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalNullChannel(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, null_channel, null_value=0):
        self.null_channel = null_channel
        self.null_value = null_value

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = self._create_random_tensor(*shape)
        inital[:, self.null_channel, ...] = self.null_value
        return inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class RandomNormalNullChannels(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __init__(self, null_channels, null_values, depth=None):
        """
        Args:
            null_channels: a list of channels
            null_values: a lit of values for the respective channels
            depth: number of frames for the temporal channel in dynamic models; Default: None, for static models
        """
        self.null_channels = null_channels
        self.null_values = null_values
        self.depth = depth

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        inital = self._create_random_tensor(*shape)
        for ii in range(len(self.null_channels)):
            inital[:, self.null_channels[ii], ...] = self.null_values[ii]
        return inital[:, :, :self.depth, ...] if self.depth else inital

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"