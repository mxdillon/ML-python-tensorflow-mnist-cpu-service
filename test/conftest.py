#!/usr/bin/python
# coding=utf-8

import pytest
import numpy as np

from service import MNIST


@pytest.fixture
def mnist_class():
    return MNIST()


@pytest.fixture
def json_out(mnist_class):
    return mnist_class.format_payload(index=0, y_pred=np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))


@pytest.fixture
def test_image():
    return np.zeros(shape=(28, 28))
