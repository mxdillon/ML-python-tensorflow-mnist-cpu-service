#!/usr/bin/python
# coding=utf-8

import pytest
from service import MNIST, Intro


@pytest.mark.parametrize('route,expected', [("/mnist/0", {'label': 7, 'predicted': 7}),
                                            ("/mnist/333", {'label': 5, 'predicted': 5}),
                                            ("/mnist/9473", {'label': 4, 'predicted': 4}),
                                            ("/mnist/56450", {'description': 'The requested index must be between 0'
                                                              ' and 9999, inclusive.', 'title': 'Index Out of Range. '})])
def test_inference(test_client, route, expected):
    mnist = MNIST()
    test_client.app.add_route("/mnist/{index:int(min=0)}", mnist)
    actual = test_client.simulate_get(route)
    assert actual.json == expected


def test_intro(test_client):
    intro = Intro()
    test_client.app.add_route("/mnist", intro)
    result = test_client.simulate_get("/mnist")
    expected = {"message": "This service verifies a model using the MNIST Test data set. Invoke using the form \
                    /mnist/<index of test image>. For example, /mnist/24"}
    assert result.json == expected
