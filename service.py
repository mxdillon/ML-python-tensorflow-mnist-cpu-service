#!/usr/bin/python

import falcon
import json
import numpy as np
import onnx
import time
import tensorflow as tf
import onnxruntime as rt

PORT_NUMBER = 8080
start = time.time()


# API Handler for MNIST classifier

class MNIST(object):
    def __init__(self):
        """Instantiate the class. Load the MNIST test dataset (10000 images).

        :param object: [description]
        :type object: [type]
        """

        mnist = tf.keras.datasets.mnist
        (_, _), (x_test, self.y_test) = mnist.load_data()
        self.x_test = x_test / 255.0
        self.image_count = x_test.shape[0]
        end = time.time()
        print("Loading time: {0:f} secs)".format(end - start))

        # Load the ONNX model
        model = onnx.load('model.onnx')

        # Check that the IR is well formed
        onnx.checker.check_model(model)

        # Start inference session
        rt.set_default_logger_severity(0)
        self.sess = rt.InferenceSession("model.onnx")
        self.input_name = self.sess.get_inputs()[0].name

    @staticmethod
    def prepare_x_test(image_in: np.ndarray) -> np.ndarray:
        """Format an MNIST image so that it can be used for inference in onnx runtime.

        :param image_in: 2-dim numpy array that will be converted into a 4-dim array
        :type image_in: np.ndarray
        :return: 4-dim array with the first (onnxruntime specific) and last dimensions (batchsize=1) as empty
        :rtype: np.ndarray
        """
        test_image = np.asarray(image_in, dtype='float32')
        test_image = np.expand_dims(test_image, axis=2)
        return np.expand_dims(test_image, axis=0)

    def on_get(self, req, resp, index):
        if index < self.image_count:
            payload = {}
            payload["label"] = int(self.y_test[index])
            test_image = self.prepare_x_test(image_in=self.x_test[index, :, :])
            y_pred = self.sess.run(None, {self.input_name: test_image})[0]
            predicted = int(np.argmax(y_pred))
            payload["predicted"] = predicted
            resp.body = json.dumps(payload)
            resp.status = falcon.HTTP_200
        else:
            raise falcon.HTTPBadRequest(
                "Index Out of Range. ",
                "The requested index must be between 0 and {:d}, inclusive.".format(
                    self.image_count - 1
                ),
            )


# API Handler for API example message
class Intro(object):
    def on_get(self, req, resp):
        resp.body = '{"message": \
                    "This service verifies a model using the MNIST Test data set. Invoke using the form \
                    /mnist/<index of test image>. For example, /mnist/24"}'
        resp.status = falcon.HTTP_200
