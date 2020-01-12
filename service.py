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

# Load the ONNX model
model = onnx.load("model.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

rt.set_default_logger_severity(0)
sess = rt.InferenceSession("model.onnx")


# Load the data to test model with
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
image_count = x_test.shape[0]
end = time.time()
print("Loading time: {0:f} secs)".format(end - start))

# Run inference
input_name = sess.get_inputs()[0].name
# res = sess.run(None, {input_name: ximg})
# prob = res[0]

prediction_input = np.asarray(x_test[0, :, :], dtype='float32')
# if only processing a single example, maintain the batch dimension, else swap axes so they are in order model expects:
if prediction_input.ndim == 2:
    prediction_input = np.expand_dims(prediction_input, axis=2)
# onnxruntime requires an extra dimension at axis=0:
prediction_input = np.expand_dims(prediction_input, axis=0)
y_pred = sess.run(None, {input_name: prediction_input})[0]
# out = np.array2string(np.squeeze(y_test, axis=0))

# API Handler for MNIST test images


class MNIST(object):
    def on_get(self, req, resp, index):
        if index < image_count:
            payload = {}
            payload["label"] = int(labels[index])
            img = images[index].view(1, 784)
            outputs = rep.run(img.numpy())
            predicted = int(np.argmax(outputs))
            payload["predicted"] = predicted
            resp.body = json.dumps(payload)
            resp.status = falcon.HTTP_200
        else:
            raise falcon.HTTPBadRequest(
                "Index Out of Range. ",
                "The requested index must be between 0 and {:d}, inclusive.".format(
                    image_count - 1
                ),
            )


# API Handler for API example message
class Intro(object):
    def on_get(self, req, resp):
        resp.body = '{"message": \
                    "This service verifies a model using the MNIST Test data set. Invoke using the form /mnist/<index of test image>. For example, /mnist/24"}'
        resp.status = falcon.HTTP_200
