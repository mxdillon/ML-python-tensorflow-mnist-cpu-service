# ML-python-tensorflow-mnist-cpu-service
Quickstart project for executing a MNIST classifier using TensorFlow on a CPU.

Running `pip install requirements.txt` and then `python app.py` will start the app on localhost where the user can send
GET requests to perform inference.

eg `.../mnist/{index}` where `index` is in the range [0:9999]

__NOTE:__ this quickstart has an onnx model included. This is a placeholder model trained on MNIST data and is not intended to be used by the user in production. Their own models will overwrite this, once trained.