CogStream Video Analytics
=========================

Requirements: Python3.5+

Prepare
-------

    make venv
    source .venv/bin/activate

Run Server
----------

    python -m cogstream.cli.server
    
Run server and use Edge TPU if possible:

    cogstream_tpu=True python -m cogstream.cli.server


Run Client
----------

Classify all images in a directory `./images/` with a pre-trained mobilenet model:

    python -m cogstream.cli.client --engine mobilenet  --source ./images/

Pass the `--host <ip>` flag if the server is running over the network.


Setup individual ML capabilities
--------------------------------

### Using a camera as a client source

For camera streaming you need opencv on the client

    pip install opencv-python

### TensorFlow Lite

For the server hosting the ML models install tflite: https://www.tensorflow.org/lite/guide/python.
For example on a Coral Dev Board (aarch64) with Python 3.5 run:

    pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_aarch64.whl


### MXNet

For serving `ferplus` on will need mxnet.

    pip install mxnet

For using the GPU you need CUDA and the mxnet cuda pip package. Make sure the CUDA version matches the CUDA
version in the pip package of, e.g., `mxnet-cu102`:

    cat /usr/local/cuda/version.txt
    # OR
    nvcc --version

    # mxnet
    pip install mxnet-cu<version>
