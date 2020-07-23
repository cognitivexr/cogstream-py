CogStream Video Analytics
=========================

Requirements: Python3.5+

Prepare
-------

    make venv
    source .venv/bin/activate
    
For the server hosting the ML models install tflite: https://www.tensorflow.org/lite/guide/python.
For example on a Coral Dev Board (aarch64) with Python 3.5 run:
    
    pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp35-cp35m-linux_aarch64.whl

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
