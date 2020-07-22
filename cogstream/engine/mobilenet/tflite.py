import logging
import os
import time
from urllib.request import urlretrieve

import tflite_runtime.interpreter as tflite
from PIL import Image

import cogstream.engine.mobilenet.classify as classify

EDGETPU_SHARED_LIB = "libedgetpu.so.1"

logger = logging.getLogger(__name__)

_root = os.path.join(os.path.dirname(__file__), 'model')

path_model = os.path.join(_root, 'mobilenet_v2_1.0_224_quant.tflite')
path_model_tpu = os.path.join(_root, 'mobilenet_v2_1.0_224_quant_edgetpu.tflite')
path_classes = os.path.join(_root, 'imagenet_labels.txt')


def urlretrieve_ifne(url, filename):
    if not os.path.exists(filename):
        logger.info('downloading %s into %s', url, filename)
        urlretrieve(url, filename)


def require_model():
    if not os.path.exists(_root):
        os.mkdir(_root)

    repo_url = 'https://github.com/google-coral/edgetpu/raw/master/test_data/'

    urlretrieve_ifne('%s/mobilenet_v2_1.0_224_quant.tflite' % repo_url, path_model)
    urlretrieve_ifne('%s/mobilenet_v2_1.0_224_quant_edgetpu.tflite' % repo_url, path_model_tpu)
    urlretrieve_ifne('%s/imagenet_labels.txt' % repo_url, path_classes)


def inference(interpreter, img, n_classes, threshold):
    size = classify.input_size(interpreter)
    image = img.convert('RGB').resize(size, Image.ANTIALIAS)

    classify.set_input(interpreter, image)
    interpreter.allocate_tensors()

    start = time.time()
    interpreter.invoke()
    inf_time = time.time() - start
    print("Net forward-pass time:", inf_time * 1000)

    return classify.get_output(interpreter, n_classes, threshold)


def make_interpreter(use_tpu=False):
    if use_tpu:
        interpreter = tflite.Interpreter(path_model_tpu,
                                         experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
    else:
        interpreter = tflite.Interpreter(path_model)

    return interpreter


def get_classes(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).
    Args:
      path: path to label file.
      encoding: label file encoding.
    Returns:
      Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}
