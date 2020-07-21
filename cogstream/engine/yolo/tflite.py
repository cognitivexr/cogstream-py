# RUN WITH:
# python3 inference-simple.py --tpu --model models/quant_coco-tiny-v3-relu_edgetpu.tflite --quant --anchors cfg/tiny_yolo_anchors.txt --classes cfg/coco.names --image ../demo/images/

# model path for TPU = models/quant_coco-tiny-v3-relu_edgetpu.tflite
# model path         = models/quant_coco-tiny-v3-relu.tflite
# classes = cfg/coco.names
# anchors = cfg/tiny_yolo_anchors.txt


import argparse
import os
import time

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

EDGETPU_SHARED_LIB = "libedgetpu.so.1"

_root = os.path.dirname(__file__)
path_model = os.path.join(_root, 'model', 'quant_coco-tiny-v3-relu.tflite')
path_model_tpu = os.path.join(_root, 'model', 'quant_coco-tiny-v3-relu_edgetpu.tflite')
path_classes = os.path.join(_root, 'model', 'coco.names')
path_anchors = os.path.join(_root, 'model', 'tiny_yolo_anchors.txt')


def download():
    # TODO: makedir
    # TODO: download

    # 'https://raw.githubusercontent.com/edgerun/edge-tpu-tiny-yolo/master/cfg/coco.names' -> path_classes
    # 'https://raw.githubusercontent.com/edgerun/edge-tpu-tiny-yolo/master/cfg/tiny_yolo_anchors.txt' -> path_anchors
    # 'https://raw.githubusercontent.com/edgerun/edge-tpu-tiny-yolo/master/models/quant_coco-tiny-v3-relu.tflite' -> path_model
    # 'https://raw.githubusercontent.com/edgerun/edge-tpu-tiny-yolo/master/models/quant_coco-tiny-v3-relu_edgetpu.tflite ' -> path_model
    pass

# Maximum number of boxes. Only the top scoring ones will be considered.
MAX_BOXES = 30


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def letterbox_image_pil(image, size):
    """
    Resize image with unchanged aspect ratio using padding.

    :param image:
    :param size:
    :return:
    """
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def featuresToBoxes(outputs, anchors, n_classes, net_input_shape, img_orig_shape, threshold):
    grid_shape = outputs.shape[1:3]
    n_anchors = len(anchors)

    # Numpy screwaround to get the boxes in reasonable amount of time
    grid_y = np.tile(np.arange(grid_shape[0]).reshape(-1, 1), grid_shape[0]).reshape(1, grid_shape[0], grid_shape[0],
                                                                                     1).astype(np.float32)
    grid_x = grid_y.copy().T.reshape(1, grid_shape[0], grid_shape[1], 1).astype(np.float32)
    outputs = outputs.reshape(1, grid_shape[0], grid_shape[1], n_anchors, -1)
    _anchors = anchors.reshape(1, 1, 3, 2).astype(np.float32)

    # Get box parameters from network output and apply transformations
    bx = (sigmoid(outputs[..., 0]) + grid_x) / grid_shape[0]
    by = (sigmoid(outputs[..., 1]) + grid_y) / grid_shape[1]
    # Should these be inverted?
    bw = np.multiply(_anchors[..., 0] / net_input_shape[1], np.exp(outputs[..., 2]))
    bh = np.multiply(_anchors[..., 1] / net_input_shape[2], np.exp(outputs[..., 3]))

    # Get the scores
    scores = sigmoid(np.expand_dims(outputs[..., 4], -1)) * \
             sigmoid(outputs[..., 5:])
    scores = scores.reshape(-1, n_classes)

    # Reshape boxes and scale back to original image size
    ratio = net_input_shape[2] / img_orig_shape[1]
    letterboxed_height = ratio * img_orig_shape[0]
    scale = net_input_shape[1] / letterboxed_height
    offset = (net_input_shape[1] - letterboxed_height) / 2 / net_input_shape[1]
    bx = bx.flatten()
    by = (by.flatten() - offset) * scale
    bw = bw.flatten()
    bh = bh.flatten() * scale
    half_bw = bw / 2.
    half_bh = bh / 2.

    tl_x = np.multiply(bx - half_bw, img_orig_shape[1])
    tl_y = np.multiply(by - half_bh, img_orig_shape[0])
    br_x = np.multiply(bx + half_bw, img_orig_shape[1])
    br_y = np.multiply(by + half_bh, img_orig_shape[0])

    # Get indices of boxes with score higher than threshold
    indices = np.argwhere(scores >= threshold)
    selected_boxes = []
    selected_scores = []
    for i in indices:
        i = tuple(i)
        selected_boxes.append(((tl_x[i[0]], tl_y[i[0]]), (br_x[i[0]], br_y[i[0]])))
        selected_scores.append(scores[i])

    selected_boxes = np.array(selected_boxes)
    selected_scores = np.array(selected_scores)
    selected_classes = indices[:, 1]

    return selected_boxes, selected_scores, selected_classes


def get_anchors(path):
    anchors_path = os.path.expanduser(path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_classes(path):
    classes_path = os.path.expanduser(path)
    with open(classes_path) as f:
        classes = [line.strip('\n') for line in f.readlines()]
    return classes


def nms_boxes(boxes, scores, classes):
    present_classes = np.unique(classes)

    assert (boxes.shape[0] == scores.shape[0])
    assert (boxes.shape[0] == classes.shape[0])

    # Sort based on score
    indices = np.arange(len(scores))
    scores, sorted_is = (list(l) for l in zip(*sorted(zip(scores, indices), reverse=True)))
    boxes = list(boxes[sorted_is])
    classes = list(classes[sorted_is])

    # Run nms for each class
    i = 0
    while True:
        if len(boxes) == 1 or i >= len(boxes) or i == MAX_BOXES:
            break

        # Get box with highest score
        best_box = boxes[i]
        best_cl = classes[i]

        # Iterate over other boxes
        to_remove = []
        for j in range(i + 1, len(boxes)):
            other_cl = classes[j]
            # if other_cl != best_cl:
            #    continue

            other_box = boxes[j]
            box_iou = iou(best_box, other_box)
            if box_iou > 0.15:
                to_remove.append(j)

        if len(to_remove) == 0:
            break
        else:
            # Remove boxes
            for r in to_remove[::-1]:
                del boxes[r]
                del scores[r]
                del classes[r]
                i += 1

    return boxes[:MAX_BOXES], scores[:MAX_BOXES], classes[:MAX_BOXES]


def iou(box1, box2):
    xi1 = max(box1[0][0], box2[0][0])
    yi1 = max(box1[0][1], box2[0][1])
    xi2 = min(box1[1][0], box2[1][0])
    yi2 = min(box1[1][1], box2[1][1])
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    # Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[1][1] - box1[0][1]) * (box1[1][0] - box1[0][0])
    box2_area = (box2[1][1] - box2[0][1]) * (box2[1][0] - box2[0][0])
    union_area = (box1_area + box2_area) - inter_area
    # compute the IoU
    IoU = inter_area / union_area

    return IoU


def make_interpreter(model_path, use_tpu=False):
    if use_tpu:
        interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
    else:
        interpreter = tflite.Interpreter(model_path)

    return interpreter


# Run YOLO inference on the image, returns detected boxes
def inference(interpreter, img, anchors, n_classes, threshold, quant=True):
    input_details, output_details, net_input_shape = \
        get_interpreter_details(interpreter)

    img_orig_shape = img.size
    # Crop frame to network input shape
    img = letterbox_image_pil(img, (416, 416))
    # Add batch dimension
    img = np.expand_dims(img, 0)

    if not quant:
        # Normalize image from 0 to 1
        img = np.divide(img, 255.).astype(np.float32)

    # Set input tensor
    # tensor_index = interpreter.get_input_details()[0]['index']
    # input_tensor = interpreter.tensor(tensor_index)()[0]
    # input_tensor[:, :] = img
    interpreter.set_tensor(input_details[0]['index'], img)

    start = time.time()

    # Run model
    interpreter.invoke()

    inf_time = time.time() - start
    print("Net forward-pass time:", inf_time * 1000)

    # Retrieve outputs of the network
    out1 = interpreter.get_tensor(output_details[0]['index'])
    out2 = interpreter.get_tensor(output_details[1]['index'])

    # If this is a quantized model, dequantize the outputs
    if quant:
        # Dequantize output
        o1_scale, o1_zero = output_details[0]['quantization']
        out1 = (out1.astype(np.float32) - o1_zero) * o1_scale
        o2_scale, o2_zero = output_details[1]['quantization']
        out2 = (out2.astype(np.float32) - o2_zero) * o2_scale

    # Get boxes from outputs of network
    start = time.time()
    _boxes1, _scores1, _classes1 = featuresToBoxes(out1, anchors[[3, 4, 5]],
                                                   n_classes, net_input_shape, img_orig_shape, threshold)
    _boxes2, _scores2, _classes2 = featuresToBoxes(out2, anchors[[1, 2, 3]],
                                                   n_classes, net_input_shape, img_orig_shape, threshold)
    inf_time = time.time() - start
    print("Box computation time:", inf_time * 1000)

    # This is needed to be able to append nicely when the output layers don't
    # return any boxes
    if _boxes1.shape[0] == 0:
        _boxes1 = np.empty([0, 2, 2])
        _scores1 = np.empty([0, ])
        _classes1 = np.empty([0, ])
    if _boxes2.shape[0] == 0:
        _boxes2 = np.empty([0, 2, 2])
        _scores2 = np.empty([0, ])
        _classes2 = np.empty([0, ])

    boxes = np.append(_boxes1, _boxes2, axis=0)
    scores = np.append(_scores1, _scores2, axis=0)
    classes = np.append(_classes1, _classes2, axis=0)

    if len(boxes) > 0:
        boxes, scores, classes = nms_boxes(boxes, scores, classes)

    return boxes, scores, classes


def get_interpreter_details(interpreter):
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]

    return input_details, output_details, input_shape


def do_inference(interpreter, anchors, stream, classes, threshold):
    """
    Run inference on the image using pill
    :param interpreter:
    :param anchors:
    :param stream:
    :param classes:
    :param threshold:
    :return:
    """
    input_details, output_details, input_shape = get_interpreter_details(interpreter)

    n_classes = len(classes)
    img = Image.open(stream)

    # Run inference, get boxes
    boxes, scores, pred_classes = inference(interpreter, img, anchors, n_classes, threshold)

    print(stream, boxes, scores, pred_classes)


def main():
    anchors = get_anchors(path_anchors)
    classes = get_classes(path_classes)

    # Generate random colors for each detection
    colors = np.random.uniform(30, 255, size=(len(classes), 3))

    interpreter = make_interpreter(path_model, use_tpu=False) # TODO
    interpreter.allocate_tensors()

    image_dir = '~/' # TODO

    for f in os.listdir(image_dir):
        img_path = os.path.join(image_dir, f)

        start_time = time.time()
        do_inference(interpreter, anchors, img_path, classes, threshold=0.25)
        elapsed_ms = (time.time() - start_time) * 1000
        annotate_text = '%-15s took %.1fms' % (f, elapsed_ms)
        print(annotate_text)
