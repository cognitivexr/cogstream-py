import os
from typing import Tuple

from cogstream.engine import Engine, Colorspace, Transformation

driver = None


class YoloEngine(Engine):

    @property
    def driver(self):
        global driver
        return driver

    def __init__(self, use_tpu=None) -> None:
        super().__init__()
        self._do_inference = None
        self.use_tpu = use_tpu if use_tpu is not None else bool(os.getenv('cogstream_tpu', False))
        self.anchors = None
        self.classes = None

    def setup(self):
        if self._do_inference is not None:
            return

        global driver
        if driver is None:
            import cogstream.engine.yolo.tflite as driver_import
            driver = driver_import
            driver.require_model()

        interpreter = driver.make_interpreter(use_tpu=self.use_tpu)
        interpreter.allocate_tensors()

        anchors = driver.get_anchors(driver.path_anchors)
        classes = driver.get_classes(driver.path_classes)

        self.anchors = anchors
        self.classes = classes

        n_classes = len(classes)

        def do_inference(frame):
            return driver.inference(interpreter, frame, anchors, n_classes, threshold=0.25)

        self._do_inference = do_inference

    def inference(self, frame):
        return self._do_inference(frame)

    def shape(self) -> Tuple[int, int]:
        return 416, 416

    def transformation(self) -> Transformation:
        return Transformation.letterbox

    def colorspace(self) -> Colorspace:
        return Colorspace.rgb
