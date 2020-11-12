import time
from typing import Tuple

from PIL import Image

from cogstream.engine import Engine, Transformation, Colorspace


class DebugEngine(Engine):

    def setup(self):
        pass

    def inference(self, frame: Image):
        return {'size': frame.size, 'time': time.time()}

    def shape(self) -> Tuple[int, int]:
        return 500, 500

    def transformation(self) -> Transformation:
        return Transformation.letterbox

    def colorspace(self) -> Colorspace:
        return Colorspace.rgb
