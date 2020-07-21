from typing import Tuple

from cogstream.engine import Engine, Colorspace, Transformation


class YoloEngine(Engine):
    def shape(self) -> Tuple[int, int]:
        return 416, 416

    def transformation(self) -> Transformation:
        return Transformation.letterbox

    def colorspace(self) -> Colorspace:
        return Colorspace.rgb
