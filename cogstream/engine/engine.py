from enum import Enum
from typing import Tuple


class StreamType(Enum):
    unknown = 0
    frames = 1
    video = 2


class Colorspace(Enum):
    unknown = 0
    rgb = 1
    greyscale = 2


class Transformation(Enum):
    unknown = 0
    scale = 1
    letterbox = 2


class Engine:
    def shape(self) -> Tuple[int, int]:
        raise NotImplementedError

    def colorspace(self) -> Colorspace:
        raise NotImplementedError

    def transformation(self) -> Transformation:
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def inference(self, frame):
        raise NotImplementedError
