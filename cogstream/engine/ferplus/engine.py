import json
import os
from typing import Tuple

from cogstream.engine import Engine, Transformation, Colorspace


def _load(model_dir=None):
    from .model import FERPlusService, Context

    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), 'model')

    with open(os.path.join(model_dir, 'MAR-INF/MANIFEST.json'), 'r') as fd:
        manifest = json.load(fd)

    ctx = Context('FERPlus', model_dir, manifest, batch_size=1, gpu=0, mms_version='0.0')
    service = FERPlusService()
    service.initialize(ctx)
    return service


class FerplusEngine(Engine):

    def shape(self) -> Tuple[int, int]:
        return 64, 64

    def colorspace(self) -> Colorspace:
        return Colorspace.greyscale

    def transformation(self) -> Transformation:
        return Transformation.scale

    def __init__(self) -> None:
        super().__init__()
        self._do_inference = None

    def setup(self):
        # TODO: download model automatically
        service = _load()
        ctx = service._context

        def inference(data):
            return service.handle(data, ctx)

        self._do_inference = inference

    def inference(self, frame):
        return self._do_inference(frame)
