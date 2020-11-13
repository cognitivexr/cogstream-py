import os
import unittest

from PIL import Image

from cogstream.engine import load_engine
from cogstream.engine.ferplus.engine import FerplusEngine


class FerplusEngineTestCase(unittest.TestCase):
    def test_inference(self):
        engine: FerplusEngine = load_engine('ferplus')

        engine.setup()

        img = Image.open(os.path.join(os.path.dirname(__file__), 'test.jpg'))
        # warmup
        pred_classes = engine.inference(img)
        print(pred_classes)

        # prediction
        pred_classes = engine.inference(img)
        print(pred_classes)

if __name__ == '__main__':
    unittest.main()
