import os
import unittest

from PIL import Image

from cogstream.engine import load_engine
from cogstream.engine.mobilenet.engine import MobileNetEngine


class MobileNetEngineTestCase(unittest.TestCase):
    def test_inference(self):
        engine: MobileNetEngine = load_engine('mobilenet')

        engine.setup()

        img = Image.open(os.path.join(os.path.dirname(__file__), 'test.jpg'))
        pred_classes = engine.inference(img)

        for i, score in pred_classes:
            print(engine.classes[i])


if __name__ == '__main__':
    unittest.main()
