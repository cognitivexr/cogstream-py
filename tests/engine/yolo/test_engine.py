import os
import unittest

from PIL import Image

from cogstream.engine import load_engine
from cogstream.engine.yolo.engine import YoloEngine


class YoloEngineTestCase(unittest.TestCase):
    def test_inference(self):
        engine: YoloEngine = load_engine('yolo')

        engine.setup()

        img = Image.open(os.path.join(os.path.dirname(__file__), 'test.jpg'))
        boxes, scores, pred_classes = engine.inference(img)

        print(boxes)
        for i in pred_classes:
            print(i, engine.classes[i])


if __name__ == '__main__':
    unittest.main()
