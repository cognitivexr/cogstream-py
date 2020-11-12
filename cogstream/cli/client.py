import argparse
import logging
import os
import time
from queue import Full, Empty

from PIL import Image

from cogstream.client import Client, ThreadedProcessor

logger = logging.getLogger(__name__)


def stream_directory(client, source):
    for f in os.listdir(source):
        if not f.endswith(('.jpg', '.JPG', '.JPEG', '.jpeg', '.png', '.PNG')):
            continue
        fd = os.path.join(source, f)

        # TODO: send image stream
        then = time.time()

        img: Image.Image = Image.open(fd)
        # for testing the effect on latency if we do filtering on the client:
        # from cogstream.engine.yolo.tflite import letterbox_image_pil
        # img = letterbox_image_pil(img, (416, 416))
        response = client.request(img)

        logger.info('%s: %s (%.2fms)', f, response, ((time.time() - then) * 1000))

    pass


def stream_camera(client, source_description):
    import cv2

    video = cv2.VideoCapture(0)
    processor = ThreadedProcessor(client, threads=1)
    source = processor.source
    sink = processor.sink

    processor.start()
    i = 0
    try:
        while True:
            i += 1

            then = time.time()
            check, frame = video.read()
            logger.info('%010d capture took %.2f ms' % (i, ((time.time() - then) * 1000)))

            cv2.imshow("Capturing", frame)
            logger.debug(frame.shape)
            try:
                source.put_nowait((i, frame))
            except Full:
                logger.warning('queue is full, dropping frame %010d', i)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            try:
                response = sink.get(2)
                logger.info('%010d response: %s', i, response)
            except Empty:
                break

            took = time.time() - then
            logger.info('%010d rtt %.2f (%.0f fps)', i, (took * 1000), 1/took)

    except KeyboardInterrupt:
        pass
    finally:
        video.release()
        print('shutting down processor')
        processor.shutdown(5)


def main():
    parser = argparse.ArgumentParser(description='CogStream Client')
    parser.add_argument('--source', type=str, help='frame source (image, directory, camera, ...)', required=True)
    parser.add_argument('--host', type=str, help='the address to connect to', default='127.0.0.1')
    parser.add_argument('--port', type=int, help='the port to expose the camera feed on (default 53210)', default=53210)
    parser.add_argument('--engine', type=str, help='which engine to use (yolo|mobilenet|...)', default='mobilenet')

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    address = (args.host, args.port)

    c = Client(address, args.engine)
    c.open()

    if not c.handshake:
        print('handshake failed')
        c.close()
        return

    try:
        if args.source.startswith('camera'):
            logger.info('streaming from camera')
            stream_camera(c, args.source)
        elif os.path.isdir(args.source):
            logger.info('reading images from directory %s', args.source)
            stream_directory(c, args.source)
        elif os.path.isfile(args.source):
            logger.info('reading file', args.source)
        else:
            print('unknown source type', args.source)
    except KeyboardInterrupt:
        pass
    finally:
        c.close()


if __name__ == '__main__':
    main()
