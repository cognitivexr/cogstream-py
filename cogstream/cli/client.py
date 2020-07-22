import argparse
import logging
import os
import time

import numpy as np
from PIL import Image

from cogstream.client import Client

logger = logging.getLogger(__name__)


def stream_directory(client, source):
    for f in os.listdir(source):
        if not f.endswith(('.jpg', '.JPG', '.JPEG', '.jpeg', '.png', '.PNG')):
            continue
        fd = os.path.join(source, f)

        # TODO: send image stream
        then = time.time()

        img: Image.Image = Image.open(fd)
        response = client.request(img)

        logger.info('%s: %s (%.2fms)', f, response, ((time.time() - then) * 1000))

    pass


def main():
    parser = argparse.ArgumentParser(description='CogStream Client')
    parser.add_argument('--source', type=str, help='frame source (image, directory, camera, ...)', required=True)
    parser.add_argument('--host', type=str, help='the address to connect to', default='127.0.0.1')
    parser.add_argument('--port', type=int, help='the port to expose the camera feed on (default 53210)', default=53210)

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    address = (args.host, args.port)

    c = Client(address)
    c.open()

    if not c.handshake:
        print('handshake failed')
        c.close()
        return

    try:
        if os.path.isdir(args.source):
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
