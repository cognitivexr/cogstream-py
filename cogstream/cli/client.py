import argparse
import logging

from cogstream.client import Client

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='CogStream Client')
    parser.add_argument('--host', type=str, help='the address to connect to', default='127.0.0.1')
    parser.add_argument('--port', type=int, help='the port to expose the camera feed on (default 5555)', default=5555)

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    address = (args.host, args.port)

    c = Client(address)
    c.open()

    try:
        # TODO: send image stream
        print('test1:', c.request(b'test1'))
        print('test2:', c.request(b'test2'))
    except KeyboardInterrupt:
        pass
    finally:
        c.close()


if __name__ == '__main__':
    main()
