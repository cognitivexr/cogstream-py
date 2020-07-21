import argparse
import logging
from cogstream.server import serve


def main():
    parser = argparse.ArgumentParser(description='CogStream Server')
    parser.add_argument('--bind', type=str, help='the address to bind to', default='0.0.0.0')
    parser.add_argument('--port', type=int, help='the port to expose the camera feed on (default 53210)', default=53210)

    logging.basicConfig(level=logging.DEBUG)

    args = parser.parse_args()

    address = (args.bind, args.port)
    serve(address)


if __name__ == '__main__':
    main()
