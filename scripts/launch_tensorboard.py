import sys
import types
import os

# Mock imghdr module for Python 3.13 compatibility
# TensorBoard depends on imghdr which was removed in Python 3.13
if sys.version_info >= (3, 13):
    if 'imghdr' not in sys.modules:
        imghdr_mock = types.ModuleType('imghdr')
        imghdr_mock.what = lambda filename, h=None: None
        # Mock tests list which tensorboard appends to
        imghdr_mock.tests = []
        sys.modules['imghdr'] = imghdr_mock

from tensorboard import main as tb_main

if __name__ == '__main__':
    sys.exit(tb_main.run_main())
