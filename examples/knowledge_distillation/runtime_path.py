# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: linquan <linquan@chuangxin.com>
# Date:   17/7/6
#
# ----------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import sys


def add_base_path(cur_main_path):
    cur_dir = os.path.abspath(os.path.dirname(cur_main_path))
    base_dir = os.path.split(cur_dir)[0]
    if cur_dir not in sys.path:
        sys.path.insert(0, cur_dir)
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)


parent_file = inspect.getfile(sys._getframe(1))
add_base_path(os.path.dirname(parent_file))
