# Author: borgwang <borgwang@126.com>
# Date: 2018-12-10
#
# Filename: __init__.py
# Description: add tinnynn module path

import sys
import os
tinynn_dir = os.path.dirname(__file__)
sys.path.insert(1, tinynn_dir)
