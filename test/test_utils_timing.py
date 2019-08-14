# Author: borgwang <borgwang@126.com>
# Date: 2019-06-20
#
# Filename: test_utils_timing.py
# Description: test unit for utils/timer.py

import runtime_path  # isort:skip
import time

import pytest
from utils.timer import Timer


@pytest.fixture
def test_task_timer():
    return Timer("test_task")


def test_timer_duration(test_task_timer):
    test_task_timer.start()
    time.sleep(2)
    test_task_timer.pause()
    time.sleep(1)
    test_task_timer.start()
    time.sleep(2)
    test_task_timer.stop()
    assert 2 * 2 <= test_task_timer.duration <= 2 * 2 + 0.1


def test_timer_count(test_task_timer):
    test_task_timer.start()
    time.sleep(2)
    test_task_timer.pause()
    time.sleep(1)
    test_task_timer.start()
    time.sleep(2)
    test_task_timer.stop()
    assert test_task_timer.count == 2
