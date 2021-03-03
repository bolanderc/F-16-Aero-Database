#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:06:58 2021

@author: christian
"""

import time
import random
import numpy as np
import itertools
import machupX as MX

from multiprocessing import Process, Queue, current_process, freeze_support

#
# Function run by worker processes
#

def worker(input, output):
    my_scene = MX.Scene("F16_input.json")
    for args in iter(input.get, 'STOP'):
        t0 = time.time()
        my_scene.set_aircraft_state(state={"alpha": args[0],
                                           "beta": args[1],
                                           "velocity": 222.5211})
        result = my_scene.solve_forces()
        print(time.time() - t0)
        output.put(result)

#
# Function used to calculate result
#

def calculate(func, args):
    result = func(*args)
    return '%s says that %s%s = %s' % \
        (current_process().name, func.__name__, args, result)

#
# Functions referenced by tasks
#

def mul(a, b):
    time.sleep(0.5*random.random())
    return a * b

def plus(a, b):
    time.sleep(0.5*random.random())
    return a + b

#
#
#

def test():
    NUMBER_OF_PROCESSES = 4
    alpha = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
    beta = np.linspace(-np.deg2rad(30), np.deg2rad(30), 6)
    TASKS1 = list(itertools.product(alpha, beta))

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    # Submit tasks
    for task in TASKS1:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    print('Unordered results:')
    for i in range(len(TASKS1)):
        print('\t', done_queue.get())

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')


if __name__ == '__main__':
    freeze_support()
    test()