from gym_simulator.envs.OrderBook import OrderBook
from gym_simulator.envs.OrderBookOracle import OrderBookOracle
import pandas as pd
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

q = deque([3,4,2,5])
q.append(13)
q.append(12)

for i in q:
    print(i)
    q.popleft()
