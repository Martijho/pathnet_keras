import random
import numpy as np


a1 = input('Active from task 1: ')
a1 = int(a1)

a2 = int(input('Active from task 2: '))

layer = [1]*a1 + [0]*(10-a1)
x = 10000000

obs = [0]*4
for _ in range(x):

    hit = 0

    hit = sum(random.sample(layer, a2))

    obs[hit] += 1

for i, o in enumerate(obs):
    print(i, 'modules reused:', o/x)