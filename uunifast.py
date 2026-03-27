# https://github.com/porya-gohary/real-time-task-generators.git
# This code is adapted from the uunifast implementation in that repository, with modifications to generate constrained task sets.
import random
import math
import pandas as pd
def uunifast(n, u):
    sumU = u
    utilizations = []
    for i in range(1, n):
        nextSumU = sumU * (random.random() ** (1.0 / (n - i)))
        utilizations.append(sumU - nextSumU)
        sumU = nextSumU
    utilizations.append(sumU)
    return utilizations
def generate_constrained_taskset(n: int, target_u: float) -> pd.DataFrame:
    utils = uunifast(n, target_u)
    tasks = []
    for i, u in enumerate(utils):
        period = math.exp(random.uniform(math.log(10), math.log(1000)))
        period = int(round(period))
        period = max(2, period)
        c = u * period
        c = int(round(c))
        c = max(1, c)
        period = max(c, period)
        if random.random() < 0.5:
            d = period
        else:
            d = random.randint(c, period)
        bcet = random.randint(1, c)
        tasks.append({
            'Name': f'T{i}',
            'BCET': bcet,
            'WCET': c,
            'Period': period,
            'Deadline': d
        })
    return pd.DataFrame(tasks)
