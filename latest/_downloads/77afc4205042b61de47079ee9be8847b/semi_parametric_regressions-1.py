import matplotlib.pyplot as plt
import numpy as np
from assetlife.datasets import load_insulator_string
from assetlife.lifetime_models import SemiParametricProportionalHazard
dataset = load_insulator_string()
covar = [
    (dataset[name] - dataset[name].mean()) / dataset[name].std()
    for name in ("pHCl", "pH2SO4", "HNO3")
]
cox = SemiParametricProportionalHazard(
    time=dataset["time"], covar=covar,
    event=dataset["event"], entry=dataset["entry"],
)
estimation = cox.sf(*(c[:2] for c in covar), se=False)
timeline, sf_values = estimation.timeline, estimation.values
_ = plt.plot(timeline, sf_values[0], label="asset 0")
_ = plt.plot(timeline, sf_values[1], label="asset 1")
_ = plt.xlabel("time")
_ = plt.ylabel("survival probability")
_ = plt.legend()
plt.show()
