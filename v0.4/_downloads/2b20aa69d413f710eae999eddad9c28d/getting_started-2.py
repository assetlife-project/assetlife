import numpy as np
import matplotlib.pyplot as plt
from assetlife.datasets import load_circuit_breaker
from assetlife.lifetime_models import KaplanMeier, Weibull
dataset = load_circuit_breaker()
km = KaplanMeier(
    dataset["time"], event=dataset["event"], entry=dataset["entry"]
)
weibull = Weibull().fit(
    dataset["time"], event=dataset["event"], entry=dataset["entry"]
)
timeline = np.arange(0, 145)
_ = km.plot("sf", label="Kaplan-Meier")
_ = weibull.plot("sf", timeline, label="Weibull")
_ = plt.xlabel("Time")
_ = plt.ylabel("Estimated survival function")
_ = plt.legend()
plt.show()
