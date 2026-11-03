import matplotlib.pyplot as plt
from assetlife.datasets import load_circuit_breaker
from assetlife.lifetime_models import KaplanMeier
dataset = load_circuit_breaker()
km = KaplanMeier(dataset["time"], event=dataset["event"], entry=dataset["entry"])
_ = km.plot("sf")
_ = plt.xlabel("Time")
_ = plt.ylabel("Estimated survival function")
_ = plt.title("Kaplan-Meier")
plt.show()
