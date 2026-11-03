import matplotlib.pyplot as plt
from assetlife.datasets import load_circuit_breaker
from assetlife.lifetime_models import NelsonAalen
dataset = load_circuit_breaker()
na = NelsonAalen(dataset["time"], event=dataset["event"], entry=dataset["entry"])
_ = na.plot("chf")
_ = plt.xlabel("Time")
_ = plt.ylabel("Estimated cumulative hazard")
_ = plt.title("Nelson-Aalen")
plt.show()
