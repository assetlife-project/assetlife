import numpy as np
import matplotlib.pyplot as plt
from assetlife.datasets import load_circuit_breaker
from assetlife.lifetime_models import KaplanMeier, Weibull, Gamma, Gompertz, LogLogistic
dataset = load_circuit_breaker()
km = KaplanMeier(dataset["time"], event=dataset["event"], entry=dataset["entry"])
_ = km.plot("sf", ci=False, color="black", linestyle="--", label="Kaplan-Meier")
timeline = np.arange(0, 145)
for distribution in (Weibull(), Gamma(), Gompertz(), LogLogistic()):
    fitted = distribution.fit(
        dataset["time"], event=dataset["event"], entry=dataset["entry"]
    )
    _ = fitted.plot("sf", timeline, ci=False, label=type(fitted).__name__)
_ = plt.xlabel("Time")
_ = plt.ylabel("Estimated survival function")
_ = plt.legend()
plt.show()
