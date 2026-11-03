import numpy as np
import matplotlib.pyplot as plt
from assetlife.datasets import load_circuit_breaker
from assetlife.lifetime_models import Weibull
from assetlife.policies import AgeReplacementPolicy
dataset = load_circuit_breaker()
weibull = Weibull().fit(
    dataset["time"], event=dataset["event"], entry=dataset["entry"]
)
policy = AgeReplacementPolicy(weibull)
ar_star = policy.compute_optimal_ar(cf=11., cp=3., discounting_rate=0.04)
rng = np.random.default_rng(42)
a0 = rng.binomial(60, 0.5, 1000).astype(float)
timeline, replacements = policy.annual_number_of_replacements(
    170, ar=ar_star, a0=a0
)
_, failures = policy.annual_number_of_failures(170, ar=ar_star, a0=a0)
fig, ax = plt.subplots(figsize=(10, 4))
_ = ax.bar(
    timeline + 2025, replacements.sum(axis=1), align="edge", width=1.,
    label="all replacements", color="C1", edgecolor="black", linewidth=0.3,
)
_ = ax.bar(
    timeline + 2025, failures.sum(axis=1), align="edge", width=1.,
    label="failure replacements", color="C0", edgecolor="black", linewidth=0.3,
)
_ = ax.set_xlabel("Year")
_ = ax.set_ylabel("Number of annual replacements")
_ = ax.set_xlim(left=2025, right=2025 + 170)
_ = ax.set_ylim(bottom=0)
_ = ax.legend(loc="upper right")
plt.show()
