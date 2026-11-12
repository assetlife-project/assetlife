import matplotlib.pyplot as plt
from assetlife.datasets import load_circuit_breaker
from assetlife.lifetime_models import Weibull
from assetlife.stochastic_processes import RenewalRewardProcess
dataset = load_circuit_breaker()
weibull = Weibull().fit(
    time=dataset["time"], event=dataset["event"], entry=dataset["entry"]
)
process = RenewalRewardProcess(lifetime_model=weibull)
for label, costs in (
    ("run to failure", {"cf": 11.}),
    ("ar = 50", {"cf": 11., "cp": 3., "ar": 50}),
):
    timeline, q = process.expected_equivalent_annual_worth(
        tf=300., nb_steps=301, discounting_rate=0.04, **costs
    )
    q_inf = process.asymptotic_expected_equivalent_annual_worth(
        discounting_rate=0.04, **costs
    )
    line, = plt.plot(timeline, q, label=label)
    _ = plt.axhline(q_inf, color=line.get_color(), linestyle="--")
_ = plt.xlabel("Time")
_ = plt.ylabel("Expected equivalent annual worth")
_ = plt.legend()
plt.show()
