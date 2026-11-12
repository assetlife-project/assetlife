import matplotlib.pyplot as plt
from assetlife.datasets import load_circuit_breaker
from assetlife.lifetime_models import Weibull
from assetlife.stochastic_processes import RenewalProcess
dataset = load_circuit_breaker()
weibull = Weibull().fit(
    time=dataset["time"], event=dataset["event"], entry=dataset["entry"]
)
renewal_process = RenewalProcess(lifetime_model=weibull)
timeline, m = renewal_process.renewal_function(tf=200., nb_steps=201)
_, m_ar = renewal_process.renewal_function(tf=200., nb_steps=201, ar=58.8)
_, failures_ar = renewal_process.expected_number_of_events(tf=200., nb_steps=201, ar=58.8)
_ = plt.plot(timeline, m, label="run to failure: all renewals")
_ = plt.plot(timeline, m_ar, label="ar = 58.8: all renewals")
_ = plt.plot(timeline, failures_ar, label="ar = 58.8: failures only")
_ = plt.plot(
    timeline, timeline / weibull.mean(), "k--", label="t / E[X]"
)
_ = plt.xlabel("Time")
_ = plt.ylabel("Expected number of renewals")
_ = plt.legend()
plt.show()
