import matplotlib.pyplot as plt
from assetlife.lifetime_models import Weibull
from assetlife.sampling import sample_process
from assetlife.stochastic_processes import RenewalProcess
weibull = Weibull(7, 0.05)
renewal_process = RenewalProcess(weibull)
sample = sample_process(renewal_process, 100, (0, 75), seed=10)
print(sample.events.shape)
# Expected:
## (100, 349)
print(sample.timeline.shape)
# Expected:
## (349,)
fig, ax = plt.subplots(figsize=(8, 6))
cumul_events = sample.events.cumsum(axis=1)
for sample_id in range(10):
    _ = ax.plot(sample.timeline, cumul_events[sample_id], alpha=0.3, color="gray")
mean = cumul_events.mean(axis=0)
std = cumul_events.std(axis=0)
_ = ax.plot(sample.timeline, mean, color="red")
_ = ax.fill_between(sample.timeline, mean - std, mean + std, color="red", alpha=0.2)
plt.show()
