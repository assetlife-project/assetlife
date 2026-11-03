from assetlife.stochastic_processes import NonHomogeneousPoissonProcess
nhpp = NonHomogeneousPoissonProcess(weibull)
sample = sample_process(nhpp, 100, (0, 100), ar=30, seed=10)
fig, ax = plt.subplots(figsize=(8, 6))
cumul_events = sample.events.cumsum(axis=1)
for sample_id in range(10):
    _ = ax.plot(sample.timeline, cumul_events[sample_id], alpha=0.3, color="gray")
mean = cumul_events.mean(axis=0)
std = cumul_events.std(axis=0)
_ = ax.plot(sample.timeline, mean, color="red")
_ = ax.fill_between(sample.timeline, mean - std, mean + std, color="red", alpha=0.2)
plt.show()
