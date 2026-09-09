import numpy as np

from assetlife.lifetime_models import LifetimeDistribution, ParametricLifetimeModel
from assetlife.sampling._iterables import (
    Kijima1ProcessIterable,
    Kijima2ProcessIterable,
    NonHomogeneousPoissonProcessIterable,
    RenewalProcessIterable,
)
from assetlife.stochastic_processes import (
    Kijima1Process,
    Kijima2Process,
    NonHomogeneousPoissonProcess,
    RenewalProcess,
)


def test_kijima_1_iterable(distribution: LifetimeDistribution):
    kijima_1 = Kijima1Process(distribution, q=0.5)
    nb_samples = 10
    t0 = 0.0
    tf = distribution.ppf(0.95).item()

    iterable = Kijima1ProcessIterable(kijima_1, nb_samples, (t0, tf))
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("id", "timeline"))

    # Check Kijima I property for each sample
    for i in range(nb_samples):
        select_sample = struct_array[struct_array["id"] == i]
        np.testing.assert_almost_equal(
            kijima_1.q * select_sample["entry"][1:], select_sample["virtual_age"][:-1]
        )


def test_kijima_2_iterable(distribution: LifetimeDistribution):
    kijima_2 = Kijima2Process(distribution, q=0.5)
    nb_samples = 10
    t0 = 0.0
    tf = distribution.ppf(0.95).item()

    iterable = Kijima2ProcessIterable(kijima_2, nb_samples, (t0, tf))
    struct_array = np.concatenate(tuple(iterable))
    struct_array = np.sort(struct_array, order=("id", "timeline"))

    # Check Kijima II property for each sample
    for i in range(nb_samples):
        select_sample = struct_array[struct_array["id"] == i]
        np.testing.assert_almost_equal(
            kijima_2.q
            * (
                select_sample["time"][1:]
                - select_sample["entry"][1:]
                + select_sample["virtual_age"][:-1]
            ),
            select_sample["virtual_age"][1:],
        )


class TestNonHomogeneousPoissonProcessIterable:
    def test_nhpp_property(self, distribution: LifetimeDistribution):
        nhpp = NonHomogeneousPoissonProcess(distribution)
        nb_samples = 10
        t0 = 0.0
        tf = distribution.ppf(0.95).item()

        iterable = NonHomogeneousPoissonProcessIterable(nhpp, nb_samples, (t0, tf))
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("id", "timeline"))

        # Check NHPP property for each sample
        for i in range(nb_samples):
            select_sample = struct_array[struct_array["id"] == i]
            np.testing.assert_equal(
                select_sample["time"][:-1], select_sample["entry"][1:]
            )

    def test_age_replacement_property(self, distribution: ParametricLifetimeModel[()]):
        nhpp = NonHomogeneousPoissonProcess(distribution)
        ar = 5

        nb_samples = 10
        t0 = distribution.ppf(0.3).item()
        tf = 3 * distribution.ppf(0.95).item()

        iterable = NonHomogeneousPoissonProcessIterable(
            nhpp, nb_samples, (t0, tf), ar=ar
        )
        struct_array = np.concatenate(tuple(iterable))
        assert (struct_array["time"] <= ar + 1e-5).all()

    def test_broadcasting_with_regression(
        self, regression: ParametricLifetimeModel[()]
    ):
        a0 = 5
        ar = 20 * np.ones((3,))
        nhpp = NonHomogeneousPoissonProcess(regression)
        t0 = 0.0
        tf = regression.ppf(0.95).min()
        nb_samples = 10

        # Assert run with no error
        _ = np.concatenate(
            tuple(
                NonHomogeneousPoissonProcessIterable(
                    nhpp, nb_samples, (t0, tf), a0=a0, ar=ar
                )
            )
        )


class TestRenewalProcessIterable:
    def test_no_entry(self, distribution: LifetimeDistribution):
        rp = RenewalProcess(distribution)
        nb_samples = 10
        t0 = 0.0
        tf = distribution.ppf(0.95).item()

        iterable = RenewalProcessIterable(rp, nb_samples, (t0, tf))
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(struct_array, order=("id", "timeline"))

        assert np.all(struct_array["entry"] == 0)

    def test_with_age_replacement(self, distribution: LifetimeDistribution):
        rp = RenewalProcess(distribution)
        ar = 5

        nb_samples = 10
        t0 = distribution.ppf(0.3).item()
        tf = 3 * distribution.ppf(0.95).item()

        iterable = RenewalProcessIterable(rp, nb_samples, (t0, tf), ar=ar)
        struct_array = np.concatenate(tuple(iterable))
        assert np.all(struct_array["time"] <= ar + 1e-5)

    def test_broadcasting_with_regression(
        self, regression: ParametricLifetimeModel[()]
    ):
        a0 = 5
        ar = 20 * np.ones((3,))
        rp = RenewalProcess(regression)
        t0 = 0.0
        tf = regression.ppf(0.95).min()
        nb_samples = 10
        # Assert run with no error
        _ = np.concatenate(
            tuple(RenewalProcessIterable(rp, nb_samples, (t0, tf), a0=a0, ar=ar))
        )
