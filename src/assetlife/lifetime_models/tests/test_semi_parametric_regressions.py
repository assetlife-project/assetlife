import numpy as np
import polars as pl
import pytest

from assetlife.lifetime_models import SemiParametricProportionalHazard


def test_cox_params_eq(insulator_string_data: pl.DataFrame) -> None:
    # From manual experiment and comparison to lifelines results
    insulator_data_cox_params = np.array([5.08787802, -2.98553117, 4.51758019])
    timeline_head = np.array(
        [
            1.1,
            2.6,
            3.0,
            3.1,
            3.2,
            3.3,
            4.0,
            4.6,
            4.7,
            4.9,
            5.1,
            5.3,
            5.4,
            6.0,
            6.1,
            6.4,
            6.5,
            6.6,
            6.9,
        ],
        dtype=np.float64,
    )
    sf_head = np.array(
        [
            [0.99997434, 0.99985195],
            [0.99994812, 0.99970062],
            [0.99992171, 0.99954827],
            [0.99986881, 0.99924318],
            [0.99984205, 0.99908886],
            [0.99981526, 0.99893439],
            [0.99978816, 0.99877815],
            [0.99976071, 0.9986199],
            [0.99973319, 0.99846125],
            [0.99970561, 0.9983023],
            [0.99967792, 0.99814277],
            [0.99959458, 0.99766265],
            [0.9995667, 0.9975021],
            [0.99951059, 0.99717901],
            [0.99948252, 0.99701738],
            [0.99939808, 0.9965314],
            [0.99936975, 0.9963684],
            [0.99931304, 0.99604217],
            [0.99925581, 0.99571303],
        ],
        dtype=np.float64,
    )

    covar_column_names = ["pHCl", "pH2SO4", "HNO3"]
    # without left truncation, as in the lifelines comparison
    re_model = SemiParametricProportionalHazard(
        insulator_string_data.select("time", "event", *covar_column_names),
        covar_column_names,
    )
    covar = [insulator_string_data[name].to_numpy()[:2] for name in covar_column_names]
    sf_assetlife = re_model.sf(*covar, se=False)

    assert re_model.get_params() == pytest.approx(insulator_data_cox_params, rel=1e-3)
    assert sf_assetlife[0][:19] == pytest.approx(timeline_head, rel=1e-3)
    assert np.transpose(sf_assetlife[1][:, :19]) == pytest.approx(sf_head, rel=1e-3)
