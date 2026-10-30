import matplotlib as mpl
import pytest

mpl.use("Agg")


# The guides call plt.show() so the code reads as in a session; under Agg it only warns,
# and pyproject's filterwarnings = ["error"] would turn that into a failure.
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(
            pytest.mark.filterwarnings(
                "ignore:FigureCanvasAgg is non-interactive:UserWarning"
            )
        )
