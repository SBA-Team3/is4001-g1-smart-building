import pytest

from sbanalytics import data as D
from sbanalytics import health as HL


def pytest_configure(config):
    if not D.HOURLY.exists():
        raise pytest.UsageError("data/cleaned/ is missing — run `python scripts/build_clean_data.py` first.")


@pytest.fixture(scope="session")
def h5():
    return D.load_hourly(5)


@pytest.fixture(scope="session")
def H5():
    return D.load_health(5)


@pytest.fixture(scope="session")
def S5(H5):
    return HL.score_point_days(H5)


@pytest.fixture(scope="session")
def building():
    return D.load_hourly(None)
