import pytest
from streamlit.testing.v1 import AppTest

from src import config as C
from tests.conftest import results

APP = str(C.ROOT / "dashboard" / "app.py")


@results
def test_dashboard_starts_without_error():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 4


@results
@pytest.mark.parametrize("page", range(1, 13))
def test_every_dashboard_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=180).run()
    options = at.sidebar.radio[0].options
    at.sidebar.radio[0].set_value(options[page]).run()
    assert not at.exception, f"{options[page]}: {at.exception}"
    assert not at.error, f"{options[page]}: {[e.value for e in at.error]}"


@results
@pytest.mark.parametrize("page", [5, 6])
def test_experiment_pages_never_print_nan(page):
    """Undefined metrics (e.g. silhouette with one cluster) read as words."""
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[page]).run()
    text = " ".join(m.value for m in at.markdown)
    assert "Stage 2" in text
    assert "nan" not in text.lower().split()


@results
def test_deep_link_opens_the_requested_section():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["page"] = "comparison"
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.radio[0].index == 9
