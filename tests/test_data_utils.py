# tests/test_data_utils.py
from utils.data_utils import clean_rows

def test_clean_rows_basic():
    raw = ["  hello  world  ", "", "   ", None, "Foo\nBar"]
    cleaned = clean_rows(raw)
    assert cleaned == ["hello world", "Foo Bar"]
