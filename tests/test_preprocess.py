import os
import tempfile
import pandas as pd
from openspec.changes.add_spam_classification.scripts.train_baseline import clean_text, load_data


def test_clean_text_basic():
    assert clean_text('Hello WORLD!!!') == 'hello world'
    assert clean_text('123 ABC') == '123 abc'
    assert clean_text(None) == ''


def test_load_data(tmp_path):
    # create a small csv mimicking the dataset (no header)
    csv = tmp_path / 'small.csv'
    content = 'ham,Hello there\nspam,Win money now\n'
    csv.write_text(content)

    df = load_data(str(csv))
    assert 'message_clean' in df.columns
    assert 'label_bin' in df.columns
    assert df['label_bin'].tolist() == [0, 1]
