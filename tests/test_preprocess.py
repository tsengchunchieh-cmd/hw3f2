# ======================================================
# tests/test_preprocess.py
# ------------------------------------------------------
# 測試資料清理與預處理模組 (train_baseline.clean_text / load_data)
# 加入動態 sys.path 設定，確保 openspec 可被 import。
# ======================================================

import sys
import os
import pytest

# 讓 Python 找到專案根目錄，確保 openspec 模組可被載入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 從 openspec 模組導入要測試的函式
from openspec.changes.add_spam_classification.scripts.train_baseline import clean_text, load_data


def test_clean_text_basic():
    """測試 clean_text 是否能移除符號並轉成小寫"""
    text = "Hello!!! THIS is a TEST."
    cleaned = clean_text(text)
    assert cleaned == "hello this is a test"


def test_load_data_valid(tmp_path):
    """測試 load_data 是否能正確讀取 CSV 檔案"""
    # 建立暫存 CSV
    csv_content = "label,text\nham,Hello there\nspam,Buy now!"
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(csv_content)

    df = load_data(str(csv_path))
    assert not df.empty
    assert list(df.columns) == ["label", "text"]
    assert df.shape == (2, 2)