# ======================================================
# tests/test_train_smoke.py
# ------------------------------------------------------
# 測試 train_baseline.train() 是否能運行（smoke test）
# 加入動態 sys.path 設定，確保 openspec 可被 import。
# ======================================================

import sys
import os
import pytest

# 讓 Python 找到專案根目錄，確保 openspec 模組可被載入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 導入 train() 函式
from openspec.changes.add_spam_classification.scripts.train_baseline import train


def test_train_smoke(tmp_path):
    """基本 smoke test：確認 train() 可以執行並產生輸出"""
    # 準備暫存資料路徑與輸出路徑
    dummy_data_path = tmp_path / "dummy.csv"
    dummy_data_path.write_text("label,text\nham,Hello\nspam,Buy now!")
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir()

    # 嘗試訓練
    try:
        train(
            data_path=str(dummy_data_path),
            model="logreg",
            out_dir=str(out_dir)
        )
    except Exception as e:
        pytest.fail(f"train() raised an unexpected exception: {e}")

    # 確認訓練成果有輸出檔
    files = list(out_dir.glob("*\n"))
    assert len(files) > 0, "No output artifacts were generated."