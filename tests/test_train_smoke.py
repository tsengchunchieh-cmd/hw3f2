import os
import shutil
import pandas as pd
from openspec.changes.add_spam_classification.scripts.train_baseline import train


def test_train_smoke(tmp_path):
    # Create a tiny dataframe to exercise training pipeline
    df = pd.DataFrame({
        'label': ['ham', 'spam', 'ham', 'spam'],
        'message': ['hello friend', 'win prize', 'see you', 'claim now']
    })
    df['message_clean'] = df['message'].str.lower()
    df['label_bin'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

    out = tmp_path / 'artifacts'
    model_path, vec_path, metrics_path = train(df, model_type='logreg', out_dir=str(out))

    assert os.path.exists(model_path)
    assert os.path.exists(vec_path)
    assert os.path.exists(metrics_path)
    # Clean up
    shutil.rmtree(str(out))
