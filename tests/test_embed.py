"""Embedding pipeline logic without model weights."""

import numpy as np

from pokertell.behavior.embed import EMBED_DIM, embed_columns, pool_stream


def test_embed_columns_shape():
    cols = embed_columns()
    assert len(cols) == 5 + 4 * EMBED_DIM
    assert cols[5] == "e_face_mean_0"
    assert cols[-1] == f"e_pose_std_{EMBED_DIM - 1}"


def test_pool_stream_stats_and_nan_guard():
    embs = np.vstack([np.zeros(EMBED_DIM), np.ones(EMBED_DIM) * 2] * 3)
    pooled = pool_stream(embs)
    assert pooled.shape == (2 * EMBED_DIM,)
    assert np.allclose(pooled[:EMBED_DIM], 1.0)
    assert np.allclose(pooled[EMBED_DIM:], 1.0)
    assert np.isnan(pool_stream(embs[:2])).all()
