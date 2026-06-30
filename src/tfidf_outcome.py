from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class LognormSmoothTfidf:
    """Custom tf-idf: length-normalized tf passed through log1p, times a
    smooth-idf that omits sklearn's "+1" inside the log numerator.

        tf(t,d)         = f_{t,d} / sum_tau f_{tau,d}
        lognorm-tf(t,d) = log(1 + tf(t,d))
        smooth-idf(t,D) = log(|D| / (1 + df(t))) + 1
        tfidf           = lognorm-tf * smooth-idf

    Differs from TfidfTransformer(sublinear_tf=True) in two ways: tf is
    length-normalized *before* log1p (rather than 1+log(count)), and no
    final L2 row-normalization is applied. IDF is fit on train docs only,
    matching TfidfTransformer's fit/transform discipline (no test leakage).
    """
    def fit(self, counts):
        if sp.issparse(counts):
            n  = counts.shape[0]
            df = np.asarray((counts > 0).sum(axis=0)).ravel()
        else:
            counts = np.asarray(counts, dtype=np.float64)
            n  = len(counts)
            df = (counts > 0).sum(axis=0)
        self.idf_ = np.log(n / (1.0 + df)) + 1.0
        return self

    def transform(self, counts):
        if sp.issparse(counts):
            counts    = counts.astype(np.float64)
            row_sums  = np.asarray(counts.sum(axis=1)).ravel()
            row_scale = np.where(row_sums > 1e-12, 1.0 / row_sums, 0.0)
            tf        = sp.diags(row_scale) @ counts
            tf        = tf.copy(); tf.data = np.log1p(tf.data)
            return tf.multiply(self.idf_)
        counts  = np.asarray(counts, dtype=np.float64)
        doc_len = counts.sum(axis=1, keepdims=True)
        tf      = counts / np.maximum(doc_len, 1e-12)
        return np.log1p(tf) * self.idf_[None, :]

    def fit_transform(self, counts):
        return self.fit(counts).transform(counts)
