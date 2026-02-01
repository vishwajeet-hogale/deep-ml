import numpy as np
from collections import Counter

def compute_tf_idf(corpus, query):
    N = len(corpus)

    # Document frequency (df)
    df = {}
    for doc in corpus:
        for w in set(doc):
            df[w] = df.get(w, 0) + 1

    # IDF with smoothing
    idf = {w: np.log((N + 1) / (df[w] + 1)) + 1 for w in df}
    default_idf = np.log((N + 1) / 1) + 1  # df=0 case

    # TF
    tf = {}
    for di, doc in enumerate(corpus):
        counts = Counter(doc)
        denom = len(doc) if len(doc) > 0 else 1
        for w, c in counts.items():
            tf[(w, di)] = c / denom

    # Output: rows = documents, cols = query words
    outputs = []
    for di, _doc in enumerate(corpus):
        row = []
        for w in query:
            tf_val = tf.get((w, di), 0.0)
            idf_val = idf.get(w, default_idf)
            row.append(round(tf_val * idf_val, 5))
        outputs.append(row)

    return outputs
