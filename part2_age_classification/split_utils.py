"""Shared helpers: balanced clip sampling + speaker-independent split that
keeps every age class present in BOTH train and test."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def balanced_sample(df, label_col="age", cap_per_class=None, seed=42):
    """Sample up to `cap_per_class` clips per class, spreading picks across
    speakers (round-robin) so a few speakers don't dominate. cap=None -> all."""
    if cap_per_class is None:
        return df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    parts = []
    for _, g in df.groupby(label_col):
        if len(g) <= cap_per_class:
            parts.append(g)
            continue
        by_spk = {k: list(rng.permutation(v.index.values)) for k, v in g.groupby("client_id")}
        keys = list(by_spk); rng.shuffle(keys)
        picked = []
        while len(picked) < cap_per_class and by_spk:
            for k in list(keys):
                if k in by_spk and by_spk[k]:
                    picked.append(by_spk[k].pop())
                    if len(picked) >= cap_per_class:
                        break
                if k in by_spk and not by_spk[k]:
                    del by_spk[k]
            keys = [k for k in keys if k in by_spk]
        parts.append(g.loc[picked])
    return __import__("pandas").concat(parts).reset_index(drop=True)


def speaker_independent_split(labels, groups, test_size=0.25, seed=42, tries=500):
    """GroupShuffleSplit (no speaker in both sets). Retries seeds until all
    classes appear in train AND test; falls back to the best split found."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    n_cls = len(np.unique(labels))
    best, best_score = None, -1
    for t in range(tries):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + t)
        tr, te = next(gss.split(labels, groups=groups))
        score = min(len(np.unique(labels[tr])), len(np.unique(labels[te])))
        if score > best_score:
            best, best_score = (tr, te), score
        if score == n_cls:
            break
    if best_score < n_cls:
        print(f"[warn] only {best_score}/{n_cls} classes present in both splits "
              f"(too few speakers for some class) - use --full / more clips.")
    return best
