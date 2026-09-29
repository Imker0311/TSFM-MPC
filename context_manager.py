"""Context-window manager for the closed loop.

The context is a fixed protected block from the excitation dataset followed by a rolling tail
of what actually happened. The tail is pruned by deleting the span between two near-nominal
points, so both cut ends hold the same state and the join introduces no step change that no
input caused.
"""

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
MAX_CONTEXT = 12_500        # rows; a prune fires at or above this
PRUNE_TO = 11_500           # rows; target length after a prune

# Near-nominal match, in units of per-channel spreads.
MATCH_WINDOW = 12           # rows matched back from a candidate cut point
TOL = 0.10                  # accepted distance for a cut point
TOL_MAX = 0.30              # never splice across a join worse than this
TOL_GROWTH = 1.5            # widening factor when no pair is found at TOL

MAX_PRUNE_FRACTION = 0.5    # of the tail, removable in one prune
MIN_PRUNE_SPAN = 50         # rows; smaller cuts are not worth a splice
KEEP_RECENT = 200           # rows of tail never eligible as a cut point
# ====================================================================


def window_distances(targets, nominal, scale, window):
    """RMS normalised deviation from nominal over every window, indexed by its last row."""
    dev = (((np.asarray(targets, dtype=float) - nominal) / scale) ** 2).mean(axis=1)
    out = np.full(len(dev), np.nan)
    if len(dev) >= window:
        c = np.concatenate([[0.0], np.cumsum(dev)])
        out[window - 1:] = np.sqrt((c[window:] - c[:-window]) / window)
    return out


def select_protected_block(frame, length, target_columns, nominal, scale=None,
                           match_window=MATCH_WINDOW, tol=TOL, tol_max=TOL_MAX,
                           tol_growth=TOL_GROWTH):
    """Pick a block ending at a near-nominal window, so it joins the live tail cleanly."""
    tgt = frame[list(target_columns)].to_numpy(dtype=float)
    scale = tgt.std(axis=0) if scale is None else np.asarray(scale, dtype=float)
    d = window_distances(tgt, np.asarray(nominal, dtype=float), scale, match_window)
    d[:length - 1] = np.nan

    t = tol
    while t <= tol_max:
        hits = np.flatnonzero(d <= t)
        if hits.size:
            end = int(hits[-1])
            block = frame.iloc[end + 1 - length:end + 1].copy()
            block.attrs["seam_distance"] = float(d[end])
            block.attrs["end_row"] = end
            return block
        t *= tol_growth
    raise ValueError(f"no window within {tol_max:g} of nominal (closest {np.nanmin(d):.3f})")


class ContextManager:
    """Holds the protected block plus the growing live tail, and prunes the tail."""

    def __init__(self, protected_block, target_columns, covariate_columns,
                 nominal=None, scale=None, dt=None, max_context=MAX_CONTEXT,
                 prune_to=PRUNE_TO, match_window=MATCH_WINDOW, tol=TOL, tol_max=TOL_MAX,
                 tol_growth=TOL_GROWTH, max_prune_fraction=MAX_PRUNE_FRACTION,
                 min_prune_span=MIN_PRUNE_SPAN, keep_recent=KEEP_RECENT):
        self.target_columns = list(target_columns)
        self.channels = self.target_columns + list(covariate_columns)
        self._protected = protected_block[self.channels].to_numpy(dtype=float)

        self.max_context, self.prune_to = int(max_context), int(prune_to)
        self.match_window, self.tol = int(match_window), float(tol)
        self.tol_max, self.tol_growth = float(tol_max), float(tol_growth)
        self.max_prune_fraction = float(max_prune_fraction)
        self.min_prune_span = max(int(min_prune_span), self.match_window)
        self.keep_recent = int(keep_recent)

        tgt = protected_block[self.target_columns].to_numpy(dtype=float)
        self.nominal = np.asarray(np.median(tgt, axis=0) if nominal is None else nominal, dtype=float)
        self.scale = np.asarray(tgt.std(axis=0) if scale is None else scale, dtype=float)
        self.dt = pd.Timedelta(seconds=float(dt))
        self.item_id = protected_block["item_id"].iloc[0]
        self.start = protected_block["timestamp"].iloc[0]

        # The block's own trailing window is a cut point too when it sits at nominal.
        self.head_distance = float(window_distances(
            tgt[-self.match_window:], self.nominal, self.scale, self.match_window)[-1])

        self._tail = []
        self._dist = []
        self.prune_log = []

    def __len__(self):
        return len(self._protected) + len(self._tail)

    @property
    def tail(self):
        return np.array(self._tail, dtype=float) if self._tail else np.empty((0, len(self.channels)))

    def append(self, step_record):
        """Append one timestep of covariates and measured targets to the tail."""
        self._tail.append(np.array([float(step_record[c]) for c in self.channels]))
        self._dist.append(self._window_distance(len(self._tail) - 1))
        if len(self) >= self.max_context:
            self.prune()

    def _window_distance(self, i):
        """Distance from nominal of the window ending at tail row i."""
        lo = i + 1 - self.match_window
        if lo < 0:
            return np.nan
        win = np.array(self._tail[lo:i + 1])[:, :len(self.target_columns)]
        return float(window_distances(win, self.nominal, self.scale, self.match_window)[-1])

    def prune(self):
        """Cut oldest removable spans until the context is back under the cap."""
        while len(self) >= self.max_context:
            if self._prune_once() is None:
                break

    def _prune_once(self):
        need = len(self) - self.prune_to
        max_span = int(self.max_prune_fraction * len(self._tail))
        if max_span < self.min_prune_span:
            return None

        # Widen within the bound rather than splice a mismatched span.
        tol = self.tol
        pair = self._find_pair(need, max_span, tol)
        while pair is None and tol < self.tol_max:
            tol = min(tol * self.tol_growth, self.tol_max)
            pair = self._find_pair(need, max_span, tol)
        if pair is None:
            return None

        a, b = pair
        head = np.asarray(self._protected)[-1] if a < 0 else self._tail[a]
        self.prune_log.append(dict(cut_from=a + 1, cut_to=b, span=b - a, tol=tol,
                                   join_mismatch=np.abs(self._tail[b] - head)))

        del self._tail[a + 1:b + 1]
        del self._dist[a + 1:b + 1]
        # Windows straddling the new join were measured on rows that are gone.
        for i in range(a + 1, min(a + self.match_window, len(self._tail))):
            self._dist[i] = self._window_distance(i)
        return self.prune_log[-1]

    def _find_pair(self, need, max_span, tol):
        """Earliest near-nominal pair bounding a removable span; oldest material goes first."""
        cutoff = len(self._tail) - 1 - self.keep_recent
        near = [i for i in np.flatnonzero(np.asarray(self._dist, dtype=float) <= tol)
                if i <= cutoff]
        if self.head_distance <= tol:
            near.insert(0, -1)          # the protected/live boundary
        for j, a in enumerate(near):
            fallback = None
            for b in near[j + 1:]:
                span = b - a
                if span > max_span:
                    break
                if span < self.min_prune_span:
                    continue
                if span >= need:
                    return a, b
                fallback = b
            if fallback is not None:
                return a, fallback
        return None

    def get_context(self):
        """Protected block plus tail as one frame, re-indexed at a single sample interval."""
        data = np.vstack([np.asarray(self._protected), self.tail])
        df = pd.DataFrame(data, columns=self.channels)
        df.insert(0, "timestamp", pd.date_range(self.start, periods=len(df), freq=self.dt))
        df.insert(0, "item_id", self.item_id)
        return df
