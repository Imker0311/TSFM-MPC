"""Context-window manager for closed-loop TSFM MPC.

The context handed to the forecaster is always two parts concatenated in time:

    [ protected generated block ][ live rolling tail ]

The protected block is a fixed slice of the open-loop excitation dataset. It is never
pruned: it carries the rich valve->state dynamics that a well-behaved closed loop never
excites again. The tail is what actually happened - measured states, applied valve
commands, measured disturbances - appended one row per control step.

The tail cannot grow forever, so it is pruned. Cutting an arbitrary span out of a state
trajectory splices two unrelated states together and hands the model a step change that
no input caused. So the manager logs every tail index whose RECENT WINDOW sits at the
nominal operating point (window, not instant, so the short-term trajectory matches too,
not just the position), and a prune deletes the span between the two oldest such indices.
Both cut ends are then at the same state, moving the same way, and the join is continuous.

Nothing here knows how many covariates or targets there are, or what they mean: channels
are whatever columns the protected block was built with.
"""

import warnings

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
MAX_CONTEXT = 12_500        # rows, protected + tail; a prune fires at or above this
PRUNE_TO = 11_500           # rows, target length a prune aims for

# Near-nominal match. Distance is RMS over (MATCH_WINDOW rows x target channels) of the
# deviation from NOMINAL divided by the per-channel scale, so TOL is in units of "channel
# spreads": 0.1 means the window sits within a tenth of a channel's own spread of nominal.
MATCH_WINDOW = 12           # rows matched back from a candidate cut point
TOL = 0.10                  # accepted distance for a cut point
TOL_MAX = 0.30              # never splice across a join worse than this
TOL_GROWTH = 1.5            # widening factor applied when no pair is found at TOL

# Prune guards.
MAX_PRUNE_FRACTION = 0.5    # of the current tail, removable in one prune
MIN_PRUNE_SPAN = 50         # rows; a cut smaller than this is not worth a splice
KEEP_RECENT = 200           # rows of tail never eligible as a cut point
# ====================================================================


def window_distances(targets, nominal, scale, window):
    """RMS normalised deviation from nominal over every window of `window` rows.

    Element i is the distance of the window ENDING at row i; the first window-1 are nan.
    Position and short-term trajectory are matched together: a single instant can sit on
    the nominal value while moving through it at speed.
    """
    dev = (((np.asarray(targets, dtype=float) - nominal) / scale) ** 2).mean(axis=1)
    out = np.full(len(dev), np.nan)
    if len(dev) >= window:
        c = np.concatenate([[0.0], np.cumsum(dev)])
        out[window - 1:] = np.sqrt((c[window:] - c[:-window]) / window)
    return out


def select_protected_block(frame, length, target_columns, nominal, scale=None,
                           match_window=MATCH_WINDOW, tol=TOL, tol_max=TOL_MAX,
                           tol_growth=TOL_GROWTH, end_at=None):
    """A `length`-row slice of `frame` that ENDS at a near-nominal window.

    The protected/live boundary is a splice like any other, and it exists from the first
    control step. Choosing where the block ends is the only way to make it continuous: the
    block is fixed, so it cannot be spliced later. `end_at` biases the search towards a
    given row (the latest qualifying end at or before it), otherwise the latest in `frame`.

    Widens the tolerance within `tol_max`, then raises - an excitation dataset that never
    revisits the nominal point cannot supply a continuous seam, and silently returning a
    mismatched block would hide that.
    """
    if length > len(frame):
        raise ValueError(f"asked for a {length}-row block from a {len(frame)}-row frame")
    tgt = frame[list(target_columns)].to_numpy(dtype=float)
    scale = tgt.std(axis=0) if scale is None else np.asarray(scale, dtype=float)
    d = window_distances(tgt, np.asarray(nominal, dtype=float), scale, match_window)
    d[:length - 1] = np.nan                                   # no room for the block
    if end_at is not None:
        d[int(end_at) + 1:] = np.nan

    t = tol
    while True:
        hits = np.flatnonzero(d <= t)
        if hits.size:
            end = int(hits[-1])
            block = frame.iloc[end + 1 - length:end + 1].copy()
            block.attrs["seam_distance"] = float(d[end])
            block.attrs["end_row"] = end
            return block
        if t >= tol_max:
            raise ValueError(
                f"no {match_window}-row window within {tol_max:g} of nominal in this frame - "
                f"the excitation never returns to the operating point the live loop sits at, "
                f"so no protected block can join it continuously (closest {np.nanmin(d):.3f})")
        t = min(t * tol_growth, tol_max)


class ContextManager:
    """Holds the protected block plus the growing live tail, and prunes the tail."""

    def __init__(self, protected_block, target_columns, covariate_columns,
                 nominal=None, scale=None, dt=None, item_id=None,
                 max_context=MAX_CONTEXT, prune_to=PRUNE_TO, match_window=MATCH_WINDOW,
                 tol=TOL, tol_max=TOL_MAX, tol_growth=TOL_GROWTH,
                 max_prune_fraction=MAX_PRUNE_FRACTION, min_prune_span=MIN_PRUNE_SPAN,
                 keep_recent=KEEP_RECENT):
        self.target_columns = list(target_columns)
        self.covariate_columns = list(covariate_columns)
        self.channels = self.target_columns + self.covariate_columns

        missing = [c for c in self.channels if c not in protected_block.columns]
        if missing:
            raise ValueError(f"protected block is missing channels: {missing}")
        self._protected = protected_block[self.channels].to_numpy(dtype=float).copy()
        self._protected.flags.writeable = False

        self.max_context = int(max_context)
        self.prune_to = int(prune_to)
        self.match_window = int(match_window)
        self.tol = float(tol)
        self.tol_max = float(tol_max)
        self.tol_growth = float(tol_growth)
        self.max_prune_fraction = float(max_prune_fraction)
        self.min_prune_span = max(int(min_prune_span), self.match_window)
        self.keep_recent = int(keep_recent)
        if self.prune_to >= self.max_context:
            raise ValueError("prune_to must be below max_context")
        if len(self._protected) > self.prune_to:
            raise ValueError(
                f"protected block ({len(self._protected)} rows) leaves no room below "
                f"prune_to ({self.prune_to}) - it is never pruned")

        tgt = protected_block[self.target_columns].to_numpy(dtype=float)
        self.nominal = self._as_vector(nominal, np.median(tgt, axis=0))
        # Scale defaults to the protected block's own per-channel spread, which is what
        # makes one tolerance meaningful across K, mol/L and m at once.
        self.scale = self._as_vector(scale, tgt.std(axis=0))
        self.scale = np.where(self.scale > 0, self.scale, 1.0)

        self.item_id = item_id if item_id is not None else (
            protected_block["item_id"].iloc[0] if "item_id" in protected_block else "context")
        self.dt = self._infer_dt(protected_block, dt)
        self.start = (protected_block["timestamp"].iloc[0]
                      if "timestamp" in protected_block else pd.Timestamp("2026-01-01"))

        # The protected block's own trailing window is a cut point like any other when it
        # sits at nominal: cutting there joins the block straight onto later live data and
        # keeps the tail one contiguous recent span, instead of stranding the first
        # match_window-1 rows (too early to have a window, so never removable) forever.
        self.head_distance = float(window_distances(
            tgt[-self.match_window:], self.nominal, self.scale, self.match_window)[-1]) \
            if len(tgt) >= self.match_window else np.inf

        self._tail = []             # list of per-row channel vectors
        self._dist = []             # window distance from nominal, per tail row (nan if short)
        self.prune_log = []
        self.skipped_prunes = 0
        self._warned = False        # a skipped prune is re-attempted every step; warn once

    # ------------------------------------------------------------------ setup helpers
    def _as_vector(self, value, default):
        if value is None:
            return np.asarray(default, dtype=float)
        if isinstance(value, dict):
            return np.array([float(value[c]) for c in self.target_columns])
        v = np.asarray(value, dtype=float).ravel()
        if v.size != len(self.target_columns):
            raise ValueError(f"expected {len(self.target_columns)} values, got {v.size}")
        return v

    @staticmethod
    def _infer_dt(protected_block, dt):
        if dt is not None:
            return pd.Timedelta(seconds=float(dt)) if not isinstance(dt, pd.Timedelta) else dt
        if "timestamp" in protected_block and len(protected_block) > 1:
            return protected_block["timestamp"].iloc[1] - protected_block["timestamp"].iloc[0]
        if "dt" in protected_block.attrs:
            return pd.Timedelta(seconds=float(protected_block.attrs["dt"]))
        raise ValueError("cannot infer dt - pass dt=<seconds>")

    # ------------------------------------------------------------------ state
    def __len__(self):
        return len(self._protected) + len(self._tail)

    @property
    def protected_length(self):
        return len(self._protected)

    @property
    def tail_length(self):
        return len(self._tail)

    @property
    def protected_block(self):
        return pd.DataFrame(np.asarray(self._protected), columns=self.channels)

    @property
    def tail(self):
        return (np.array(self._tail, dtype=float) if self._tail
                else np.empty((0, len(self.channels))))

    @property
    def distances(self):
        return np.asarray(self._dist, dtype=float)

    @property
    def seam_mismatch(self):
        """Per-target jump across the protected/live boundary, in channel spreads.

        The block is fixed, so this seam is only ever as good as where the block ends (see
        select_protected_block). A boundary-anchored prune re-forms it against a later live
        row rather than removing it, so it is worth reading after pruning as well as before.
        """
        if not self._tail:
            return np.zeros(len(self.target_columns))
        nt = len(self.target_columns)
        return np.abs(self._tail[0][:nt] - np.asarray(self._protected)[-1, :nt]) / self.scale

    def near_nominal_indices(self, tol=None):
        """Tail indices whose recent window sits within `tol` of nominal."""
        d = self.distances
        if d.size == 0:
            return np.empty(0, dtype=int)
        return np.flatnonzero(d <= (self.tol if tol is None else tol))

    # ------------------------------------------------------------------ growth
    def append(self, step_record):
        """Add one timestep - every covariate and every measured target - to the tail."""
        if isinstance(step_record, pd.Series):
            step_record = step_record.to_dict()
        try:
            row = np.array([float(step_record[c]) for c in self.channels])
        except KeyError as e:
            raise KeyError(f"step_record is missing channel {e}") from None

        self._tail.append(row)
        self._dist.append(self._window_distance(len(self._tail) - 1))
        if len(self) >= self.max_context:
            self.prune()
        return len(self._tail) - 1

    def _window_distance(self, i):
        """RMS normalised deviation from nominal over the window ENDING at tail row i."""
        lo = i + 1 - self.match_window
        if lo < 0:
            return np.nan
        win = np.array(self._tail[lo:i + 1])[:, :len(self.target_columns)]
        return float(window_distances(win, self.nominal, self.scale, self.match_window)[-1])

    # ------------------------------------------------------------------ pruning
    def prune(self):
        """Cut oldest removable spans until the context is back under the cap."""
        events = []
        while len(self) >= self.max_context:
            event = self._prune_once()
            if event is None:
                break
            events.append(event)
        return events

    def _prune_once(self):
        need = len(self) - self.prune_to
        max_span = int(self.max_prune_fraction * len(self._tail))
        if max_span < self.min_prune_span:
            self._skip(
                f"context {len(self)} rows: tail too short to prune "
                f"({len(self._tail)} rows, {self.max_prune_fraction:.0%} of it is below the "
                f"{self.min_prune_span}-row minimum span) - not pruning")
            return None

        tol = self.tol
        pair = self._find_pair(need, max_span, tol)
        # Nothing matched closely enough. Widen within the bound rather than splice a
        # mismatched span; if the bound is reached too, leave the context over its cap.
        while pair is None and tol < self.tol_max:
            tol = min(tol * self.tol_growth, self.tol_max)
            pair = self._find_pair(need, max_span, tol)
        if pair is None:
            self._skip(
                f"context {len(self)} rows (cap {self.max_context}): no near-nominal pair "
                f"spanning {self.min_prune_span}-{max_span} rows even at tol={self.tol_max:g} "
                f"- skipping the prune, context is over its cap")
            return None

        a, b = pair
        # join_mismatch is the discontinuity the splice INTRODUCES: the two ends being glued
        # should hold the same state. join_jump is the step actually left in the context at
        # the seam, which is larger whenever an input happens to move on the row after b -
        # that part is a real, input-driven move and is not an artifact.
        event = dict(
            cut_from=a + 1, cut_to=b, span=b - a, tol=tol,
            dist_before=self._dist_at(a), dist_after=self._dist[b],
            length_before=len(self), tail_before=len(self._tail),
            join_mismatch=np.abs(self._tail[b] - self._row(a)),
            join_jump=np.abs(self._tail[b + 1] - self._row(a)) if b + 1 < len(self._tail)
            else np.zeros(len(self.channels)),
        )

        del self._tail[a + 1:b + 1]
        del self._dist[a + 1:b + 1]
        # Windows straddling the new join were measured on rows that are gone.
        for i in range(a + 1, min(a + self.match_window, len(self._tail))):
            self._dist[i] = self._window_distance(i)

        event["length_after"] = len(self)
        event["tail_after"] = len(self._tail)
        self.prune_log.append(event)
        self._warned = False
        return event

    def _skip(self, message):
        self.skipped_prunes += 1
        if not self._warned:
            warnings.warn(message, stacklevel=4)
            self._warned = True

    def _row(self, i):
        """Tail row i, or the protected block's last row for the virtual index -1."""
        return np.asarray(self._protected)[-1] if i < 0 else self._tail[i]

    def _dist_at(self, i):
        return self.head_distance if i < 0 else self._dist[i]

    def _find_pair(self, need, max_span, tol):
        """Earliest (a, b) of logged near-nominal indices bounding a removable span.

        Earliest a wins, so the OLDEST tail material goes first. For that a the smallest b
        that covers `need` wins, so no more is deleted than has to be; if nothing covers
        `need` within the guards, the largest allowed span for that a is taken and the
        caller prunes again.
        """
        cutoff = len(self._tail) - 1 - self.keep_recent
        near = [i for i in self.near_nominal_indices(tol) if i <= cutoff]
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

    # ------------------------------------------------------------------ output
    def get_context(self):
        """[protected][tail] as one contiguous frame at a single sample interval.

        Timestamps are re-indexed across the whole frame: absolute wall-clock either side
        of the seam is meaningless, an unbroken sample interval is not.
        """
        data = np.vstack([np.asarray(self._protected), self.tail])
        df = pd.DataFrame(data, columns=self.channels)
        df.insert(0, "timestamp", pd.date_range(self.start, periods=len(df), freq=self.dt))
        df.insert(0, "item_id", self.item_id)
        df.attrs["dt"] = self.dt.total_seconds()
        return df
