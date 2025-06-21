# src/data_utils.py
import gym, d4rl
import numpy as np


def load_dataset(env_name: str, ds_name: str):
    """Return X, y, env exactly as required by XGBModelScaled."""
    env  = gym.make(f"{env_name}-{ds_name}-v2")
    data = env.get_dataset()

    s = data["observations"].astype(np.float32)
    a = data["actions"].astype(np.float32)
    r = data["rewards"].astype(np.float32)
    d = (
        data.get("terminals", np.zeros_like(r, bool)) |
        data.get("timeouts",  np.zeros_like(r, bool))
    ).astype(bool)

    # ------------------------------------------------------------------
    # Trajectory IDs: increment *after* each terminal step  (off‑by‑one fix)
    traj_id = np.zeros_like(d, dtype=np.int32)
    np.cumsum(d[:-1], out=traj_id[1:])
    # ------------------------------------------------------------------

    # Vectorised RTG and timestep
    rtg_raw = np.zeros_like(r, dtype=np.float32)
    tstep   = np.zeros_like(r, dtype=np.float32)

    ends   = np.where(d)[0]           # indices of terminal steps
    starts = np.insert(ends[:-1] + 1, 0, 0)

    for s0, e0 in zip(starts, ends):
        seg          = slice(s0, e0 + 1)
        seg_r        = r[seg]
        rtg_raw[seg] = np.flip(np.cumsum(np.flip(seg_r))).astype(np.float32)
        tstep[seg]   = np.arange(e0 + 1 - s0, dtype=np.float32)

    # Normalise RTG to [0,1]
    ref_min = env.ref_min_score
    ref_max = env.ref_max_score
    rtg     = np.clip((rtg_raw - ref_min) / (ref_max - ref_min), 0.0, 1.0)

    X = np.hstack([s, rtg[:, None], tstep[:, None]]).astype(np.float32)
    y = a

    print(f"[DEBUG] Loaded {X.shape[0]} samples for {env_name}-{ds_name}")
    return X, y, env
