from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def envelope(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)
    return np.minimum(1, t * 4) * np.minimum(1, (1 - t) * 4)


def main() -> None:
    base = Path("data/raw")
    calm_dir = base / "calm"
    urg_dir = base / "urgency"
    calm_dir.mkdir(parents=True, exist_ok=True)
    urg_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    rng = np.random.default_rng(42)

    # calm samples
    for i in range(12):
        dur = rng.uniform(3.8, 6.2)
        n = int(sr * dur)
        t = np.arange(n) / sr
        f = rng.uniform(120, 190)
        y = 0.08 * np.sin(2 * np.pi * f * t) + 0.03 * np.sin(2 * np.pi * (f * 2) * t)
        y += 0.01 * rng.normal(size=n)
        y *= envelope(n)
        sf.write(str(calm_dir / f"calm_{i:02d}.wav"), y.astype(np.float32), sr)

    # urgency samples
    for i in range(12):
        dur = rng.uniform(2.5, 5.0)
        n = int(sr * dur)
        t = np.arange(n) / sr
        f = rng.uniform(220, 420)
        jitter = 1 + 0.02 * np.sin(2 * np.pi * 7 * t)
        y = 0.18 * np.sin(2 * np.pi * (f * jitter) * t)
        y += 0.06 * np.sin(2 * np.pi * (f * 2.5) * t)
        y += 0.03 * rng.normal(size=n)
        for _ in range(4):
            c = int(rng.integers(0, max(1, n - 1200)))
            y[c : c + 1200] *= 1.6
        y *= envelope(n)
        sf.write(str(urg_dir / f"urgency_{i:02d}.wav"), y.astype(np.float32), sr)

    count = len(list(base.rglob("*.wav")))
    print(f"[OK] Generated {count} wav files under {base.resolve()}")


if __name__ == "__main__":
    main()
