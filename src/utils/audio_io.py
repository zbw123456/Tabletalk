from __future__ import annotations

from pathlib import Path
from typing import Optional

import soundfile as sf


def get_audio_metadata(path: Path) -> Optional[dict]:
    """Return basic metadata for an audio file; None if unreadable."""
    try:
        info = sf.info(str(path))
        duration = (info.frames / float(info.samplerate)) if info.samplerate else 0.0
        return {
            "path": str(path),
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration_sec": float(duration),
            "format": info.format,
            "subtype": info.subtype,
        }
    except Exception:
        return None
