"""Helpers for importing the external VITA-Audio codebase."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_vita_audio_importable() -> None:
    try:
        import vita_audio  # noqa: F401
        return
    except ImportError:
        pass

    candidates = []
    env_root = os.environ.get("VITA_AUDIO_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "third_party" / "VITA-Audio",
            repo_root.parent / "vita_audio_rl" / "VITA-Audio",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            sys.path.append(str(candidate))
            glm_voice = candidate / "third_party" / "GLM-4-Voice"
            if glm_voice.exists():
                sys.path.append(str(glm_voice))
            try:
                import vita_audio  # noqa: F401
                return
            except ImportError:
                continue

    raise ImportError(
        "Unable to import `vita_audio`. Set `VITA_AUDIO_ROOT` to the local VITA-Audio checkout "
        "before running WavAlign training."
    )
