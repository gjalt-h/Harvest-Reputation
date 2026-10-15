"""Locate unmodified SocialJax on disk and add it to ``sys.path``."""
from __future__ import annotations

import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_ROOTS = (
    _CODE_ROOT / "foundation_code" / "SocialJax-main",
    _CODE_ROOT / "foundation_code" / "SocialJax-main" / "SocialJax-main",
    _CODE_ROOT / "socialjax_clean_reference" / "SocialJax-main",
)


def ensure_socialjax_on_path() -> Path:
    """Prepend the SocialJax repo root to ``sys.path`` if ``import socialjax`` fails."""
    try:
        import socialjax  # noqa: F401

        return Path(socialjax.__file__).resolve().parent.parent
    except ImportError:
        pass

    for root in _CANDIDATE_ROOTS:
        if (root / "socialjax").is_dir():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return root

    raise ImportError(
        "Could not import socialjax. Install SocialJax or place a clean copy under "
        f"one of: {', '.join(str(p) for p in _CANDIDATE_ROOTS)}"
    )
