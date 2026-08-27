"""
Selective extraction of the real OASIS-1 cross-sectional archives.
=================================================================

OASIS-1 ships as 12 ``.tar.gz`` discs totalling ~16 GB compressed. A full
extraction is far larger (the ``RAW/`` directory holds up to four separate
un-averaged acquisitions per subject, plus preview GIFs and FreeSurfer
segmentations), and this pipeline needs none of it.

Only the atlas-registered MPRAGE volumes are extracted:

``PROCESSED/MPRAGE/T88_111/*_t88_gfc.{img,hdr}``
    Gain-field-corrected, N4-corrected, averaged across the subject's
    acquisitions, resampled to 1 mm isotropic and registered to the Talairach-88
    atlas. Skull is **present**, so the pipeline's own skull-stripping stage
    (M4) still runs as designed.

``PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.{img,hdr}``
    The same volume with OASIS's own brain mask applied. Extracted so the
    configuration can select it without re-reading 16 GB of gzip, but not the
    default: using it would bypass M4.

Why the T88 volumes rather than ``RAW/`` or ``SUBJ_111/``
--------------------------------------------------------

The Harvard-Oxford atlas is defined in MNI space. The T88 volumes are already
registered to a standard atlas space with an atlas-centred affine, so ROI
localization lands on the correct anatomy. ``RAW/`` volumes are in native
scanner space with an arbitrary origin and would need registration first;
``SUBJ_111/`` is 1 mm but still subject-space.

This script extracts real data only. It generates nothing.

Usage::

    python tools/extract_oasis1.py \\
        --archives "C:/Users/avisy/Downloads/dataset" \\
        --out      "C:/Users/avisy/Downloads/dataset/extracted"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.common.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: Filename fragments to extract. Everything else in the archive is skipped.
WANTED_SUFFIXES: Tuple[str, ...] = (
    "_t88_gfc.img",
    "_t88_gfc.hdr",
    "_t88_masked_gfc.img",
    "_t88_masked_gfc.hdr",
)


def wanted(name: str) -> bool:
    """Return whether an archive member should be extracted."""
    return name.endswith(WANTED_SUFFIXES)


#: Matches an OASIS-1 session directory name exactly, e.g. ``OAS1_0001_MR1``.
#: Anchored on purpose: the *filenames* also begin with ``OAS1_``
#: (``OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.img``), so a simple
#: ``startswith("OAS1_")`` counts every file as an extra subject and inflates
#: the total roughly five-fold.
_SESSION_RE = re.compile(r"^OAS1_\d{4}_MR\d+$")


def _session_ids(member_name: str) -> List[str]:
    """Return the OASIS session IDs appearing as path components."""
    return [p for p in Path(member_name).parts if _SESSION_RE.match(p)]


def _safe_member(member: tarfile.TarInfo, out_dir: Path) -> bool:
    """Reject archive members that would write outside ``out_dir``.

    Replaces the ``filter="data"`` argument, which only exists from Python 3.12.
    An absolute path or a ``..`` traversal in a tar member can otherwise write
    anywhere on the filesystem.
    """
    name = member.name.replace("\\", "/")
    if name.startswith("/") or ".." in Path(name).parts:
        logger.warning("Refusing unsafe archive member: %s", member.name)
        return False
    resolved = (out_dir / name).resolve()
    if not str(resolved).startswith(str(out_dir.resolve())):
        logger.warning("Refusing out-of-tree archive member: %s", member.name)
        return False
    return True


def extract_archive(archive: Path, out_dir: Path) -> Dict[str, object]:
    """Extract the wanted members of one disc archive.

    Uses Python's ``tarfile`` rather than shelling out to ``tar``, so the
    member filter is explicit and the same on every platform.

    Args:
        archive: Path to one ``oasis_cross-sectional_discN.tar.gz``.
        out_dir: Destination root.

    Returns:
        A per-archive summary dict.
    """
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    subjects: set = set()
    total_bytes = 0

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if not wanted(member.name):
                skipped += 1
                continue
            target = out_dir / member.name
            if target.exists() and target.stat().st_size == member.size:
                # Already extracted at the right size; resume cheaply.
                extracted += 1
                total_bytes += member.size
                subjects.update(_session_ids(member.name))
                continue
            # `filter=` is only available from Python 3.12 (and late 3.11
            # patch releases). Fall back when it is absent; the member name is
            # already validated by `_safe_member` either way.
            if not _safe_member(member, out_dir):
                skipped += 1
                continue
            try:
                tar.extract(member, path=out_dir, filter="data")
            except TypeError:
                tar.extract(member, path=out_dir)
            extracted += 1
            total_bytes += member.size
            subjects.update(_session_ids(member.name))

    seconds = time.perf_counter() - started
    summary = {
        "archive": archive.name,
        "extracted_files": extracted,
        "skipped_members": skipped,
        "subjects": sorted(subjects),
        "n_subjects": len(subjects),
        "bytes": total_bytes,
        "seconds": round(seconds, 1),
    }
    logger.info(
        "%s: %d file(s) for %d subject(s), %.1f MB in %.0fs",
        archive.name, extracted, len(subjects), total_bytes / 1e6, seconds,
    )
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Selectively extract real OASIS-1 T88 MPRAGE volumes."
    )
    parser.add_argument("--archives", type=Path, required=True,
                        help="Directory holding the 12 .tar.gz discs.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Destination directory for extracted volumes.")
    parser.add_argument("--discs", type=str, default="",
                        help="Comma-separated disc numbers; default all found.")
    args = parser.parse_args()

    setup_logging()
    archives_dir = Path(args.archives)
    if not archives_dir.exists():
        print(f"REAL OASIS-1 DATA REQUIRED: {archives_dir} does not exist.",
              file=sys.stderr)
        return 2

    archives = sorted(
        archives_dir.glob("oasis_cross-sectional_disc*.tar.gz"),
        key=lambda p: int("".join(c for c in p.stem.split("disc")[-1]
                                  if c.isdigit()) or 0),
    )
    if args.discs:
        keep = {d.strip() for d in args.discs.split(",") if d.strip()}
        archives = [a for a in archives
                    if any(f"disc{d}." in a.name or f"disc{d}.tar" in a.name
                           for d in keep)]

    if not archives:
        print(
            "REAL OASIS-1 DATA REQUIRED: no "
            f"oasis_cross-sectional_disc*.tar.gz found in {archives_dir}.",
            file=sys.stderr,
        )
        return 2

    print(f"Found {len(archives)} OASIS-1 disc archive(s).")
    summaries: List[Dict[str, object]] = []
    all_subjects: set = set()

    for archive in archives:
        print(f"  extracting {archive.name} ...", flush=True)
        summary = extract_archive(archive, Path(args.out))
        summaries.append(summary)
        all_subjects.update(summary["subjects"])

    manifest = {
        "dataset": "OASIS-1",
        "source": "Washington University / OASIS "
                  "(https://sites.wustl.edu/oasisbrains/home/oasis-1/)",
        "is_synthetic": False,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "archives_dir": archives_dir.as_posix(),
        "out_dir": Path(args.out).as_posix(),
        "wanted_suffixes": list(WANTED_SUFFIXES),
        "n_archives": len(archives),
        "n_subjects": len(all_subjects),
        "subjects": sorted(all_subjects),
        "per_archive": summaries,
        "note": "Only PROCESSED/MPRAGE/T88_111 volumes were extracted. RAW "
                "acquisitions, preview GIFs and FSL segmentations were skipped "
                "as this pipeline does not use them.",
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "OASIS1_EXTRACTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nExtracted {len(all_subjects)} subject session(s) to {out_dir}")
    print(f"Manifest: {out_dir / 'OASIS1_EXTRACTION_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
