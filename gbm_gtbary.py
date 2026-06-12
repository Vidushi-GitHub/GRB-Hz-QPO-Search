#!/usr/bin/env python3
"""Barycenter-correct GBM TTE using GBM TTE + POSHIST (via gtbary).

gtbary expects LAT-style FITS layouts. This script:
  1. Builds an FT2-like spacecraft file from glg_poshist_all_*.fit
  2. Repackages the GBM TTE into a gtbary-compatible event file
  3. Runs gtbary to produce a barycenter-corrected TTE

Directory layout (relative to this script):
  ../gbm-data/              input TTE and POSHIST files
  ./                        barycenter-corrected outputs (default)

Usage:
  python gbm_gtbary.py \\
      --tte ../glg_tte_n8_bn220910242_v00.fit \\
      --poshist ../glg_poshist_all_220910_v00.fit

  # or, with defaults for trigger bn220910242 / detector n8:
  python gbm_gtbary.py --detector n8 --trigger bn220910242
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "gbm-data"
OUT_DIR = SCRIPT_DIR


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def default_tte(trigger: str, detector: str) -> Path:
    return DATA_DIR / f"glg_tte_{detector}_{trigger}_v00.fit"


def default_poshist(trigger: str) -> Path:
    # trigger bn220910242 -> date 220910
    date = trigger.replace("bn", "")[:6]
    return DATA_DIR / f"glg_poshist_all_{date}_v00.fit"


def default_outfile(tte_path: Path) -> Path:
    return OUT_DIR / (tte_path.stem + "_b.fit")


def poshist_to_scfile(poshist_path: str, tstart: float, tstop: float, out_path: str) -> None:
    """Convert GBM POSHIST to an FT2 SC_DATA file for gtbary."""
    ph = Table.read(poshist_path, hdu=1)
    hdr1 = fits.getheader(poshist_path, 1)
    hdr0 = fits.getheader(poshist_path, 0)

    met = np.asarray(ph["SCLK_UTC"], dtype=np.float64)
    pos = np.column_stack([ph["POS_X"], ph["POS_Y"], ph["POS_Z"]]).astype(np.float32)

    pad = 5.0
    mask = (met >= tstart - pad) & (met <= tstop + pad)
    if not np.any(mask):
        raise ValueError(
            f"POSHIST does not cover TTE times [{tstart}, {tstop}]. "
            f"POSHIST range is [{met.min()}, {met.max()}]."
        )

    met = met[mask]
    pos = pos[mask]

    start = met.copy()
    stop = np.empty_like(start)
    stop[:-1] = met[1:]
    stop[-1] = met[-1] + 1.0

    sc = Table(data=[start, stop, pos], names=["START", "STOP", "SC_POSITION"])

    primary = fits.PrimaryHDU()
    for key in ("TELESCOP", "INSTRUME", "MJDREFI", "MJDREFF", "TIMESYS", "TIMEUNIT"):
        if key in hdr1:
            primary.header[key] = hdr1[key]
        elif key in hdr0:
            primary.header[key] = hdr0[key]
    primary.header["EXTEND"] = True
    primary.header["FILENAME"] = os.path.basename(out_path)

    sc_hdr = fits.Header()
    sc_hdr["EXTNAME"] = "SC_DATA"
    sc_hdr["TELESCOP"] = primary.header.get("TELESCOP", "GLAST")
    sc_hdr["INSTRUME"] = "LAT"
    for key in ("MJDREFI", "MJDREFF", "TIMESYS", "TIMEUNIT", "TIMEREF"):
        if key in hdr1:
            sc_hdr[key] = hdr1[key]
    sc_hdr["TSTART"] = float(start[0])
    sc_hdr["TSTOP"] = float(stop[-1])
    sc_hdr["TUNIT1"] = "s"
    sc_hdr["TUNIT2"] = "s"
    sc_hdr["TUNIT3"] = "m"

    sc_hdu = fits.BinTableHDU(sc, header=sc_hdr, name="SC_DATA")
    fits.HDUList([primary, sc_hdu]).writeto(out_path, overwrite=True)


def tte_to_gtbary_evfile(tte_path: str, out_path: str) -> tuple[float, float, float, float]:
    """Repackage GBM TTE (EBOUNDS + EVENTS + GTI) for gtbary."""
    with fits.open(tte_path) as hdul:
        events = hdul["EVENTS"].copy()
        src_primary = hdul[0].header

    tstart = float(events.header["TSTART"])
    tstop = float(events.header["TSTOP"])
    ra = float(src_primary.get("RA_OBJ", 0.0))
    dec = float(src_primary.get("DEC_OBJ", 0.0))

    primary = fits.PrimaryHDU()
    primary.header["EXTEND"] = True
    primary.header["TELESCOP"] = src_primary.get("TELESCOP", "GLAST")
    primary.header["INSTRUME"] = "LAT"
    primary.header["OBJECT"] = src_primary.get("OBJECT", "")
    for key in ("MJDREFI", "MJDREFF", "TIMESYS", "TIMEUNIT"):
        if key in events.header:
            primary.header[key] = events.header[key]

    events.header["INSTRUME"] = "LAT"
    events.header["HDUCLAS1"] = "EVENTS"
    events.header["HDUCLAS2"] = "ALL"
    events.header["TIMEREF"] = "LOCAL"

    fits.HDUList([primary, events]).writeto(out_path, overwrite=True)
    return tstart, tstop, ra, dec


def merge_bary_tte(tte_path: str, bary_events_path: str, out_path: str) -> None:
    """Write a GBM-style TTE with barycenter-corrected times."""
    bary_times = Table.read(bary_events_path, hdu=1)["TIME"]
    with fits.open(tte_path) as hdul:
        out = fits.HDUList([h.copy() for h in hdul])
        out["EVENTS"].data["TIME"] = np.asarray(bary_times, dtype=np.float64)

        shift = float(bary_times[0] - hdul["EVENTS"].data["TIME"][0])

        for hdu in out:
            for key in ("TSTART", "TSTOP", "TRIGTIME"):
                if key in hdu.header:
                    hdu.header[key] = hdu.header[key] + shift

        if "GTI" in out:
            out["GTI"].data["START"] = out["GTI"].data["START"] + shift
            out["GTI"].data["STOP"] = out["GTI"].data["STOP"] + shift

        out.writeto(out_path, overwrite=True)


def run_gtbary(
    evfile: str,
    scfile: str,
    outfile: str,
    ra: float,
    dec: float,
    timing_dir: str | None,
) -> None:
    gtbary = shutil.which("gtbary")
    if gtbary is None:
        raise RuntimeError("gtbary not found in PATH (activate the fermi conda env).")

    env = os.environ.copy()
    if timing_dir:
        env["TIMING_DIR"] = timing_dir
    elif "TIMING_DIR" not in env:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidate = os.path.join(
                conda_prefix, "share/fermitools/refdata/fermi/jplephem"
            )
            if os.path.isdir(candidate):
                env["TIMING_DIR"] = candidate

    cmd = [
        gtbary,
        f"evfile={evfile}",
        f"scfile={scfile}",
        f"outfile={outfile}",
        f"ra={ra}",
        f"dec={dec}",
        "clobber=yes",
    ]
    print("Running:", " ".join(cmd))
    if env.get("TIMING_DIR"):
        print("TIMING_DIR =", env["TIMING_DIR"])
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tte", type=Path, default=None, help=f"Input TTE (default: {DATA_DIR}/glg_tte_<det>_<trig>_v00.fit)")
    parser.add_argument("--poshist", type=Path, default=None, help=f"Input POSHIST (default: {DATA_DIR}/glg_poshist_all_<date>_v00.fit)")
    parser.add_argument("--outfile", "-o", type=Path, default=None, help=f"Output TTE (default: {OUT_DIR}/<tte_stem>_b.fit)")
    parser.add_argument("--trigger", default="bn220910242", help="Trigger name for default file paths")
    parser.add_argument("--detector", default="n8", help="Detector id for default TTE path (e.g. n8, nb)")
    parser.add_argument("--ra", type=float, default=9.1, help="Source RA (deg) for barycentric correction")
    parser.add_argument("--dec", type=float, default=-1.1, help="Source Dec (deg) for barycentric correction")
    parser.add_argument("--keep-intermediate", action="store_true", help="Save derived scfile and evfile next to output")
    parser.add_argument("--timing-dir", default=None, help="Directory with JPLEPH.405, leapsec.fits, tai-utc.dat")
    args = parser.parse_args()

    tte = resolve_path(args.tte) if args.tte else default_tte(args.trigger, args.detector)
    poshist = resolve_path(args.poshist) if args.poshist else default_poshist(args.trigger)
    outfile = resolve_path(args.outfile) if args.outfile else default_outfile(tte)

    if not tte.is_file():
        raise FileNotFoundError(f"TTE not found: {tte}")
    if not poshist.is_file():
        raise FileNotFoundError(f"POSHIST not found: {poshist}")

    outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input TTE:    {tte}")
    print(f"Input POSHIST:{poshist}")
    print(f"Output:       {outfile}")

    with tempfile.TemporaryDirectory(prefix="gbm_gtbary_") as tmp:
        ev_prep = os.path.join(tmp, "evfile_gtbary.fits")
        sc_prep = os.path.join(tmp, "scfile_from_poshist.fits")

        tstart, tstop, _, _ = tte_to_gtbary_evfile(str(tte), ev_prep)
        ra = args.ra
        dec = args.dec

        print(f"TTE time range: {tstart} – {tstop}")
        print(f"Using RA={ra}, Dec={dec}")

        poshist_to_scfile(str(poshist), tstart, tstop, sc_prep)

        bary_tmp = os.path.join(tmp, "bary_events.fits")
        run_gtbary(ev_prep, sc_prep, bary_tmp, ra, dec, args.timing_dir)
        merge_bary_tte(str(tte), bary_tmp, str(outfile))

        if args.keep_intermediate:
            base = str(outfile.with_suffix(""))
            shutil.copy2(sc_prep, base + "_scfile_from_poshist.fits")
            shutil.copy2(ev_prep, base + "_evfile_gtbary.fits")
            print("Saved intermediate files:")
            print(" ", base + "_scfile_from_poshist.fits")
            print(" ", base + "_evfile_gtbary.fits")

    print("Wrote", outfile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
