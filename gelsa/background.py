"""Background estimation and source detection with Gnuastro NoiseChisel."""

import os
import shutil
import subprocess
import tempfile

import numpy as np
from astropy.io import fits

# Path to the astnoisechisel executable.  Override with the GELSA_NOISECHISEL
# environment variable or the ``cmd`` argument of :func:`noisechisel`.
NOISECHISEL_CMD = os.environ.get("GELSA_NOISECHISEL", "astnoisechisel")

# Default NoiseChisel parameters.  Tuples are passed comma separated, booleans
# are flags (True to set, False to leave out).  Sizes are in FITS order (x, y).
DEFAULT_PARAMS = {
    # Run options
    "quiet": True,
    "quietmmap": True,
    "continueaftercheck": True,
    "minmapsize": 2**63 - 1,        # Keep arrays in RAM (from gnuastro.conf).
    "tableformat": "fits-binary",   # Needed if gnuastro.conf is not installed.

    # Tessellation
    "tilesize": (10, 10),           # Regular tile size on dim.s (FITS order).
    "numchannels": (1, 1),          # No. of channels in dim.s (FITS order).
    "remainderfrac": 0.1,           # Fraction of remainder to split last tile.
    "workoverch": True,             # Work (not tile) over channel edges.
    "interpmetric": "manhattan",    # Interpolation metric (radial, manhattan).
    "interpnumngb": 20,             # No. of neighbors to use for interpolation.
    "interponlyblank": True,        # Only interpolate over the blank tiles.
    "ignoreblankintiles": True,     # Ignore input's blank pixels in tiles.
    "largetilesize": (50, 50),      # Sim. to tilesize, but for larger tiles.

    # Detection
    "meanmedqdiff": 0.05,           # Max. mean and median quant diff. per tile.
    "qthresh": 0.3,                 # Quantile threshold on convolved image.
    "outliersclip": (3, 0.2),       # Sigma-clip params for qthresh outliers.
    "outliersigma": 5,              # Multiple of sigma to define outliers.
    "smoothwidth": 3,               # Flat kernel width to smooth interpolated.
    "erode": 3,                     # Number of erosions after thresholding.
    "erodengb": 4,                  # Connectivity in erosion (number of ngbs).
    "noerodequant": 0.99,           # Quantile for no erosion.
    "opening": 1,                   # Depth of opening after erosion.
    "openingngb": 8,                # Connectivity in opening (number of ngbs).
    "minskyfrac": 0.7,              # Min. fraction of undetected area in tile.
    "sigmaclip": (3, 0.2),          # Sigma multiple and, tolerance or number.
    "dthresh": 0,                   # Sigma threshold for Pseudo-detections.
    "dopening": 1,                  # Depth of opening after dthresh.
    "dopeningngb": 4,               # 4 or 8 connectivity for dthresh opening.
    "holengb": 8,                   # 4 or 8 connectivity for filling holes.
    "pseudoconcomp": 4,             # 4 or 8 neighbors for labeling pseudo-dets.
    "snminarea": 10,                # Min. pseudo-detection area for S/N dist.
    "minnumfalse": 20,              # Minimum number for S/N estimation.
    "snquant": 0.99,                # Quantile in pseudo-det. to define true.
    "detgrowquant": 0.9,            # Minimum quant. to expand true detections.
    "detgrowmaxholesize": 100,      # Max. area of holes after growth to fill.
}


def _build_args(params):
    """Convert a parameter dictionary to NoiseChisel command line options."""
    args = []
    for key, val in params.items():
        if val is None or val is False:
            continue
        if val is True:
            args.append(f"--{key}")
        elif isinstance(val, (tuple, list, np.ndarray)):
            args.append(f"--{key}=" + ",".join(str(v) for v in val))
        else:
            args.append(f"--{key}={val}")
    return args


def noisechisel(image, mask=None, cmd=None, **options):
    """Run NoiseChisel on an image array.

    Parameters
    ----------
    image : 2D array
        Input image.  NaN pixels are treated as blank by NoiseChisel.
    mask : 2D bool array, optional
        Pixels where mask is True are set to NaN (blank) before detection.
    cmd : str, optional
        Path to the astnoisechisel executable (default ``NOISECHISEL_CMD``).
    **options
        NoiseChisel options overriding ``DEFAULT_PARAMS``, e.g.
        ``tilesize=(20, 20)`` or ``qthresh=0.4``.  Pass False to drop a flag.

    Returns
    -------
    dict
        Output images keyed by HDU name in lower case: ``input-no-sky``,
        ``detections``, ``sky`` and ``sky_std``.  ``detections`` is 1 on
        detected pixels, 0 on the sky and NaN on blank (masked or NaN) input
        pixels, as is ``input-no-sky``.  ``sky`` and ``sky_std`` are
        interpolated over blank pixels.
    """
    cmd = cmd or NOISECHISEL_CMD
    if shutil.which(cmd) is None:
        raise FileNotFoundError(f"NoiseChisel executable not found: {cmd}")

    image = np.array(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if mask is not None:
        image[np.asarray(mask, dtype=bool)] = np.nan

    params = DEFAULT_PARAMS.copy()
    params.update(options)

    with tempfile.TemporaryDirectory(prefix="gelsa_noisechisel_") as tmpdir:
        inputfile = os.path.join(tmpdir, "input.fits")
        outputfile = os.path.join(tmpdir, "output.fits")
        fits.PrimaryHDU(image).writeto(inputfile)

        args = [cmd, inputfile, "--hdu=0", f"--output={outputfile}"] + _build_args(params)
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"NoiseChisel failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
            )

        out = {}
        with fits.open(outputfile) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    out[hdu.name.lower()] = hdu.data.astype(hdu.data.dtype.newbyteorder("="))
    return out
