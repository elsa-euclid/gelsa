import logging
import math

import numba
import numpy as np
import reproject
import skimage
from astropy.wcs import WCS
from matplotlib import cm, colors
from reproject import mosaicking
from scipy import ndimage

from . import spec_crop, visu

logger = logging.getLogger(__name__)


def make_wcs(ra, dec, pa=0, rotate=0, pixel_size_arcsec=1, shape=None, width_deg=0.9):
    """ """
    wcs_out = WCS()
    res = pixel_size_arcsec/3600
    if shape is None:
        n = int(width_deg / res)
        shape = (n, n)
        logger.debug(f"{shape=}")

    orientation = np.deg2rad(-pa + 90 + rotate)
    cd_matrix = np.array([[np.cos(orientation), -np.sin(orientation)],
                          [np.sin(orientation), np.cos(orientation)]])
    rot_matrix = np.array([[0, -1], [1, 0]])
    cd_matrix = rot_matrix.dot(cd_matrix)
    cd_matrix *= pixel_size_arcsec / 3600


    wcs_out.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs_out.wcs.crval = (ra, dec)
    wcs_out.wcs.crpix = (shape[0]//2,shape[1]//2)
    wcs_out.wcs.cd = cd_matrix
    wcs_out.array_shape = shape
    return wcs_out


def make_wcs_from_specframe(frame, ra=None, dec=None, pixel_size_arcsec=1, shape=None, width_deg=0.9, rotate=0):
    """ """
    pa = frame.params['PA']
    if ra is None:
        ra = frame.params['RA']
        dec = frame.params['DEC']

    return make_wcs(ra, dec, pa,
                    rotate=rotate,
                    pixel_size_arcsec=pixel_size_arcsec,
                    shape=shape,
                    width_deg=width_deg)


def composite(im_rgb, brightness, mask=None, levels=(20, 99), normalize=True):
    """ """

    im_hsv = skimage.color.rgb2hsv(im_rgb)
    if normalize:
        brightness_norm = visu.normalize_image(brightness, levels=levels)
    else:
        brightness_norm = brightness
    im_hsv[:, :, 2] = brightness_norm
    im_rgb = skimage.color.hsv2rgb(im_hsv)
    if mask is not None:
        valid = mask > 0
        im_rgb[:,:,0] *= valid
        im_rgb[:,:,1] *= valid
        im_rgb[:,:,2] *= valid
    return im_rgb


def mosaic_image(image_list, wcs_out):
    """ """
    image, foot = mosaicking.reproject_and_coadd(
        image_list,
        wcs_out,
        shape_out=wcs_out.array_shape,
        reproject_function=reproject.reproject_interp
    )
    return image, foot




@numba.njit(cache=True)
def histogram_rgb(y, x, weight, bins_y, bins_x, hist):
    """Histogram 2D"""
    n = len(x)
    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    for i in range(n):
        if y[i] < bins_y[0]:
            continue
        if y[i] > bins_y[-1]:
            continue
        if x[i] < bins_x[0]:
            continue
        if x[i] > bins_x[-1]:
            continue
        ind_y = int((y[i] - bins_y[0]) / step_y)
        ind_x = int((x[i] - bins_x[0]) / step_x)
        hist[ind_y, ind_x, 0] += weight[i, 0]
        hist[ind_y, ind_x, 1] += weight[i, 1]
        hist[ind_y, ind_x, 2] += weight[i, 2]


@numba.njit(cache=True, nogil=True)
def _histogram2d_accumulate(y, x, weight0, bins_y, bins_x, hist):
    """Histogram 2D accumulate into a 2D array — serial."""
    n = len(x)

    if n == 0:
        return

    ny = len(bins_y) - 1
    nx = len(bins_x) - 1

    step_y = bins_y[1] - bins_y[0]
    step_x = bins_x[1] - bins_x[0]

    for i in range(n):
        ind_y = math.floor((y[i] - bins_y[0]) / step_y)
        if ind_y < 0 or ind_y >= ny:
            continue
        ind_x = math.floor((x[i] - bins_x[0]) / step_x)
        if ind_x < 0 or ind_x >= nx:
            continue
        hist[ind_y, ind_x] += weight0



def colorize(specframe, image, wcs, fixed_wavelength, wcs_out=None,
             level=99, alpha=2, wave_step=10, wavelength_range=None,
             cmap='rainbow'):
    """ """
    if wcs_out is None:
        wcs_out = wcs

    image_ = image**alpha

    thresh = np.percentile(image_, level)

    xx = np.arange(image.shape[1])
    yy = np.arange(image.shape[0])
    xx, yy = np.meshgrid(xx, yy)

    sel = image_ >= thresh
    xx = xx[sel]
    yy = yy[sel]
    im_sel = image_[sel]

    ra_, dec_ = wcs.all_pix2world(xx, yy, 0)

    wave_start, wave_end = specframe.params['wavelength_range']
    nbins = int((wave_end - wave_start) / wave_step + 1)
    wavelength_trace = np.linspace(*specframe.params['wavelength_range'], nbins)

    if wavelength_range is None:
        wavelength_range = specframe.params['wavelength_range']

    norm = colors.Normalize(wavelength_range[0], wavelength_range[-1])
    smap = cm.ScalarMappable(norm, cmap=cmap)
    rgba = smap.to_rgba(wavelength_trace)

    out_shape = wcs_out.array_shape

    im_rgb = np.zeros((out_shape[0], out_shape[1], 3), dtype=float)
    im_norm = np.zeros(out_shape, dtype=float)

    bin_y = np.arange(out_shape[0]+1)
    bin_x = np.arange(out_shape[1]+1)

    for i in range(len(ra_)):
        flux_weight = im_sel[i]

        x, y = specframe.framecoord.radec_to_fov(
            ra_[i]*np.ones(len(wavelength_trace)),
            dec_[i]*np.ones(len(wavelength_trace)),
            wavelength_trace
        )

        rgba_sel = rgba

        # map to RA, Dec at fixed wavelength
        ra_tmp, dec_tmp = specframe.framecoord.fov_to_radec(x, y, fixed_wavelength)
        x, y = wcs_out.wcs_world2pix(ra_tmp, dec_tmp, 0)

        valid = (x > 0) & (y > 0) & (x < out_shape[1]) & (y < out_shape[0])

        x = x[valid]
        if len(x) == 0:
            continue
        y = y[valid]
        rgba_sel = rgba_sel[valid]

        histogram_rgb(y, x, rgba_sel*flux_weight, bin_y, bin_x, im_rgb)
        _histogram2d_accumulate(y, x, flux_weight, bin_y, bin_x, im_norm)

    valid = im_norm > 0
    im_rgb[valid, 0] = im_rgb[valid, 0] / im_norm[valid]
    im_rgb[valid, 1] = im_rgb[valid, 1] / im_norm[valid]
    im_rgb[valid, 2] = im_rgb[valid, 2] / im_norm[valid]

    return im_rgb


def reproject_specframe(frame_list, wcs_out, fixed_wavelength=15500, width=2000, ndrops=1,
                        subtract_continuum=False,
                        median_filter_size_pix=50):
    """ """
    image_stack = np.zeros(wcs_out.array_shape, dtype='d')
    norm_stack = np.zeros(wcs_out.array_shape, dtype='d')

    x = np.random.uniform(0, 1, 10000)*wcs_out.array_shape[1]
    y = np.random.uniform(0, 1, 10000)*wcs_out.array_shape[0]
    ra_, dec_ = wcs_out.all_pix2world(x, y, 0)

    for frame in frame_list:

        frame.params['apply_median_filter'] = subtract_continuum
        frame.params['median_filter_size_pix'] = median_filter_size_pix

        x, y, detectors = frame.radec_to_pixel(ra_, dec_, fixed_wavelength)
        detector_list = np.unique(detectors)

        for det in detector_list:
            if det < 0:
                continue

            det_sel = detectors == det
            ra_c = np.mean(ra_[det_sel])
            dec_c = np.mean(dec_[det_sel])

            image, mask, var = frame.get_detector(det)
            shape = image.shape
            crop = spec_crop.SpecCrop(
                image, mask, var,
                detector=det,
                bbox=[0, shape[1], 0, shape[0]],
                frame=frame
            )
            try:
                image_out, var_image, norm_image, pix_bins = crop.resample_on_radec(
                    ra_c, dec_c,
                    fixed_wavelength,
                    width=width,
                    wcs=wcs_out,
                    ndrops=ndrops,
                    pixel_shrink=1,
                    use_rel_flux_calib=False,
                    use_abs_flux_calib=False,
                )
            except Exception:
                continue
            valid = norm_image > 0
            spec = image_out * 0
            spec[valid] = image_out[valid]

            image_stack[valid] += image_out[valid]
            norm_stack[valid] += norm_image[valid]

    valid = norm_stack > 0
    image_stack[valid] = image_stack[valid] / norm_stack[valid]

    # valid = mask_stack > 0
    # image_stack = image_stack.astype('f4')
    # image_stack[~valid] = -99

    return image_stack, norm_stack > 0


def spectro_composite(spec_im, im_rgb, wcs_in, wcs_out, levels=(5, 98), power=1,
                      sigma=1, strip_rows=512, dtype=np.float32):
    """Low-memory spectro composite, processed in strips of rows.

    Equivalent to replacing the HSV value channel of the reprojected, smoothed
    `im_rgb` with the normalized `spec_im`: with hue and saturation fixed,
    rgb_new = rgb * v_new / max(rgb) (grey v_new where max(rgb) == 0).

    dtype : np.float32 (values in [0,1]) or np.uint8 (values 0-255).
    """
    ny, nx = wcs_out.array_shape

    # percentile levels over finite pixels, as in visu.normalize_image
    vals = spec_im[np.isfinite(spec_im)]
    low, high = np.percentile(vals, levels, overwrite_input=True)
    del vals
    delta = (high - low) or 1

    rgb_in = np.ascontiguousarray(np.moveaxis(im_rgb, -1, 0))   # (3, ny_in, nx_in)
    halo = int(4 * sigma + 0.5)  # gaussian_filter radius with default truncate=4
    out = np.zeros((ny, nx, 3), dtype=dtype)

    for y0 in range(0, ny, strip_rows):
        y1 = min(y0 + strip_rows, ny)
        a0, a1 = max(y0 - halo, 0), min(y1 + halo, ny)

        # reproject all 3 channels at once onto this strip (+halo)
        proj, _ = reproject.reproject_interp(
            (rgb_in, wcs_in), wcs_out[a0:a1, :], shape_out=(3, a1 - a0, nx))
        proj = proj.astype(np.float32)
        if sigma:
            for c in range(3):
                ndimage.gaussian_filter(proj[c], sigma, output=proj[c])
        nan = np.isnan(proj).any(axis=0)       # NaN footprint, already grown by the blur
        proj[:, nan] = 0
        rgb = np.moveaxis(proj[:, y0 - a0:y1 - a0], 0, -1)       # (h, nx, 3)
        nan = nan[y0 - a0:y1 - a0]

        s = spec_im[y0:y1]
        v = np.nan_to_num((s - low) / delta, nan=0).astype(np.float32)
        np.clip(v, 0, 1, out=v)
        v **= power
        v *= s > 0                                                 # mask

        vmax = rgb.max(axis=-1)
        scale = np.divide(v, vmax, out=np.zeros_like(v), where=vmax > 0)
        strip = rgb * scale[..., None]
        grey = (vmax <= 0) | nan                                  # grey, as rgb2hsv gives S=0
        strip[grey] = v[grey, None]

        if out.dtype == np.uint8:
            out[y0:y1] = strip * np.float32(255.99)   # 255.999999 rounds to 256 in float32
        else:
            out[y0:y1] = strip
    return out
