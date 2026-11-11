# Release notes

## 4.2

### Calibration
- Default absolute flux calibration updated to `EUC_SIR_W-AbsoluteFlux-SCALE_ALL_20250725T060421.817237Z.fits`.
- Updated the calibration directories searched on Datalabs.

### Euclid archive
- MER segmentation maps are queried with the same `processing_mode` and `purpose` as the mosaic.

### Decontamination
- Sources in the outer region of a MER tile are taken from the tile where they have a valid segmentation map.

### Plotting
- `reproject_specframe` uses less memory.

## 4.1

### Euclid archive
- New survey options `wide_mode` and `deep_mode` for MER catalogues in the calibration fields (`cosmos` maps to `deep_mode`).
- Unrecognised survey names now raise `ValueError` instead of defaulting to `deep`.
- MER product queries accept `processing_mode` and `purpose` filters.

### Line Map Studio
- Fixed loading a target, which failed in 4.0.
- Line maps can be centred at any observed wavelength; only frames whose grism covers it are used.
- Wavelength step buttons (±0.1 nm) and an option to label the line map with its wavelength.
- gelsa and astroquery log messages are shown in the log box.

### Plotting
- `spectro_composite` uses much less memory.
- `plot2d` and `plot2d_target` take a `figsize` argument (`plot2d_target` no longer takes `fig`).
