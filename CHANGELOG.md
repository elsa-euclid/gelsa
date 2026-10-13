# Release notes

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
