# Raman Spectroscopy Preprocessing Pipeline

A configurable Python pipeline for preprocessing Raman spectral data — cosmic ray removal, baseline correction, smoothing, normalization, and wavenumber calibration — with batch processing and stage-by-stage plotting.

## Features

- **Four-stage preprocessing pipeline**, each stage independently enabled/disabled:
  1. **Cosmic ray removal** — Median Filter, Modified Z-score, Hampel Filter, Wavelet Despiking
  2. **Baseline correction** — AsLS, ModPoly, airPLS, Polynomial Fitting, Rubber Band, arPLS
  3. **Smoothing** — Savitzky-Golay, Gaussian Filter, Median Filter, Moving Average, Wavelet Denoising
  4. **Normalization** — Vector (L2), Min-Max Scaling, SNV, Area Normalization, Maximum Peak Normalization
- **Wavenumber calibration** against a silicon reference spectrum (expected peak at 520.7 cm⁻¹), with automatic peak detection and constant-offset correction
- **Batch or single-file processing** — point at one file or an entire folder
- **Stage-by-stage plotting** (raw → cosmic removal → baseline → smoothing → normalization → final), with configurable plot selection modes (`all`, `first`, `first_per_file`, `selected`, `random`, `none`)
- **Calibration reports** printed to console and optionally saved to file
- Optional export of processed spectra to `.txt`

## Requirements
