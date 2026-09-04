# Raman Spectroscopy Preprocessing Pipeline
A configurable Python pipeline for preprocessing Raman spectral data — cosmic ray removal, baseline correction, smoothing, normalization, and wavenumber calibration — with batch processing and stage-by-stage plotting.

Features
Four-stage preprocessing pipeline, each stage independently enabled/disabled:
Cosmic ray removal — Median Filter, Modified Z-score, Hampel Filter, Wavelet Despiking
Baseline correction — AsLS, ModPoly, airPLS, Polynomial Fitting, Rubber Band, arPLS
Smoothing — Savitzky-Golay, Gaussian Filter, Median Filter, Moving Average, Wavelet Denoising
Normalization — Vector (L2), Min-Max Scaling, SNV, Area Normalization, Maximum Peak Normalization
Wavenumber calibration against a silicon reference spectrum (expected peak at 520.7 cm⁻¹), with automatic peak detection and constant-offset correction
Batch or single-file processing — point at one file or an entire folder
Stage-by-stage plotting (raw → cosmic removal → baseline → smoothing → normalization → final), with configurable plot selection modes (all, first, first_per_file, selected, random, none)
Calibration reports printed to console and optionally saved to file
Optional export of processed spectra to .txt
Requirements
numpy
pandas
matplotlib
scipy

Wavelet-based methods (Wavelet Despiking, Wavelet Denoising) additionally require:

PyWavelets
Usage

All configuration is done by editing the variables at the top of the script — there is no CLI or config file yet.

Set file_path (single file) or folder_path (batch mode) — leave the other empty.
Set output_folder for processed outputs and plots.
Choose algorithms for each stage (cosmic_removal_tool_name, baseline_correction_tool_name, smoothing_tool_name, normalization_tool_name) and toggle stages on/off with the ENABLE_* flags.
(Optional) Set SILICON_REFERENCE_FILE and ENABLE_WAVENUMBER_CALIBRATION = True to calibrate the x-axis against a silicon reference before processing.
Run:
bash
python Aug_27_experiment.py

Processed plots are saved under <output_folder>/Plots/<filename>/, and a calibration report (if enabled) is saved to <output_folder>/Calibration_Report.txt.

Supported input formats

.txt, .csv, .xls, .xlsx — either as two-column (x, y) files or with a shared x-axis provided separately via x_axis_file_path.

Status

Experimental / work in progress — algorithm parameters are hardcoded per method in dictionaries near the top of the file (COSMIC_PARAMS, BASELINE_PARAMS, SMOOTHING_PARAMS, NORMALIZATION_PARAMS, CALIBRATION_PARAMS) and are meant to be tuned per dataset.

License

See the LICENSE file for details.
