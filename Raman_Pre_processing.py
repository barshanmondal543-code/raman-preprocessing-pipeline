import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d     
from scipy.sparse import diags                     
from scipy.sparse.linalg import spsolve            
from scipy.spatial import ConvexHull                  
from datetime import datetime          

###############################################################################
# Input things
###############################################################################

folder_path = "" #  path to folder containing multiple files for batch processing (leave empty for single file)
file_path   = "" # path to single file (leave empty for batch processing)
output_folder = "" # folder where all processed files will be saved

x_axis_file_path = "" # path to single file that you want to usage as x axis.
SILICON_REFERENCE_FILE = ""  # Path to silicon reference spectrum

###############################################################################
# Preprocessing Algorithm Selection
###############################################################################
# Median Filter | Modified Z-score | Hampel Filter | Wavelet Despiking  # ! cosmic 
# AsLS | ModPoly | airPLS | Polynomial Fitting | Rubber Band | arPLS       # ! baseline
# Savitzky Golay | Gaussian Filter | Median Filter | Moving Average | Wavelet Denoising     # ! smoothing
# Vector Normalization | Min-Max Scaling | SNV | Area Normalization | Maximum Peak Normalization        # ! normalization

cosmic_removal_tool_name       = "Median Filter"            
baseline_correction_tool_name  = "ModPoly"                  
smoothing_tool_name            = "Savitzky Golay"           
normalization_tool_name        = "Min-Max Scaling"          

# Enable / Disable each step ( True    False   )
ENABLE_COSMIC_REMOVAL      = True
ENABLE_BASELINE_CORRECTION = True
ENABLE_SMOOTHING           = True
ENABLE_NORMALIZATION       = True

###############################################################################
# ADVANCED PLOTTING CONFIGURATION
###############################################################################
# Plot Mode
PLOT_MODE = "first"
# ! none | first | first_per_file | selected | random | all

# Plot Saving ( True    False   )
ENABLE_PLOT_SAVING = True  

# Save Normalized TXT File ( True    False   )
SAVE_NORMALIZED_TXT = True

# Plot Display
SHOW_PLOTS = True

# Processing Stage Plots
PLOT_RAW = True     
PLOT_COSMIC = True
PLOT_BASELINE = True
PLOT_SMOOTHING = True 
PLOT_NORMALIZATION = True
PLOT_BASELINE_CORRECTED_NORMAL = True
PLOT_FINAL = True

PLOT_CALIBRATION = True        # * Plot the calibration offset like where it shoukd be and where it is comming.

###############################################################################
# WAVENUMBER CALIBRATION CONFIGURATION
###############################################################################

# Enable / Disable ( True    False   )
ENABLE_WAVENUMBER_CALIBRATION = True

# Calibration Method
CALIBRATION_TOOL = "Constant Offset"
# Constant Offset | Linear Calibration | Polynomial Calibration

# Output
SHOW_CALIBRATION_PLOT = True
SAVE_CALIBRATION_REPORT = True

###############################################################################
# ADVANCED PLOTTING -- parameters
###############################################################################

PLOTTING_PARAMS = {

    "Plot Selection": {
        "Selected_Spectra": [0, 2, 5],
        "Random_Spectra": 5,
    },

    "Plot Style": {
        "Reverse_X_Axis": False,
        "Figure_Size": (10, 6),
        "Figure_DPI": 300,
        "Line_Width": 1.5,
        "Show_Grid": False,
    },

    "Fonts": {
        "Title_Size": 18,
        "Axes_Size": 15,
        "Tick_Size": 12,
        "Legend_Size": 11,
    },

    "Save": {
        "Format": "png",
        "DPI": 300,
    },
}

###############################################################################
# WAVENUMBER CALIBRATION -- parameters
###############################################################################

CALIBRATION_PARAMS = {
    "Constant Offset": {
        "Expected_Silicon_Peak": 520.7,      # cm⁻¹ - theoretical silicon peak
        "Peak_Search_Range": (500, 540),     # Search window (cm⁻¹)
    },

    "Linear Calibration": {
        "Reference_Peaks": [520.7],          # Future: multiple reference peaks
        "Peak_Search_Ranges": [(500, 540)],
    },

    "Polynomial Calibration": {
        "Reference_Peaks": [520.7],          # Future: multiple reference peaks
        "Peak_Search_Ranges": [(500, 540)],
        "Polynomial_Order": 2,
    },
}
###############################################################################
# COSMIC RAY REMOVAL TOOLS -- parameters
###############################################################################

COSMIC_PARAMS = {
    "Median Filter": {
        "Window_size": 5,      # odd_number
        "Threshold": 3,
    },
    "Modified Z-score": {
        "Threshold": 3.5,
        "MAD_constant": 0.6745,
        "Local_Window": 7,     # optional, odd
    },
    "Hampel Filter": {
        "Window_size": 7,      # odd_number
        "Threshold": 3,        # MAD multiplier
    },
    "Wavelet Despiking": {
        "Wavelet_type": "db4",
        "Decomposition_level": 4,
        "Threshold_rule": "universal",
        "Threshold_mode": "soft",
    },
}

###############################################################################
# BASELINE CORRECTION TOOLS -- parameters
###############################################################################

BASELINE_PARAMS = {
    "AsLS": {
        "Lambda": 1e5,          # smoothness
        "P": 0.01,              # asymmetry
        "Max_iterations": 10,
    },
    "ModPoly": {
        "Polynomial_order": 2,
        "Max_iterations": 100,
        "Tolerance": 1e-3,
    },
    "airPLS": {
        "Lambda": 1e5,
        "Max_iterations": 15,
        "Tolerance": 1e-3,
    },
    "Polynomial Fitting": {
        "Polynomial_order": 3,
    },
    "Rubber Band": {
        "Convex_hull": "Default",
        "Interpolation": "Linear",
    },
    "arPLS": {
        "Lambda": 1e5,
        "Ratio": 1e-6,          # convergence
        "Max_iterations": 50,
    },
}

###############################################################################
# SMOOTHING TOOLS -- parameters
###############################################################################

SMOOTHING_PARAMS = {
    "Savitzky Golay": {
        "Window_length": 11,   # odd
        "Polynomial_order": 3,
    },
    "Gaussian Filter": {
        "Sigma": 1.0,
    },
    "Median Filter": {
        "Kernel_size": 5,       # odd
    },
    "Moving Average": {
        "Window_size": 5,
    },
    "Wavelet Denoising": {
        "Wavelet": "db4",
        "Level": 4,
        "Threshold_rule": "universal",
        "Threshold_mode": "soft",
    },
}

###############################################################################
# NORMALIZATION TOOLS -- parameters
###############################################################################

NORMALIZATION_PARAMS = {
    "Vector Normalization": {
        "Norm": "L2",
    },
    "Min-Max Scaling": {
        "Minimum": 0,
        "Maximum": 1,
    },
    "SNV": {
        # uses spectrum mean / spectrum std -- nothing to configure
    },
    "Area Normalization": {
        "Integration_method": "Trapezoidal",
    },
    "Maximum Peak Normalization": {
        # divides by max intensity -- nothing to configure
    },
}


###############################################################################
# WAVENUMBER CALIBRATION -- implementations
###############################################################################

def detect_silicon_peak(x, y, search_range, expected_peak):
    """
    Automatically detect the silicon peak position in a reference spectrum.
    
    Args:
        x: Wavenumber array (cm⁻¹)
        y: Intensity array
        search_range: Tuple (min_wn, max_wn) defining the search window
        expected_peak: Expected silicon peak position (for reference, not used in detection)
    
    Returns:
        detected_peak_position: Wavenumber of the detected peak (cm⁻¹)
    
    Raises:
        ValueError: If no peak is detected or search window is invalid
    """
    min_wn, max_wn = search_range
    
    # Validate search range
    if min_wn >= max_wn:
        raise ValueError(f"Invalid search range: min ({min_wn}) must be less than max ({max_wn})")
    
    # Find indices within search range
    mask = (x >= min_wn) & (x <= max_wn)
    
    if not np.any(mask):
        raise ValueError(
            f"Search window ({min_wn}–{max_wn} cm⁻¹) contains no data. "
            f"Available x-axis range: {x.min():.2f}–{x.max():.2f} cm⁻¹"
        )
    
    x_search = x[mask]
    y_search = y[mask]
    
    # Find peaks using scipy.signal.find_peaks
    # Use prominence to avoid noise peaks
    peaks, properties = find_peaks(y_search, prominence=y_search.std())
    
    if len(peaks) == 0:
        raise ValueError(
            f"No peaks detected in search range ({min_wn}–{max_wn} cm⁻¹). "
            "Check if the reference spectrum is correct or adjust PEAK_SEARCH_RANGE."
        )
    
    # Find the peak with the highest intensity
    peak_intensities = y_search[peaks]
    highest_peak_idx = np.argmax(peak_intensities)
    
    # Check for multiple peaks with identical height (ambiguous)
    max_intensity = peak_intensities[highest_peak_idx]
    num_peaks_at_max = np.sum(np.isclose(peak_intensities, max_intensity, rtol=1e-5))
    
    if num_peaks_at_max > 1:
        raise ValueError(
            f"Multiple peaks ({num_peaks_at_max}) have identical maximum intensity "
            f"in search range ({min_wn}–{max_wn} cm⁻¹). Cannot determine which is silicon peak."
        )
    
    # Get the wavenumber of the detected peak
    detected_peak_position = x_search[peaks[highest_peak_idx]]
    
    return detected_peak_position


def calculate_calibration_offset(detected_peak, expected_peak):
    """
    Calculate the wavenumber calibration offset.
    
    Args:
        detected_peak: Measured silicon peak position (cm⁻¹)
        expected_peak: Theoretical silicon peak position (cm⁻¹)
    
    Returns:
        offset: Calibration offset (cm⁻¹)
    """
    offset = detected_peak - expected_peak
    return offset


def apply_wavenumber_calibration(x, offset, method="Constant Offset"):
    """
    Apply wavenumber calibration to correct the x-axis.
    
    Args:
        x: Original wavenumber array (cm⁻¹)
        offset: Calibration offset (cm⁻¹)
        method: Calibration method ("Constant Offset", "Linear Calibration", etc.)
    
    Returns:
        x_corrected: Calibrated wavenumber array (cm⁻¹)
    """
    if method == "Constant Offset":
        x_corrected = x - offset
    else:
        # Future methods can be added here
        # elif method == "Linear Calibration":
        #     x_corrected = apply_linear_calibration(x, params)
        # elif method == "Polynomial Calibration":
        #     x_corrected = apply_polynomial_calibration(x, params)
        raise ValueError(f"Calibration method '{method}' is not implemented yet.")
    
    return x_corrected


def print_calibration_report(reference_file, expected_peak, detected_peak, offset, method, status="Success"):
    """
    Print a formatted calibration report to console.
    
    Args:
        reference_file: Path to silicon reference spectrum
        expected_peak: Theoretical silicon peak (cm⁻¹)
        detected_peak: Measured silicon peak (cm⁻¹)
        offset: Calculated offset (cm⁻¹)
        method: Calibration method used
        status: Calibration status (Success/Failed)
    """
    print("\n" + "="*60)
    print("WAVENUMBER CALIBRATION")
    print("="*60)
    print(f"Reference file      : {os.path.basename(reference_file)}")
    print(f"Expected peak       : {expected_peak:.2f} cm⁻¹")
    print(f"Detected peak       : {detected_peak:.2f} cm⁻¹")
    print(f"Offset applied      : {-offset:.2f} cm⁻¹")
    print(f"Method              : {method}")
    print(f"Status              : {status}")
    print("="*60 + "\n")


def plot_calibration(x, y, detected_peak, expected_peak, offset, search_range):
    """
    Plot the silicon reference spectrum with detected and expected peaks.
    REPLACED BY plot_calibration_reference() - keeping for backward compatibility.
    
    Args:
        x: Wavenumber array (cm⁻¹)
        y: Intensity array
        detected_peak: Measured silicon peak position (cm⁻¹)
        expected_peak: Theoretical silicon peak position (cm⁻¹)
        offset: Calculated offset (cm⁻¹)
        search_range: Tuple (min_wn, max_wn) for visualization
    """
    # Use new advanced plotting function
    plot_calibration_reference(x, y, detected_peak, expected_peak, offset, search_range,
                               show=SHOW_CALIBRATION_PLOT, save=ENABLE_PLOT_SAVING)


def save_calibration_report(reference_file, expected_peak, detected_peak, offset, 
                            method, num_spectra, output_folder):
    """
    Save calibration report to a text file.
    
    Args:
        reference_file: Path to silicon reference spectrum
        expected_peak: Theoretical silicon peak (cm⁻¹)
        detected_peak: Measured silicon peak (cm⁻¹)
        offset: Calculated offset (cm⁻¹)
        method: Calibration method used
        num_spectra: Number of spectra processed
        output_folder: Directory to save the report
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    report_path = os.path.join(output_folder, "Calibration_Report.txt")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("WAVENUMBER CALIBRATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date                : {timestamp.split()[0]}\n")
        f.write(f"Time                : {timestamp.split()[1]}\n\n")
        f.write(f"Reference spectrum  : {reference_file}\n")
        f.write(f"Expected peak       : {expected_peak:.2f} cm⁻¹\n")
        f.write(f"Detected peak       : {detected_peak:.2f} cm⁻¹\n")
        f.write(f"Offset              : {offset:.2f} cm⁻¹\n")
        f.write(f"Correction applied  : {-offset:.2f} cm⁻¹\n")
        f.write(f"Calibration method  : {method}\n")
        f.write(f"Spectra processed   : {num_spectra}\n\n")
        f.write("="*60 + "\n")
        f.write("CALIBRATION STATUS: SUCCESS\n")
        f.write("="*60 + "\n")
    
    print(f"Calibration report saved: {report_path}")


###############################################################################
# COSMIC RAY REMOVAL -- implementations
###############################################################################

def _cosmic_median_filter(y, window_size, threshold):
    if window_size % 2 == 0:
        window_size += 1
    filtered = medfilt(y, kernel_size=window_size)
    residual = y - filtered
    std = np.std(residual)
    spikes = np.abs(residual) > threshold * std
    corrected = y.copy()
    corrected[spikes] = filtered[spikes]
    return corrected


def _cosmic_modified_zscore(y, threshold, mad_constant, local_window):
    n = len(y)
    half = local_window // 2
    corrected = y.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = y[lo:hi]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if mad == 0:
            mad = 1e-9
        mod_z = mad_constant * (y[i] - med) / mad
        if abs(mod_z) > threshold:
            corrected[i] = med
    return corrected


def _cosmic_hampel_filter(y, window_size, threshold):
    n = len(y)
    half = window_size // 2
    k = 1.4826
    corrected = y.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = y[lo:hi]
        med = np.median(window)
        mad = k * np.median(np.abs(window - med))
        if mad == 0:
            mad = 1e-9
        if abs(y[i] - med) > threshold * mad:
            corrected[i] = med
    return corrected


def _cosmic_wavelet_despike(y, wavelet_type, level, threshold_rule, threshold_mode):
    try:
        import pywt
    except ImportError:
        raise ImportError(
            "Wavelet Despiking needs PyWavelets. Install it with: "
            "pip install PyWavelets"
        )
    n = len(y)
    max_level = pywt.dwt_max_level(n, wavelet_type)
    level = min(level, max_level)
    coeffs = pywt.wavedec(y, wavelet_type, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(n)) if threshold_rule == "universal" else sigma
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode=threshold_mode) for c in coeffs[1:]]
    denoised = pywt.waverec(new_coeffs, wavelet_type)
    return denoised[:n]


def apply_cosmic_removal(y, tool_name):
    p = COSMIC_PARAMS[tool_name]
    if tool_name == "Median Filter":
        return _cosmic_median_filter(y, p["Window_size"], p["Threshold"])
    if tool_name == "Modified Z-score":
        return _cosmic_modified_zscore(y, p["Threshold"], p["MAD_constant"], p["Local_Window"])
    if tool_name == "Hampel Filter":
        return _cosmic_hampel_filter(y, p["Window_size"], p["Threshold"])
    if tool_name == "Wavelet Despiking":
        return _cosmic_wavelet_despike(y, p["Wavelet_type"], p["Decomposition_level"],
                                        p["Threshold_rule"], p["Threshold_mode"])
    raise ValueError(f"Unknown cosmic removal tool: {tool_name}")


###############################################################################
# BASELINE CORRECTION -- implementations (each returns the BASELINE curve;
# the pipeline subtracts it from the spectrum)
###############################################################################

def _baseline_asls(y, lam, p, niter):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y.copy()
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = (W + D).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _baseline_modpoly(y, x, order, max_iter, tol):
    poly = np.polyval(np.polyfit(x, y, order), x)
    for _ in range(max_iter):
        working = np.minimum(y, poly)
        new_coeffs = np.polyfit(x, working, order)
        new_poly = np.polyval(new_coeffs, x)
        denom = np.sum(np.abs(poly)) if np.sum(np.abs(poly)) > 0 else 1e-9
        change = np.sum(np.abs(new_poly - poly)) / denom
        poly = new_poly
        if change < tol:
            break
    return poly


def _baseline_airpls(y, lam, max_iter, tol):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    H = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y.copy()
    for i in range(1, max_iter + 1):
        W = diags(w, 0, shape=(L, L))
        Z = (W + H).tocsc()
        z = spsolve(Z, w * y)
        d = y - z
        dssn = np.abs(d[d < 0].sum())
        if dssn < tol * np.abs(y).sum():
            break
        w = np.zeros(L)
        neg = d < 0
        w[neg] = np.exp(i * np.abs(d[neg]) / dssn)
        w[0] = np.exp(i * np.abs(d[neg]).max() / dssn) if neg.any() else w[0]
        w[-1] = w[0]
    return z


def _baseline_polyfit(y, x, order):
    coeffs = np.polyfit(x, y, order)
    return np.polyval(coeffs, x)


def _baseline_rubberband(y, x):
    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    vertices = hull.vertices
    start = np.argmin(x[vertices])
    vertices = np.roll(vertices, -start)
    end = np.argmax(x[vertices])
    lower_idx = np.sort(vertices[:end + 1])
    return np.interp(x, x[lower_idx], y[lower_idx])


def _baseline_arpls(y, lam, ratio, max_iter):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    H = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y.copy()
    for _ in range(max_iter):
        W = diags(w, 0, shape=(L, L))
        Z = (W + H).tocsc()
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        if len(dn) == 0:
            break
        m, s = np.mean(dn), np.std(dn)
        if s == 0:
            break
        wt = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            w = wt
            break
        w = wt
    return z


def apply_baseline_correction(y, x, tool_name):
    p = BASELINE_PARAMS[tool_name]
    if tool_name == "AsLS":
        baseline = _baseline_asls(y, p["Lambda"], p["P"], p["Max_iterations"])
    elif tool_name == "ModPoly":
        baseline = _baseline_modpoly(y, x, p["Polynomial_order"], p["Max_iterations"], p["Tolerance"])
    elif tool_name == "airPLS":
        baseline = _baseline_airpls(y, p["Lambda"], p["Max_iterations"], p["Tolerance"])
    elif tool_name == "Polynomial Fitting":
        baseline = _baseline_polyfit(y, x, p["Polynomial_order"])
    elif tool_name == "Rubber Band":
        baseline = _baseline_rubberband(y, x)
    elif tool_name == "arPLS":
        baseline = _baseline_arpls(y, p["Lambda"], p["Ratio"], p["Max_iterations"])
    else:
        raise ValueError(f"Unknown baseline correction tool: {tool_name}")
    return y - baseline, baseline


###############################################################################
# SMOOTHING -- implementations
###############################################################################

def _smooth_savgol(y, window_length, polyorder):
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
    return savgol_filter(y, window_length, polyorder)


def _smooth_gaussian(y, sigma):
    return gaussian_filter1d(y, sigma)


def _smooth_median(y, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(y, kernel_size)


def _smooth_moving_average(y, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(y, kernel, mode="same")


def _smooth_wavelet(y, wavelet, level, threshold_rule, threshold_mode):
    try:
        import pywt
    except ImportError:
        raise ImportError(
            "Wavelet Denoising needs PyWavelets. Install it with: "
            "pip install PyWavelets"
        )
    n = len(y)
    max_level = pywt.dwt_max_level(n, wavelet)
    level = min(level, max_level)
    coeffs = pywt.wavedec(y, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(n)) if threshold_rule == "universal" else sigma
    new_coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode=threshold_mode) for c in coeffs[1:]]
    denoised = pywt.waverec(new_coeffs, wavelet)
    return denoised[:n]


def apply_smoothing(y, tool_name):
    p = SMOOTHING_PARAMS[tool_name]
    if tool_name == "Savitzky Golay":
        return _smooth_savgol(y, p["Window_length"], p["Polynomial_order"])
    if tool_name == "Gaussian Filter":
        return _smooth_gaussian(y, p["Sigma"])
    if tool_name == "Median Filter":
        return _smooth_median(y, p["Kernel_size"])
    if tool_name == "Moving Average":
        return _smooth_moving_average(y, p["Window_size"])
    if tool_name == "Wavelet Denoising":
        return _smooth_wavelet(y, p["Wavelet"], p["Level"], p["Threshold_rule"], p["Threshold_mode"])
    raise ValueError(f"Unknown smoothing tool: {tool_name}")


###############################################################################
# NORMALIZATION -- implementations
###############################################################################

def _norm_vector(y):
    norm = np.linalg.norm(y)
    return y / norm if norm != 0 else y


def _norm_minmax(y, minimum, maximum):
    rng = y.max() - y.min()
    if rng == 0:
        return np.full_like(y, minimum)
    return (y - y.min()) / rng * (maximum - minimum) + minimum


def _norm_snv(y):
    std = y.std()
    return (y - y.mean()) / std if std != 0 else y - y.mean()


def _norm_area(y, x):
    # np.trapz was renamed to np.trapezoid in newer NumPy (2.0+); this works either way
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    area = trapz_fn(y, x)
    return y / area if area != 0 else y


def _norm_maxpeak(y):
    peak = y.max()
    return y / peak if peak != 0 else y


def apply_normalization(y, x, tool_name):
    if tool_name == "Vector Normalization":
        return _norm_vector(y)
    if tool_name == "Min-Max Scaling":
        p = NORMALIZATION_PARAMS[tool_name]
        return _norm_minmax(y, p["Minimum"], p["Maximum"])
    if tool_name == "SNV":
        return _norm_snv(y)
    if tool_name == "Area Normalization":
        return _norm_area(y, x)
    if tool_name == "Maximum Peak Normalization":
        return _norm_maxpeak(y)
    raise ValueError(f"Unknown normalization tool: {tool_name}")


###############################################################################
# I/O
###############################################################################

def _has_header(file_path):
    """Check if a CSV/Excel file has a header row by testing if the first row is numeric."""
    try:
        df = pd.read_csv(file_path, nrows=1, header=None)
        pd.to_numeric(df.iloc[0], errors='raise')
        return False
    except (ValueError, TypeError):
        return True


def load_x_axis_from_file(x_axis_file_path):
    """
    Load x-axis data from a 2-column file.
    First column is the actual x-axis data (wavenumber/wavelength).
    Second column is ignored (trash/intensity).
    Returns x-axis array or None if loading fails.
    """
    if not x_axis_file_path or not os.path.isfile(x_axis_file_path):
        return None
    
    ext = os.path.splitext(x_axis_file_path)[1].lower()
    
    try:
        # Load data based on file type
        if ext == '.txt':
            data = np.loadtxt(x_axis_file_path)
        elif ext == '.csv':
            hdr = 0 if _has_header(x_axis_file_path) else None
            data = pd.read_csv(x_axis_file_path, header=hdr).values
        elif ext in ['.xls', '.xlsx']:
            data = pd.read_excel(x_axis_file_path, header=None).values
        else:
            print(f"Unsupported x-axis file format: {ext}")
            return None
        
        # Handle 1D array (single column file)
        if data.ndim == 1:
            x_axis = data
            print(f"Loaded x-axis from {x_axis_file_path}: {len(x_axis)} points (single column)")
            return x_axis
        
        # For 2D array, use first column as x-axis
        x_axis = data[:, 0]
        print(f"Loaded x-axis from {x_axis_file_path}: {len(x_axis)} points (first column)")
        return x_axis
        
    except Exception as e:
        print(f"Error loading x-axis file {x_axis_file_path}: {e}")
        return None


def get_files_to_process(folder_path, file_path):
    """
    Collect files based on folder_path and file_path configuration.
    Uses robust method that works for all possible conditions.
    Returns list of file paths to process.
    """
    all_file_paths = []
    
    # Collect files from folder if folder_path is provided and exists
    if folder_path and os.path.isdir(folder_path):
        for ext in ['txt', 'csv', 'xls', 'xlsx']:
            all_file_paths.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
    
    # Add single file if file_path is provided and exists
    if file_path and os.path.isfile(file_path):
        all_file_paths.append(file_path)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in all_file_paths:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return unique_files


def load_single_file_spectra(file_path, external_x=None):
    """
    Load spectra from a single file. Supports txt, csv, xls, xlsx.
    First column is skipped (detector position), other columns are spectra.
    
    Args:
        file_path: Path to the spectra file
        external_x: Optional pre-loaded x-axis array. If provided, this will be used
                   instead of extracting x from the file.
    
    Returns (x, y_list, filename_base).
    """
    ext = os.path.splitext(file_path)[1].lower()
    filename_base = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        # Load data based on file type
        if ext == '.txt':
            data = np.loadtxt(file_path)
        elif ext == '.csv':
            hdr = 0 if _has_header(file_path) else None
            data = pd.read_csv(file_path, header=hdr).values
        elif ext in ['.xls', '.xlsx']:
            data = pd.read_excel(file_path, header=None).values
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Ensure data is 2D
        if data.ndim == 1:
            raise ValueError(f"File {file_path} contains only 1 column. Need at least 2 columns.")
        
        if data.shape[1] < 2:
            raise ValueError(f"File {file_path} needs at least 2 columns")
        
        if external_x is not None:
            # A. External X-axis provided: use it directly
            x = external_x
            # column 1 = trash/ignored, columns 2 onward = spectra
            y_list = [data[:, i] for i in range(1, data.shape[1])]
        else:
            # B. No external X-axis
            if data.shape[1] == 2:
                # Exactly 2 columns: column 1 = X-axis, column 2 = spectrum
                x = data[:, 0]
                y_list = [data[:, 1]]
            else:
                raise ValueError(
                    f"File {file_path} has {data.shape[1]} columns but no external x-axis was provided. "
                    "A file with exactly 2 columns is needed to determine the x-axis."
                )
        
        return x, y_list, filename_base
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None


def save_spectra(x, y_list, filename_base):
    """
    Save processed spectra to output folder.
    Filename format: <original_filename>_normalized.txt
    """
    # Use the user-defined output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    out_path = os.path.join(output_folder, f"{filename_base}_normalized.txt")
    
    # Stack spectra as columns (x, y1, y2, ...)
    data = np.column_stack([x] + y_list)
    np.savetxt(out_path, data, fmt="%.6f")
    return out_path


###############################################################################
# ADVANCED PLOTTING SYSTEM
###############################################################################

def should_plot_spectrum(spectrum_index, total_spectra, file_index):
    """
    Determine if a spectrum should be plotted based on PLOT_MODE.
    
    Args:
        spectrum_index: Index of the current spectrum (0-based)
        total_spectra: Total number of spectra in the file
        file_index: Index of the current file (0-based)
    
    Returns:
        bool: True if spectrum should be plotted, False otherwise
    """
    plot_selection = PLOTTING_PARAMS["Plot Selection"]
    selected_spectra = plot_selection["Selected_Spectra"]
    
    if PLOT_MODE == "none":
        return False
    
    elif PLOT_MODE == "first":
        # Only first spectrum of first file
        return spectrum_index == 0 and file_index == 0
    
    elif PLOT_MODE == "first_per_file":
        # First spectrum of every file
        return spectrum_index == 0
    
    elif PLOT_MODE == "selected":
        # Only selected spectrum indices
        if spectrum_index in selected_spectra:
            return True
        return False
    
    elif PLOT_MODE == "random":
        # Random selection (handled by process_single_file)
        return False  # Will be determined externally
    
    elif PLOT_MODE == "all":
        # Plot every spectrum
        return True
    
    else:
        print(f"Warning: Unknown PLOT_MODE '{PLOT_MODE}'. Defaulting to 'first'.")
        return spectrum_index == 0 and file_index == 0


def save_plot(filename_base, stage_name, spectrum_index=None, method_name=None):
    """
    Save the current figure to file.
    
    Args:
        filename_base: Base filename (e.g., "Sample1")
        stage_name: Name of the processing stage (e.g., "Raw", "Baseline")
        spectrum_index: Optional spectrum index for multi-spectrum files
        method_name: Optional method/tool name to prefix in the filename
                     (e.g., "AsLS", "Savitzky_Golay")
    """
    save_cfg = PLOTTING_PARAMS["Save"]
    
    if not ENABLE_PLOT_SAVING:
        return
    
    # Create Plots directory
    plots_dir = os.path.join(output_folder, "Plots", filename_base)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Generate descriptive filename
    # Map stage names to descriptive lowercase names
    stage_map = {
        "Raw": "raw",
        "Calibration": "calibration",
        "Cosmic_Removal": "cosmic_removal",
        "Baseline": "baseline",
        "Baseline_Corrected": "baseline_corrected_normal",
        "Smoothing": "smoothing",
        "Normalization": "normalization",
        "Final": "final",
    }
    stage_lower = stage_map.get(stage_name, stage_name.lower())
    if method_name:
        # Include method name after stage name for method-dependent plots
        stage_part = f"{stage_lower}_{method_name.lower()}"
    else:
        stage_part = stage_lower
    if spectrum_index is not None:
        filename = f"spectrum_{spectrum_index}_{stage_part}.{save_cfg['Format']}"
    else:
        filename = f"{stage_part}.{save_cfg['Format']}"
    
    filepath = os.path.join(plots_dir, filename)
    
    # Save figure
    plt.savefig(filepath, dpi=save_cfg['DPI'], bbox_inches='tight')
    print(f"    Saved plot: {filepath}")


def plot_spectrum(x, y, title, filename_base=None, stage_name=None, spectrum_index=None, 
                  show=True, save=True, method_name=None):
    """
    Create a publication-quality Raman spectrum plot with improved layout.
    
    Args:
        x: Wavenumber array (Raman shift in cm⁻¹)
        y: Intensity array
        title: Plot title
        filename_base: Base filename for saving
        stage_name: Processing stage name
        spectrum_index: Spectrum index for multi-spectrum files
        show: Whether to display the plot
        save: Whether to save the plot
        method_name: Optional processing method name for the saved filename
    """
    plot_style = PLOTTING_PARAMS["Plot Style"]
    fonts = PLOTTING_PARAMS["Fonts"]
    
    # Create figure with constrained_layout
    fig, ax = plt.subplots(
        figsize=plot_style["Figure_Size"],
        dpi=plot_style["Figure_DPI"],
        constrained_layout=True
    )
    
    # Plot spectrum
    ax.plot(x, y, linewidth=plot_style["Line_Width"], color='#1f77b4')
    
    # Add spectrum index label with rounded box
    if spectrum_index is not None:
        ax.text(0.03, 0.95, f"Spectrum: {spectrum_index}", transform=ax.transAxes,
                fontsize=14, fontweight='semibold', color='#333333',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.85))
    
    # Set labels with publication-quality font sizes
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=fonts["Axes_Size"], fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontsize=fonts["Axes_Size"], fontweight='bold')
    ax.set_title(title, fontsize=fonts["Title_Size"], fontweight='bold')
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=fonts["Tick_Size"])
    
    # Add grid if enabled
    if plot_style["Show_Grid"]:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save plot if requested
    if save and filename_base and stage_name:
        save_plot(filename_base, stage_name, spectrum_index, method_name=method_name)
    
    # Show plot if requested
    if show:
        plt.show()
    
    # Close figure to free memory
    plt.close(fig)


def plot_baseline_correction(x, y_original, baseline, y_corrected, title, 
                             filename_base=None, stage_name=None, spectrum_index=None,
                             show=True, save=True, method_name=None):
    """
    Create a publication-quality baseline correction visualization with improved layout.
    
    Shows original spectrum, estimated baseline, and corrected spectrum.
    
    Args:
        x: Wavenumber array
        y_original: Original spectrum before baseline correction
        baseline: Estimated baseline
        y_corrected: Baseline-corrected spectrum
        title: Plot title
        filename_base: Base filename for saving
        stage_name: Processing stage name
        spectrum_index: Spectrum index
        show: Whether to display
        save: Whether to save
    """
    plot_style = PLOTTING_PARAMS["Plot Style"]
    fonts = PLOTTING_PARAMS["Fonts"]
    
    # Create figure with constrained_layout
    fig, ax = plt.subplots(
        figsize=plot_style["Figure_Size"],
        dpi=plot_style["Figure_DPI"],
        constrained_layout=True
    )
    
    # Plot all three components
    ax.plot(x, y_original, linewidth=plot_style["Line_Width"], color='#1f77b4', 
            label='Original Spectrum', alpha=0.7)
    ax.plot(x, baseline, linewidth=plot_style["Line_Width"], color='#ff7f0e', 
            linestyle='--', label='Estimated Baseline')
    ax.plot(x, y_corrected, linewidth=plot_style["Line_Width"], color='#2ca02c', 
            label='Corrected Spectrum')
    
    # Labels and formatting with publication-quality font sizes
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=fonts["Axes_Size"], fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontsize=fonts["Axes_Size"], fontweight='bold')
    ax.set_title(title, fontsize=fonts["Title_Size"], fontweight='bold')
    ax.legend(fontsize=fonts["Legend_Size"], framealpha=0.9)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=fonts["Tick_Size"])
    
    # Add grid if enabled
    if plot_style["Show_Grid"]:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Save if requested
    if save and filename_base and stage_name:
        save_plot(filename_base, stage_name, spectrum_index, method_name=method_name)
    
    # Show if requested
    if show:
        plt.show()
    
    # Close to free memory
    plt.close(fig)


def plot_calibration_reference(x, y, detected_peak, expected_peak, offset, search_range,
                               show=True, save=True):
    """
    Create publication-quality calibration plot for silicon reference with improved layout.
    
    Args:
        x: Wavenumber array
        y: Silicon reference intensity
        detected_peak: Automatically detected peak position
        expected_peak: Theoretical silicon peak (520.7 cm⁻¹)
        offset: Calculated offset
        search_range: Tuple (min, max) for search window
        show: Whether to display
        save: Whether to save
    """
    save_cfg, fig = new_func(x, y, detected_peak, expected_peak, offset, search_range)
    
    # Save if requested
    if save and ENABLE_PLOT_SAVING:
        plots_dir = os.path.join(output_folder, "Plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        filepath = os.path.join(plots_dir, f"Calibration.{save_cfg['Format']}")
        plt.savefig(filepath, dpi=save_cfg['DPI'], bbox_inches='tight')
        print(f"Calibration plot saved: {filepath}")
    
    # Show if requested         
    if show:
        plt.show()
    
    # Close to free memory
    plt.close(fig)

def new_func(x, y, detected_peak, expected_peak, offset, search_range):
    plot_style = PLOTTING_PARAMS["Plot Style"]
    fonts = PLOTTING_PARAMS["Fonts"]
    save_cfg = PLOTTING_PARAMS["Save"]
    
    # Create figure with constrained_layout
    fig, ax = plt.subplots(
        figsize=plot_style["Figure_Size"],
        dpi=plot_style["Figure_DPI"],
        constrained_layout=True
    )
    
    # Plot silicon spectrum
    ax.plot(x, y, linewidth=plot_style["Line_Width"], color='#1f77b4', label='Silicon Reference')
    
    # Mark detected peak
    ax.axvline(detected_peak, color='red', linestyle='--', linewidth=2, 
               label=f'Detected: {detected_peak:.2f} cm⁻¹')
    
    # Mark expected peak
    ax.axvline(expected_peak, color='green', linestyle='--', linewidth=2, 
               label=f'Expected: {expected_peak:.2f} cm⁻¹')
    
    # Highlight search range
    min_wn, max_wn = search_range
    ax.axvspan(min_wn, max_wn, alpha=0.2, color='yellow', label='Search Range')
    
    # Labels and formatting with publication-quality font sizes
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=fonts["Axes_Size"], fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontsize=fonts["Axes_Size"], fontweight='bold')
    
    # Concise title
    ax.set_title(f'Wavenumber Calibration (Offset: {offset:.2f} cm⁻¹)', 
                 fontsize=fonts["Title_Size"], fontweight='bold')
    ax.legend(fontsize=fonts["Legend_Size"], framealpha=0.9, loc='best')
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=fonts["Tick_Size"])
    
    # Add grid if enabled
    if plot_style["Show_Grid"]:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    return save_cfg,fig


###############################################################################
# MAIN PIPELINE
###############################################################################

_STAGE_METHOD_MAP = {
    "Cosmic_Removal": lambda: cosmic_removal_tool_name,
    "Baseline": lambda: baseline_correction_tool_name,
    "Baseline_Corrected": lambda: baseline_correction_tool_name,
    "Smoothing": lambda: smoothing_tool_name,
    "Normalization": lambda: normalization_tool_name,
}


def _get_method_name(stage_name):
    """Return the processing method name for a given stage, or None."""
    getter = _STAGE_METHOD_MAP.get(stage_name)
    return getter() if getter else None


def process_single_file(file_path, external_x=None, calibration_offset=None, file_index=0):
    """
    Process a single file containing multiple spectra with advanced plotting.
    
    Args:
        file_path: Path to the spectra file
        external_x: Optional pre-loaded x-axis array
        calibration_offset: Optional wavenumber calibration offset (cm⁻¹)
        file_index: Index of current file in batch (for plot mode logic)
    
    Returns True if successful, False otherwise.
    """
    plot_selection = PLOTTING_PARAMS["Plot Selection"]
    selected_spectra = plot_selection["Selected_Spectra"]
    random_spectra = plot_selection["Random_Spectra"]
    
    print(f"\nProcessing file: {file_path}")
    
    x, y_list, filename_base = load_single_file_spectra(file_path, external_x)
    
    if x is None or y_list is None:
        print(f"Skipping {file_path} due to loading error.")
        return False
    
    # Apply wavenumber calibration if offset is provided
    x_calibrated = x.copy()
    if calibration_offset is not None:
        x_calibrated = apply_wavenumber_calibration(x, calibration_offset, CALIBRATION_TOOL)
        print(f"  Applied wavenumber calibration: {-calibration_offset:.2f} cm⁻¹")
    
    # Determine which spectra to plot based on PLOT_MODE
    spectra_to_plot = []
    
    if PLOT_MODE == "none":
        spectra_to_plot = []
    elif PLOT_MODE == "first":
        if file_index == 0:
            spectra_to_plot = [0]
    elif PLOT_MODE == "first_per_file":
        spectra_to_plot = [0]
    elif PLOT_MODE == "selected":
        # Filter out invalid indices
        for idx in selected_spectra:
            if 0 <= idx < len(y_list):
                spectra_to_plot.append(idx)
            else:
                print(f"  Warning: Selected spectrum index {idx} is out of range (0-{len(y_list)-1}). Skipping.")
    elif PLOT_MODE == "random":
        # Randomly select N spectra without replacement
        num_to_select = min(random_spectra, len(y_list))
        spectra_to_plot = list(np.random.choice(len(y_list), size=num_to_select, replace=False))
        spectra_to_plot.sort()
        print(f"  Randomly selected {num_to_select} spectra for plotting: {spectra_to_plot}")
    elif PLOT_MODE == "all":
        spectra_to_plot = list(range(len(y_list)))
    else:
        print(f"  Warning: Unknown PLOT_MODE '{PLOT_MODE}'. Defaulting to 'first'.")
        if file_index == 0:
            spectra_to_plot = [0]
    
    processed_y_list = []
    
    for i, y_raw in enumerate(y_list):
        print(f"  Processing spectrum {i+1}/{len(y_list)}...")
        
        # Determine if this spectrum should be plotted
        should_plot = i in spectra_to_plot
        
        # Stage 1: Raw Spectrum
        if should_plot and PLOT_RAW:
            plot_spectrum(
                x_calibrated, y_raw,
                title=f"Raw Spectrum - {filename_base}",
                filename_base=filename_base,
                stage_name="Raw",
                spectrum_index=i if len(y_list) > 1 else None,
                show=SHOW_PLOTS,
                save=ENABLE_PLOT_SAVING,
                method_name=_get_method_name("Raw")
            )
        
        # Stage 2: Plot calibration effect (if calibration was applied)
        if should_plot and PLOT_CALIBRATION and calibration_offset is not None:
            plot_spectrum(
                x_calibrated, y_raw,
                title=f"After Wavenumber Calibration - {filename_base}\n(Correction: {-calibration_offset:.2f} cm⁻¹)",
                filename_base=filename_base,
                stage_name="Calibration",
                spectrum_index=i if len(y_list) > 1 else None,
                show=SHOW_PLOTS,
                save=ENABLE_PLOT_SAVING,
                method_name=_get_method_name("Calibration")
            )
        
        # Stage 3: Cosmic Ray Removal
        y_cosmic = apply_cosmic_removal(y_raw, cosmic_removal_tool_name) if ENABLE_COSMIC_REMOVAL else y_raw
        if should_plot and PLOT_COSMIC and ENABLE_COSMIC_REMOVAL:
            plot_spectrum(
                x_calibrated, y_cosmic,
                title=f"After Cosmic Ray Removal ({cosmic_removal_tool_name}) - {filename_base}",
                filename_base=filename_base,
                stage_name="Cosmic_Removal",
                spectrum_index=i if len(y_list) > 1 else None,
                show=SHOW_PLOTS,
                save=ENABLE_PLOT_SAVING,
                method_name=_get_method_name("Cosmic_Removal")
            )
        
        # Stage 4: Baseline Correction
        if ENABLE_BASELINE_CORRECTION:
            y_baseline_corrected, baseline = apply_baseline_correction(y_cosmic, x_calibrated, baseline_correction_tool_name)
        else:
            y_baseline_corrected = y_cosmic
        if should_plot and PLOT_BASELINE and ENABLE_BASELINE_CORRECTION:
            plot_baseline_correction(
                x_calibrated, y_cosmic, baseline, y_baseline_corrected,
                title=f"Baseline Correction ({baseline_correction_tool_name}) - {filename_base}",
                filename_base=filename_base,
                stage_name="Baseline",
                spectrum_index=i if len(y_list) > 1 else None,
                show=SHOW_PLOTS,
                save=ENABLE_PLOT_SAVING,
                method_name=_get_method_name("Baseline")
            )
        
        # Stage 4b: Normal plot of baseline-corrected spectrum
        if should_plot and PLOT_BASELINE_CORRECTED_NORMAL and ENABLE_BASELINE_CORRECTION:
            plot_spectrum(
                x_calibrated, y_baseline_corrected,
                title=f"Baseline Corrected ({baseline_correction_tool_name}) - {filename_base}",
                filename_base=filename_base,
                stage_name="Baseline_Corrected",
                spectrum_index=i if len(y_list) > 1 else None,
                show=SHOW_PLOTS,
                save=ENABLE_PLOT_SAVING,
                method_name=_get_method_name("Baseline_Corrected")
            )
        
        # Stage 5: Smoothing
        y_smoothed = apply_smoothing(y_baseline_corrected, smoothing_tool_name) if ENABLE_SMOOTHING else y_baseline_corrected
        if should_plot and PLOT_SMOOTHING and ENABLE_SMOOTHING:
            plot_spectrum(
                x_calibrated, y_smoothed,
                title=f"After Smoothing ({smoothing_tool_name}) - {filename_base}",
                filename_base=filename_base,
                stage_name="Smoothing",
                spectrum_index=i if len(y_list) > 1 else None,
                show=SHOW_PLOTS,
                save=ENABLE_PLOT_SAVING,
                method_name=_get_method_name("Smoothing")
            )
        
        # Stage 6: Normalization
        y_normalized = apply_normalization(y_smoothed, x_calibrated, normalization_tool_name) if ENABLE_NORMALIZATION else y_smoothed
        if should_plot and PLOT_NORMALIZATION and ENABLE_NORMALIZATION:
            plot_spectrum(
                x_calibrated, y_normalized,
                title=f"After Normalization ({normalization_tool_name}) - {filename_base}",
                filename_base=filename_base,
                stage_name="Normalization",
                spectrum_index=i if len(y_list) > 1 else None,
                show=SHOW_PLOTS,
                save=ENABLE_PLOT_SAVING,
                method_name=_get_method_name("Normalization")
            )
        
        # Stage 7: Final Processed Spectrum
        if should_plot and PLOT_FINAL:
            plot_spectrum(
                x_calibrated, y_normalized,
                title=f"Final Processed Spectrum - {filename_base}",
                filename_base=filename_base,
                stage_name="Final",
                spectrum_index=i if len(y_list) > 1 else None,
                show=SHOW_PLOTS,
                save=ENABLE_PLOT_SAVING,
                method_name=_get_method_name("Final")
            )
        
        processed_y_list.append(y_normalized)
    
    if SAVE_NORMALIZED_TXT:
        out_path = save_spectra(x_calibrated, processed_y_list, filename_base)
        print(f"  Saved: {out_path}")
    return True


def run_pipeline():
    """
    Main pipeline that handles three modes:
    1. Single file only (file_path set, folder_path empty)
    2. Batch folder only (folder_path set, file_path empty)
    3. Both (both set - process folder + specific file)
    
    Pipeline Order:
    1. Load x-axis (if external file provided)
    2. Perform wavenumber calibration (if enabled)
    3. Process all spectra with calibrated x-axis
    """
    calibration = CALIBRATION_PARAMS[CALIBRATION_TOOL]
    expected_peak = calibration["Expected_Silicon_Peak"]
    search_range = calibration["Peak_Search_Range"]
    
    # =========================================================================
    # STEP 1: Load x-axis if provided
    # =========================================================================
    external_x = None
    if x_axis_file_path:
        print(f"Loading x-axis from: {x_axis_file_path}")
        external_x = load_x_axis_from_file(x_axis_file_path)
        if external_x is None:
            print("Warning: Failed to load x-axis file. Will use x-axis from spectra files.")

    # =========================================================================
    # STEP 2: Collect files to process
    # =========================================================================
    files = get_files_to_process(folder_path, file_path)

    if not files:
        print("No files to process. Please check folder_path and file_path configuration.")
        return

    print(f"\nFound {len(files)} file(s) to process:")
    for f in files:
        print(f"  - {f}")

    # =========================================================================
    # STEP 3: If no external x-axis, search Y-files for a 2-column file
    # =========================================================================
    if external_x is None:
        print("\nNo external x-axis is provided")
        # Search for a file with exactly 2 columns
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            try:
                if ext == '.txt':
                    data = np.loadtxt(f)
                elif ext == '.csv':
                    hdr = 0 if _has_header(f) else None
                    data = pd.read_csv(f, header=hdr).values
                elif ext in ['.xls', '.xlsx']:
                    data = pd.read_excel(f, header=None).values
                else:
                    continue

                if data.ndim == 2 and data.shape[1] == 2:
                    external_x = data[:, 0]
                    print(f"Found x-axis from 2-column file: {os.path.basename(f)}")
                    break
            except Exception:
                continue

        if external_x is None:
            print("No X-axis found in Y-files")
            return

    # =========================================================================
    # STEP 4: Perform wavenumber calibration (if enabled)
    # =========================================================================
    calibration_offset = None

    if ENABLE_WAVENUMBER_CALIBRATION:
        if not SILICON_REFERENCE_FILE:
            print("\nWarning: ENABLE_WAVENUMBER_CALIBRATION is True but SILICON_REFERENCE_FILE is not set.")
            print("Skipping wavenumber calibration.\n")
        else:
            try:
                print(f"\nPerforming wavenumber calibration...")
                print(f"Loading silicon reference: {SILICON_REFERENCE_FILE}")
                
                # Load silicon reference spectrum
                si_x, si_y_list, si_filename = load_single_file_spectra(SILICON_REFERENCE_FILE, external_x)
                
                if si_x is None or si_y_list is None:
                    raise ValueError("Failed to load silicon reference spectrum")
                
                # Use the first spectrum if multiple spectra in reference file
                si_y = si_y_list[0]
                
                if len(si_y_list) > 1:
                    print(f"  Reference file contains {len(si_y_list)} spectra. Using first spectrum.")
                
                # Detect silicon peak automatically
                detected_peak = detect_silicon_peak(
                    si_x, si_y, 
                    search_range, 
                    expected_peak
                )
                
                # Calculate calibration offset
                calibration_offset = calculate_calibration_offset(
                    detected_peak, 
                    expected_peak
                )
                
                # Print calibration report
                print_calibration_report(
                    SILICON_REFERENCE_FILE,
                    expected_peak,
                    detected_peak,
                    calibration_offset,
                    CALIBRATION_TOOL,
                    status="Success"
                )
                
                # Plot calibration (if enabled)
                if SHOW_CALIBRATION_PLOT:
                    plot_calibration(
                        si_x, si_y,
                        detected_peak,
                        expected_peak,
                        calibration_offset,
                        search_range
                    )
                
            except Exception as e:
                print(f"\nERROR: Wavenumber calibration failed: {e}")
                print("Continuing without calibration.\n")
                calibration_offset = None
    
    # =========================================================================
    # STEP 5: Process all spectra
    # =========================================================================
    successful = 0
    failed = 0

    for i, file in enumerate(files):
        # Pass file index for plot mode logic
        if process_single_file(file, external_x, calibration_offset, file_index=i):
            successful += 1
        else:
            failed += 1

    # =========================================================================
    # STEP 6: Save calibration report (if enabled and calibration was done)
    # =========================================================================
    if SAVE_CALIBRATION_REPORT and calibration_offset is not None:
        try:
            detected_peak = expected_peak + calibration_offset
            save_calibration_report(
                SILICON_REFERENCE_FILE,
                expected_peak,
                detected_peak,
                calibration_offset,
                CALIBRATION_TOOL,
                successful,
                output_folder
            )
        except Exception as e:
            print(f"Warning: Failed to save calibration report: {e}")

    # =========================================================================
    # STEP 7: Print final summary
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successful: {successful} file(s)")
    print(f"Failed: {failed} file(s)")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_pipeline()
