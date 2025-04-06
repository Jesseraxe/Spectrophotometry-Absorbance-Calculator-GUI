import streamlit as st
import numpy as np
import pandas as pd
import io
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter
from functools import lru_cache

# --- Constants ---
# Default values, can be overridden by data range or user input
DEFAULT_SPECTRUM_MIN = 340
DEFAULT_SPECTRUM_MAX = 800
EPSILON = 1e-10     # Small value to prevent division by zero or log of zero

# --- Caching Functions ---

@st.cache_data
def read_spectral_file(uploaded_file):
    """
    Reads a spectral data file (space-delimited) into a pandas DataFrame.
    Caches the result to avoid re-reading the same file.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        pandas.DataFrame or None: DataFrame with 'Nanometers' and 'Counts' columns,
                                  or None if reading fails or format is invalid.
    """
    if uploaded_file is None:
        return None
    try:
        # Ensure reading from the start of the file
        uploaded_file.seek(0)
        # Read the file, trying space delimiter first, then comma
        try:
            df = pd.read_csv(uploaded_file, delim_whitespace=True)
            if 'Nanometers' not in df.columns or 'Counts' not in df.columns:
                 # If space didn't work, try comma
                 uploaded_file.seek(0)
                 df = pd.read_csv(uploaded_file) # Auto-detect delimiter (often comma)
        except Exception: # Fallback to comma if space fails or raises error
             uploaded_file.seek(0)
             df = pd.read_csv(uploaded_file)

        # --- Validation ---
        if 'Nanometers' not in df.columns or 'Counts' not in df.columns:
            st.error(f"File '{uploaded_file.name}' is missing required columns 'Nanometers' or 'Counts'. "
                     "Please ensure the header is correct and columns are separated by spaces or commas.")
            return None

        # Ensure numeric types and handle potential errors
        df['Nanometers'] = pd.to_numeric(df['Nanometers'], errors='coerce')
        df['Counts'] = pd.to_numeric(df['Counts'], errors='coerce')

        # Check for conversion errors
        if df['Nanometers'].isnull().any() or df['Counts'].isnull().any():
            st.warning(f"File '{uploaded_file.name}' contains non-numeric data in 'Nanometers' or 'Counts' columns. "
                       "Problematic rows were removed.")
            df.dropna(subset=['Nanometers', 'Counts'], inplace=True)
            if df.empty:
                 st.error(f"File '{uploaded_file.name}' resulted in no valid data after cleaning.")
                 return None

        # Sort by wavelength just in case
        df.sort_values(by='Nanometers', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    except pd.errors.EmptyDataError:
        st.error(f"File '{uploaded_file.name}' is empty.")
        return None
    except Exception as e:
        st.error(f"Error reading file '{uploaded_file.name}': {e}. Check format (space or comma separated, 'Nanometers Counts' header).")
        return None

@st.cache_data
def wavelength_to_rgb(wavelength):
    """
    Converts a wavelength (nm) to an approximate RGB color tuple (0-1 range).
    Uses caching for performance.

    Args:
        wavelength (float): Wavelength in nanometers.

    Returns:
        tuple: (R, G, B) values, each between 0 and 1.
    """
    # Define color ranges based on wavelength
    if 380 <= wavelength < 440:  # Violet
        # Modified violet formula to blend with UV at 380nm boundary
        ratio = (wavelength - 380) / (440 - 380)
        # At 380nm (ratio=0): R=0.6, B=1.0
        # At 440nm (ratio=1): R=0.0, B=1.0
        R = 0.6 * (1 - ratio)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:  # Blue
        R, G, B = 0.0, (wavelength - 440) / (490 - 440), 1.0
    elif 490 <= wavelength < 510:  # Cyan
        R, G, B = 0.0, 1.0, -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:  # Green
        R, G, B = (wavelength - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= wavelength < 645:  # Yellow to Orange
        R, G, B = 1.0, -(wavelength - 645) / (645 - 580), 0.0
    elif 645 <= wavelength <= 780:  # Red
        R, G, B = 1.0, 0.0, 0.0
    elif 340 <= wavelength < 380:  # Near UV (approx. deep purple)
        # Modified UV formula to blend with violet at 380nm boundary
        ratio = (wavelength - 340) / (380 - 340)
        # At 340nm (ratio=0): R=0.3, B=0.6
        # At 380nm (ratio=1): R=0.6, B=1.0 (matching violet at 380nm)
        R = 0.3 + (0.6 - 0.3) * ratio
        G = 0.0
        B = 0.6 + (1.0 - 0.6) * ratio
    elif 780 < wavelength <= 800:  # Near IR (approx. deep red)
        # Modified Near IR formula to blend with red at 780nm boundary
        ratio = (wavelength - 780) / (800 - 780)
        # At 780nm (ratio=0): Keep R=1.0 to match the red region
        # At 800nm (ratio=1): R=0.8
        R = 1.0 - 0.2 * ratio
        G = 0.0
        B = 0.0
    else:  # Outside defined spectrum range (grey)
        R, G, B = 0.5, 0.5, 0.5

    # Apply intensity adjustment factor for edges of visible spectrum
    factor = 1.0
    if 380 <= wavelength <= 780:
        if wavelength < 420:
            # Modified to create a smoother transition at 380nm
            # Factor increases from 0.7 to 1.0 between 380-420nm
            factor = 0.7 + 0.3 * (wavelength - 380) / 40
        elif wavelength > 700:
            # Modified to ensure a smooth transition at 780nm
            # At 700nm: factor = 1.0
            # At 780nm: factor = 0.3
            factor = 1.0 - 0.7 * (wavelength - 700) / 80
    elif wavelength < 380:  # Also apply factor to UV region
        # Keep consistent with modified factor at 380nm
        factor = 0.7
    elif wavelength > 780:  # Apply consistent factor to Near IR
        # Start with the same factor as at 780nm
        factor = 0.3

    # Ensure RGB values are within [0, 1] after applying factor
    R = max(0.0, min(1.0, R * factor))
    G = max(0.0, min(1.0, G * factor))
    B = max(0.0, min(1.0, B * factor))

    return (R, G, B)

@st.cache_data
def get_color_name(wavelength):
    """
    Returns the approximate color name for a given wavelength (nm).
    Uses caching for performance.

    Args:
        wavelength (float): Wavelength in nanometers.

    Returns:
        str: Name of the spectral region (e.g., "Violet", "Green", "Near IR").
    """
    if wavelength < 380: return "Near UV"
    elif 380 <= wavelength < 450: return "Violet"
    elif 450 <= wavelength < 485: return "Blue"
    elif 485 <= wavelength < 500: return "Cyan"
    elif 500 <= wavelength < 565: return "Green"
    elif 565 <= wavelength < 590: return "Yellow"
    elif 590 <= wavelength < 625: return "Orange"
    elif 625 <= wavelength <= 780: return "Red"
    else: return "Near IR"

# --- Data Processing Functions ---

def average_dataframes(dataframes, file_type_label):
    """
    Averages the 'Counts' column across multiple DataFrames.
    Validates that all DataFrames have matching 'Nanometers' columns.

    Args:
        dataframes (list): A list of valid pandas DataFrames to average.
        file_type_label (str): Label for the type of files being averaged (e.g., "Reference").

    Returns:
        pandas.DataFrame or None: A new DataFrame with averaged 'Counts',
                                  or None if validation fails or input is empty/invalid.
    """
    if not dataframes:
        # This case should ideally be handled before calling, but added for safety
        st.warning(f"No valid {file_type_label} dataframes provided for averaging.")
        return None

    # Use the first dataframe's wavelengths and shape as the reference
    reference_df = dataframes[0]
    reference_wavelengths = reference_df['Nanometers'].values
    reference_shape = reference_df.shape

    # Validate other dataframes
    for i, df in enumerate(dataframes[1:], 1):
        if df.shape != reference_shape or not np.array_equal(df['Nanometers'].values, reference_wavelengths):
            st.error(f"Wavelength values or data shape in {file_type_label} file {i+1} "
                     f"(starting from the second file) do not match the first {file_type_label} file. "
                     f"Please ensure all files within a category (Reference, Sample, Dark) "
                     f"have the exact same wavelength points and structure.")
            return None

    # Create a new dataframe with the reference wavelengths
    result_df = pd.DataFrame({'Nanometers': reference_wavelengths})

    # Average the 'Counts'
    try:
        # Stack counts vertically and calculate mean along axis 0
        counts_arrays = np.vstack([df['Counts'].values for df in dataframes])
        result_df['Counts'] = np.mean(counts_arrays, axis=0)
    except Exception as e:
        st.error(f"Error during averaging of {file_type_label} data: {e}")
        return None

    return result_df

def process_file_uploads(uploaded_files, file_type_label):
    """
    Reads multiple uploaded files, validates them, averages if multiple, and shows progress.

    Args:
        uploaded_files (list): List of Streamlit UploadedFile objects.
        file_type_label (str): Description of the files being processed (e.g., "Reference").

    Returns:
        pandas.DataFrame or None: Processed (potentially averaged) DataFrame,
                                  or None if no valid files are processed.
    """
    if not uploaded_files:
        return None # No files uploaded for this type

    valid_dataframes = []
    # Use a placeholder for the progress bar if needed later
    # progress_bar_placeholder = st.empty()
    # progress_text = f"Processing {len(uploaded_files)} {file_type_label} file(s)..."

    for i, file in enumerate(uploaded_files):
        # Read each file using the cached function
        df = read_spectral_file(file)
        if df is not None:
            valid_dataframes.append(df)
        # Update progress (optional, can make UI busy if updates too fast)
        # progress_bar_placeholder.progress((i + 1) / len(uploaded_files), text=progress_text)

    # progress_bar_placeholder.empty() # Remove progress bar

    if not valid_dataframes:
        st.error(f"No valid {file_type_label} files could be read or processed.")
        return None

    # Average if more than one valid file was read
    if len(valid_dataframes) > 1:
        st.info(f"Averaging {len(valid_dataframes)} {file_type_label} files...")
        averaged_df = average_dataframes(valid_dataframes, file_type_label)
        return averaged_df # Returns None if averaging fails
    else:
        # Only one valid file, return it directly
        return valid_dataframes[0]

def calculate_absorbance(df_reference, df_sample, df_dark=None, path_length_cm=1.0):
    """
    Calculates absorbance based on reference, sample, and optional dark spectra.

    Args:
        df_reference (pd.DataFrame): Processed (averaged) reference spectrum.
        df_sample (pd.DataFrame): Processed (averaged) sample spectrum.
        df_dark (pd.DataFrame, optional): Processed (averaged) dark spectrum. Defaults to None.
        path_length_cm (float): Optical path length in centimeters.

    Returns:
        pd.DataFrame or None: DataFrame containing wavelengths, raw counts,
                              corrected counts, absorbance, and absorbance per cm,
                              or None if input validation fails.
    """
    # --- Input Validation ---
    if df_reference is None or df_sample is None:
        # Errors handled during file processing, so just return None here
        return None

    # Check wavelength consistency between reference and sample
    if not np.array_equal(df_reference['Nanometers'].values, df_sample['Nanometers'].values):
        st.error("Wavelength mismatch between Reference and Sample data. "
                 "Ensure all uploaded files (Reference, Sample, Dark) share the exact same wavelength points.")
        return None

    # Check wavelength consistency with dark spectrum if provided
    if df_dark is not None and not np.array_equal(df_reference['Nanometers'].values, df_dark['Nanometers'].values):
        st.error("Wavelength mismatch between Dark data and Reference/Sample data. "
                 "Ensure all uploaded files (Reference, Sample, Dark) share the exact same wavelength points.")
        return None

    # --- Calculations ---
    df_result = pd.DataFrame({'Nanometers': df_reference['Nanometers']})
    df_result['Reference_Counts'] = df_reference['Counts']
    df_result['Sample_Counts'] = df_sample['Counts']

    # Determine dark counts (use zero if no dark spectrum provided)
    if df_dark is not None:
        dark_counts = df_dark['Counts'].values # Use numpy array for subtraction
        df_result['Dark_Counts'] = dark_counts
    else:
        dark_counts = 0 # Scalar zero, broadcasts correctly

    # Apply dark correction and clipping (prevent non-positive values for log)
    reference_corrected = np.clip(df_result['Reference_Counts'].values - dark_counts, a_min=EPSILON, a_max=None)
    sample_corrected = np.clip(df_result['Sample_Counts'].values - dark_counts, a_min=EPSILON, a_max=None)

    df_result['Reference_Corrected'] = reference_corrected
    df_result['Sample_Corrected'] = sample_corrected

    # Calculate Absorbance: A = log10(I₀ / I) = log10(Reference_Corrected / Sample_Corrected)
    # Handle potential division by zero or invalid values gracefully
    with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings during calculation
        absorbance = np.log10(reference_corrected / sample_corrected)
        # Replace NaNs or Infs resulting from division issues with 0 or np.nan
        absorbance[~np.isfinite(absorbance)] = 0 # Replace with 0 for plotting continuity

    df_result['Absorbance'] = absorbance

    # Calculate absorbance normalized by path length
    if path_length_cm is not None and path_length_cm > 0:
        df_result['Absorbance_per_cm'] = df_result['Absorbance'] / path_length_cm
    else:
        # Assign NaN or keep column absent if path length is invalid/zero
        df_result['Absorbance_per_cm'] = np.nan
        if path_length_cm == 0:
            st.warning("Path length is zero, cannot calculate Absorbance per cm.")
        # If path_length_cm is None, it means input was invalid, warning shown elsewhere

    return df_result

def apply_savitzky_golay(data, window_length, poly_order):
    """
    Applies Savitzky-Golay filter to the data. Includes basic validation.

    Args:
        data (np.array): The data series to smooth.
        window_length (int): The length of the filter window (must be odd).
        poly_order (int): The order of the polynomial used to fit the samples.

    Returns:
        np.array or None: The smoothed data, or None if parameters are invalid or error occurs.
    """
    # Basic validation (more robust checks in UI)
    if not isinstance(window_length, int) or window_length <= 0 or window_length % 2 == 0:
        st.warning(f"Savitzky-Golay: Window Length ({window_length}) must be a positive odd integer. Skipping smoothing.")
        return None
    if not isinstance(poly_order, int) or poly_order < 0:
         st.warning(f"Savitzky-Golay: Polynomial Order ({poly_order}) must be a non-negative integer. Skipping smoothing.")
         return None
    if poly_order >= window_length:
        st.warning(f"Savitzky-Golay: Polynomial Order ({poly_order}) must be less than Window Length ({window_length}). Skipping smoothing.")
        return None
    if len(data) < window_length:
        st.warning(f"Savitzky-Golay: Data length ({len(data)}) is shorter than Window Length ({window_length}). Skipping smoothing.")
        return None

    try:
        smoothed_data = savgol_filter(data, window_length, poly_order)
        return smoothed_data
    except Exception as e:
        st.error(f"Error applying Savitzky-Golay filter: {e}")
        return None

def find_absorption_peaks(wavelengths, absorbances, min_height=0.1, min_distance_nm=15, min_prominence=0.05):
    """
    Finds absorption peaks in spectral data using scipy.signal.find_peaks.
    Operates on the provided full arrays. Filtering by range happens later.

    Args:
        wavelengths (np.array): Array of wavelengths.
        absorbances (np.array): Array of corresponding absorbance values.
        min_height (float): Minimum height threshold for a peak (absolute absorbance).
        min_distance_nm (float): Minimum required horizontal distance (in nm) between peaks.
        min_prominence (float): Minimum required prominence of a peak.

    Returns:
        tuple: (peak_indices, properties) as returned by find_peaks, or ([], {}) if error/no peaks.
               Returns None if input is invalid.
    """
    if wavelengths is None or absorbances is None or len(wavelengths) != len(absorbances) or len(wavelengths) < 3:
        st.warning("Peak Finding: Insufficient or mismatched data provided.")
        return None # Indicate invalid input

    # Ensure parameters are valid numbers
    if not all(isinstance(p, (int, float)) and np.isfinite(p) for p in [min_height, min_distance_nm, min_prominence]):
         st.warning("Peak Finding: Invalid non-numeric parameter provided (Min Height, Distance, or Prominence).")
         return None

    if min_distance_nm <= 0:
         st.warning("Peak Finding: Minimum Peak Distance must be positive.")
         return None # Or default to 1 point distance? Returning None seems safer.

    # Convert min_distance from nm to number of data points
    diffs = np.diff(wavelengths)
    if np.all(diffs <= 0): # Handle non-increasing wavelengths
        avg_spacing = 1.0 # Avoid division by zero, assume unit spacing
        st.warning("Peak Finding: Wavelengths are not strictly increasing. Distance calculation might be inaccurate.")
    else:
        avg_spacing = np.mean(diffs[diffs > 0]) # Use only positive differences
        if avg_spacing <= 0: # Should not happen if diffs > 0 exists, but safety check
             avg_spacing = 1.0

    distance_points = max(1, int(round(min_distance_nm / avg_spacing))) # Ensure at least 1 point distance

    try:
        peak_indices, properties = find_peaks(
            absorbances,
            height=min_height if min_height > -np.inf else None, # find_peaks needs None for no threshold
            distance=distance_points,
            prominence=min_prominence if min_prominence > 0 else None # find_peaks needs None for no threshold
        )
        return peak_indices, properties
    except Exception as e:
        st.error(f"Error during peak finding: {e}")
        return None # Indicate error

# --- Plotting Functions ---

def create_preview_plot(df, title, color):
    """Creates a small Plotly figure for previewing uploaded file data."""
    fig = go.Figure()
    if df is None or df.empty:
        # Show a message within the plot area
        fig.update_layout(
            title=f"{title} (No data)",
            height=150, # Smaller height for preview
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis={'visible': False}, # Hide axes
            yaxis={'visible': False},
            annotations=[{
                'text': "No data to display",
                'xref': "paper", 'yref': "paper",
                'x': 0.5, 'y': 0.5, 'showarrow': False
            }]
        )
        return fig

    fig.add_trace(go.Scatter(
        x=df['Nanometers'],
        y=df['Counts'],
        mode='lines',
        line=dict(color=color, width=1.5), # Thinner line for preview
        name='Counts'
    ))
    fig.update_layout(
        title=title,
        height=150, # Smaller height
        margin=dict(l=10, r=10, t=30, b=10), # Minimal margins
        xaxis_title=None, # Hide axis titles for preview
        yaxis_title=None,
        xaxis=dict(showticklabels=False), # Hide tick labels
        yaxis=dict(showticklabels=False),
        hovermode='x unified',
        showlegend=False # Hide legend for preview
    )
    return fig

def create_base_plot(title, x_label, y_label, x_range=None):
    """Creates a base Plotly figure configuration."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(range=x_range if x_range else None), # Apply range if provided
        hovermode='x unified',
        height=500, # Standard height for main plots
        legend=dict(
            orientation="h", # Horizontal legend
            yanchor="bottom", y=1.02, # Position above plot
            xanchor="right", x=1
        ),
        margin=dict(t=80) # Increased top margin for title and legend
    )
    return fig

def plot_raw_counts(df_plot, dark_counts_plot=None, x_range=None):
    """Plots Raw Reference, Sample, and optional Dark Counts within the specified x_range."""
    fig = create_base_plot(
        title='Raw Detector Counts vs Wavelength',
        x_label='Wavelength (nm)',
        y_label='Counts',
        x_range=x_range # Pass the range to the base plot creator
    )
    # Use 'visible="legendonly"' to hide traces initially but keep them in legend
    fig.add_trace(go.Scatter(
        x=df_plot['Nanometers'], y=df_plot['Reference_Counts'], mode='lines',
        name='Reference', line=dict(color='dodgerblue'), # Changed color slightly
        hovertemplate='Ref W: %{x:.1f} nm<br>Counts: %{y:,.0f}<extra></extra>' # Formatted counts
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['Nanometers'], y=df_plot['Sample_Counts'], mode='lines',
        name='Sample', line=dict(color='mediumseagreen'), # Changed color slightly
        hovertemplate='Sample W: %{x:.1f} nm<br>Counts: %{y:,.0f}<extra></extra>'
    ))
    if dark_counts_plot is not None and not dark_counts_plot.empty:
        fig.add_trace(go.Scatter(
            x=df_plot['Nanometers'], y=dark_counts_plot, mode='lines',
            name='Dark', line=dict(color='black', dash='dot'), # Dotted line for dark
            hovertemplate='Dark W: %{x:.1f} nm<br>Counts: %{y:,.0f}<extra></extra>',
            visible='legendonly' # Hide dark counts by default
        ))
    return fig

def plot_corrected_counts(df_plot, x_range=None):
    """Plots Dark-Corrected Reference and Sample Counts within the specified x_range."""
    fig = create_base_plot(
        title='Dark-Corrected Counts vs Wavelength',
        x_label='Wavelength (nm)',
        y_label='Counts (Dark Corrected)',
        x_range=x_range # Pass the range to the base plot creator
    )
    fig.add_trace(go.Scatter(
        x=df_plot['Nanometers'], y=df_plot['Reference_Corrected'], mode='lines',
        name='Reference (Corrected)', line=dict(color='dodgerblue'),
        hovertemplate='Ref Corrected W: %{x:.1f} nm<br>Counts: %{y:,.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['Nanometers'], y=df_plot['Sample_Corrected'], mode='lines',
        name='Sample (Corrected)', line=dict(color='mediumseagreen'),
        hovertemplate='Sample Corrected W: %{x:.1f} nm<br>Counts: %{y:,.2f}<extra></extra>'
    ))
    return fig

def plot_absorbance(df_plot, path_length_mm, absorption_peaks_info, smoothed_absorbance=None, x_range=None):
    """
    Plots Absorbance vs Wavelength with a gradient fill under the curve,
    optionally with smoothed data and peaks, respecting the x_range.

    Args:
        df_plot (pd.DataFrame): DataFrame containing 'Nanometers', 'Absorbance'.
        path_length_mm (float): Path length in mm (for title).
        absorption_peaks_info (tuple or None): Result from find_absorption_peaks (indices, properties), or None.
        smoothed_absorbance (np.array, optional): Smoothed absorbance data.
        x_range (list, optional): X-axis range [min, max] to display.

    Returns:
        plotly.graph_objects.Figure: The generated plot.
    """
    path_length_text = f"{path_length_mm:.2f} mm" if path_length_mm is not None else "N/A"
    fig = create_base_plot(
        title=f'Absorbance vs Wavelength (Path Length: {path_length_text})',
        x_label='Wavelength (nm)',
        y_label='Absorbance (AU)', # AU = Absorbance Units
        x_range=x_range # Pass the range to the base plot creator
    )

    wavelengths = df_plot['Nanometers'].values
    absorbances = df_plot['Absorbance'].values

    # --- Add Gradient Fill using Segments ---
    num_segments = 300 # Increased from 100 for more seamless color transitions
    # Create segments based on the full data range first
    indices = np.linspace(0, len(wavelengths) - 1, num_segments + 1, dtype=int)

    for i in range(num_segments):
        start_idx, end_idx = indices[i], indices[i+1]
        if end_idx <= start_idx: continue

        segment_wl = wavelengths[start_idx : end_idx + 1]
        segment_abs = absorbances[start_idx : end_idx + 1]

        # Get color based on the midpoint wavelength of the segment
        mid_wavelength = segment_wl[len(segment_wl) // 2]
        rgb = wavelength_to_rgb(mid_wavelength)
        color = f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.6)' # Slightly transparent

        fig.add_trace(go.Scatter(
            x=segment_wl,
            y=segment_abs,
            mode='lines',
            fill='tozeroy',
            fillcolor=color,
            line=dict(color='rgba(0,0,0,0)', width=0), # Invisible line for segment
            showlegend=False,
            hoverinfo='skip'
        ))

    # --- Add Invisible Line for Hover and Legend Entry (Full Data) ---
    color_names = [get_color_name(wl) for wl in wavelengths]
    custom_data = np.stack((color_names,), axis=-1)

    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=absorbances,
        mode='lines',
        name='Absorbance',
        line=dict(color='rgba(0,0,0,0)', width=0), # Invisible line
        customdata=custom_data,
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Wavelength: %{x:.1f} nm<br>'
            'Absorbance: %{y:.4f} AU<extra></extra>'
        ),
        showlegend=True
    ))

    # --- Add Smoothed Absorbance Line (if available) ---
    if smoothed_absorbance is not None and len(smoothed_absorbance) == len(wavelengths):
         smoothed_custom_data = np.stack(([get_color_name(wl) for wl in wavelengths],), axis=-1)
         fig.add_trace(go.Scatter(
            x=wavelengths,
            y=smoothed_absorbance,
            mode='lines',
            name='Smoothed Absorbance',
            line=dict(color='purple', width=1.5, dash='dash'),
            customdata=smoothed_custom_data,
            hovertemplate=(
                '<b>%{customdata[0]} (Smoothed)</b><br>'
                'Wavelength: %{x:.1f} nm<br>'
                'Smoothed Abs: %{y:.4f} AU<extra></extra>'
            ),
            visible='legendonly'
         ))

    # --- Add Peak Markers and Labels (Filtered by x_range) ---
    peak_count_total = 0
    peak_count_displayed = 0
    if absorption_peaks_info is not None:
        peak_indices, properties = absorption_peaks_info
        peak_count_total = len(peak_indices)

        if peak_count_total > 0:
            # Determine which y-values to use for peak markers (raw or smoothed)
            peak_y_values = smoothed_absorbance if (smoothed_absorbance is not None and st.session_state.get('peak_source_label') == "Smoothed Absorbance") else absorbances

            all_peak_x = wavelengths[peak_indices]
            all_peak_y = peak_y_values[peak_indices]
            all_peak_prominences = properties.get('prominences', [None]*peak_count_total)
            all_peak_heights = properties.get('peak_heights', all_peak_y)

            # Filter peaks based on the plot's x_range
            display_peaks_mask = np.ones(peak_count_total, dtype=bool)
            if x_range:
                min_wl, max_wl = x_range
                display_peaks_mask = (all_peak_x >= min_wl) & (all_peak_x <= max_wl)

            peak_x = all_peak_x[display_peaks_mask]
            peak_y = all_peak_y[display_peaks_mask]
            peak_prominences = [p for p, m in zip(all_peak_prominences, display_peaks_mask) if m]
            peak_heights = [h for h, m in zip(all_peak_heights, display_peaks_mask) if m]
            peak_count_displayed = len(peak_x)

            if peak_count_displayed > 0:
                peak_hover_texts = [
                    f'<b>Peak</b><br>'
                    f'Wavelength: {px:.1f} nm<br>'
                    f'Absorbance: {py:.4f} AU<br>'
                    f'Height: {ph:.4f} AU<br>' +
                    (f'Prominence: {pp:.4f} AU' if pp is not None else '') +
                    '<extra></extra>'
                    for px, py, ph, pp in zip(peak_x, peak_y, peak_heights, peak_prominences)
                ]

                # Add peak markers (white hollow circles)
                fig.add_trace(go.Scatter(
                    x=peak_x, y=peak_y, mode='markers',
                    marker=dict(
                        size=10, 
                        color='rgba(255,255,255,0)', 
                        symbol='circle', 
                        line=dict(width=2, color='rgba(255,255,255,0.9)')
                    ),
                    name=f'Peaks ({peak_count_displayed})', # Show count in range
                    hovertemplate=peak_hover_texts
                ))

                # Add peak vertical lines and labels - limited number
                max_annotations = 20
                for i in range(min(peak_count_displayed, max_annotations)):
                    # Add vertical dashed line across each peak
                    fig.add_shape(
                        type="line",
                        x0=peak_x[i], y0=0, 
                        x1=peak_x[i], y1=peak_y[i] * 1.1,  # Extend slightly above the peak
                        line=dict(color="rgba(255, 255, 255, 0.7)", width=1.5, dash="dash"),
                    )
                    
                    # Add text label at the top of each line
                    fig.add_annotation(
                        x=peak_x[i], y=peak_y[i] * 1.1,
                        text=f"{peak_x[i]:.1f}",
                        showarrow=False,
                        yshift=2,
                        bgcolor="rgba(0,0,0,0.5)",
                        bordercolor="rgba(255,255,255,0.7)",
                        font=dict(size=9, color="white"),
                        borderpad=2
                    )
                    
                if peak_count_displayed > max_annotations:
                     st.info(f"Displaying annotations for the first {max_annotations} peaks out of {peak_count_displayed} found in the selected range.")


    # Update title with peak count summary
    peak_source_label = st.session_state.get('peak_source_label', 'Raw')
    peak_count_text = f"{peak_count_displayed} peak{'s' if peak_count_displayed != 1 else ''} found in range ({peak_source_label})"
    if peak_count_total != peak_count_displayed:
         peak_count_text += f" (out of {peak_count_total} total)"
    fig.update_layout(title=f"{fig.layout.title.text}<br><sub>{peak_count_text}</sub>")

    return fig

# --- Helper Functions ---

def create_download_button(df, label, filename, key_suffix):
    """Creates a Streamlit download button for a DataFrame."""
    try:
        csv_buffer = io.StringIO()
        # Format floats precisely, handle potential NaN/Inf
        df.to_csv(csv_buffer, index=False, float_format='%.6g', na_rep='NaN')
        csv_data = csv_buffer.getvalue().encode('utf-8') # Encode to bytes
        st.download_button(
            label=f"📥 {label}", # Add icon
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"download_{key_suffix}" # Unique key
        )
        return True
    except Exception as e:
        st.error(f"Failed to prepare '{filename}' for download: {e}")
        return False

# --- Streamlit App UI ---

st.set_page_config(page_title="Absorbance Calculator", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings & Info")

    st.markdown("#### Measurement")
    path_length_mm = st.number_input(
        "Path Length (mm)",
        min_value=0.01, max_value=1000.0, value=10.0, step=0.1,
        help="Optical path length of the cuvette/sample holder in millimeters.",
        key="path_length_mm_widget"
    )
    path_length_cm = path_length_mm / 10.0 if path_length_mm is not None else None

    # --- Analysis Range ---
    st.markdown("#### Analysis Range")
    # Use session state to remember the range if files change
    if 'analysis_min_wl' not in st.session_state:
        st.session_state.analysis_min_wl = float(DEFAULT_SPECTRUM_MIN)
    if 'analysis_max_wl' not in st.session_state:
        st.session_state.analysis_max_wl = float(DEFAULT_SPECTRUM_MAX)

    range_col1, range_col2 = st.columns(2)
    with range_col1:
        analysis_min_wl = st.number_input("Min Wavelength (nm)",
                                          min_value=0.0, # Absolute minimum
                                          value=st.session_state.analysis_min_wl,
                                          step=1.0, format="%.1f",
                                          key="analysis_min_wl_widget",
                                          help="Minimum wavelength for plot X-axis and peak filtering.")
    with range_col2:
         analysis_max_wl = st.number_input("Max Wavelength (nm)",
                                           min_value=0.0,
                                           value=st.session_state.analysis_max_wl,
                                           step=1.0, format="%.1f",
                                           key="analysis_max_wl_widget",
                                           help="Maximum wavelength for plot X-axis and peak filtering.")

    # Store valid range in session state
    if analysis_min_wl < analysis_max_wl:
        st.session_state.analysis_min_wl = analysis_min_wl
        st.session_state.analysis_max_wl = analysis_max_wl
        plot_x_range_user = [analysis_min_wl, analysis_max_wl]
    else:
        st.warning("Min Wavelength must be less than Max Wavelength. Using full data range for plots.")
        plot_x_range_user = None # Indicate invalid range, plots will use data range

    st.markdown("#### Processing")
    # Smoothing Settings
    with st.expander("Smoothing (Savitzky-Golay)", expanded=False):
        apply_smoothing = st.checkbox("Apply Smoothing", value=False, key="apply_smoothing_cb",
                                      help="Smooth the calculated Absorbance data.")
        smooth_col1, smooth_col2 = st.columns(2)
        with smooth_col1:
            sg_window_length = st.number_input("Window Length", min_value=3, value=11, step=2,
                                               help="Must be odd.", disabled=not apply_smoothing, key="sg_window")
        with smooth_col2:
            sg_poly_order = st.number_input("Polynomial Order", min_value=1, value=2, step=1,
                                            help="Must be < Window Length.", disabled=not apply_smoothing, key="sg_poly")
        if apply_smoothing:
            if sg_window_length % 2 == 0: st.warning("Window Length must be odd.")
            if sg_poly_order >= sg_window_length: st.warning("Order must be < Window Length.")

    # Peak Detection Settings
    with st.expander("Peak Detection", expanded=True):
         peak_source = st.radio("Find peaks on:", ("Raw Absorbance", "Smoothed Absorbance"), index=0, horizontal=True,
                                key="peak_source_radio", help="Choose data for peak analysis.", disabled=not apply_smoothing)
         p_col1, p_col2, p_col3 = st.columns(3)
         with p_col1: peak_height = st.number_input("Min Height (AU)", min_value=0.0, value=0.1, step=0.01, format="%.3f", help="Minimum absorbance.", key="peak_height")
         with p_col2: peak_distance = st.number_input("Min Distance (nm)", min_value=0.1, value=15.0, step=0.5, format="%.1f", help="Minimum separation.", key="peak_distance")
         with p_col3: peak_prominence = st.number_input("Min Prominence (AU)", min_value=0.0, value=0.05, step=0.01, format="%.3f", help="Required elevation.", key="peak_prominence")

    # --- Information Expanders ---
    st.markdown("---")
    st.markdown("#### ℹ️ Information")
    with st.expander("File Format Instructions"): st.markdown(...) # Keep content as before
    with st.expander("About Calculations"): st.markdown(...) # Keep content as before

# --- Main Area ---
st.title("📊 Spectrophotometry Absorbance Calculator")
st.write("Upload spectral files, adjust settings in the sidebar, and view results below. Plots update automatically.")

# --- File Upload Section ---
st.subheader("1. Upload Spectral Data")
upload_cols = st.columns(3)
# ... (Keep file upload columns as before) ...
with upload_cols[0]:
    st.markdown("##### Reference (Blank)")
    reference_files = st.file_uploader("Upload reference/blank file(s)", type=["txt", "csv"],
                                       key="reference_uploader", accept_multiple_files=True, label_visibility="collapsed")
    if reference_files:
        st.info(f"{len(reference_files)} file(s) uploaded.")
        with st.spinner("Loading reference preview..."):
            df_ref_preview = read_spectral_file(reference_files[0])
            st.plotly_chart(create_preview_plot(df_ref_preview, "First Ref Preview", "dodgerblue"), use_container_width=True)

with upload_cols[1]:
    st.markdown("##### Sample")
    sample_files = st.file_uploader("Upload sample file(s)", type=["txt", "csv"],
                                    key="sample_uploader", accept_multiple_files=True, label_visibility="collapsed")
    if sample_files:
        st.info(f"{len(sample_files)} file(s) uploaded.")
        with st.spinner("Loading sample preview..."):
            df_sample_preview = read_spectral_file(sample_files[0])
            st.plotly_chart(create_preview_plot(df_sample_preview, "First Sample Preview", "mediumseagreen"), use_container_width=True)


with upload_cols[2]:
    st.markdown("##### Dark (Optional)")
    dark_files = st.file_uploader("Upload dark file(s)", type=["txt", "csv"],
                                  key="dark_uploader", accept_multiple_files=True, label_visibility="collapsed")
    if dark_files:
        st.info(f"{len(dark_files)} file(s) uploaded.")
        with st.spinner("Loading dark preview..."):
             df_dark_preview = read_spectral_file(dark_files[0])
             st.plotly_chart(create_preview_plot(df_dark_preview, "First Dark Preview", "black"), use_container_width=True)


# --- Processing and Display Section ---
st.subheader("2. Results")

if reference_files and sample_files:
    with st.spinner("Processing files and calculating results..."):
        df_reference = process_file_uploads(reference_files, "Reference")
        df_sample = process_file_uploads(sample_files, "Sample")
        df_dark = process_file_uploads(dark_files, "Dark") if dark_files else None

        df_result = None
        if df_reference is not None and df_sample is not None:
            if dark_files and df_dark is None:
                st.error("Dark files uploaded but failed processing. Cannot proceed.")
                st.stop()
            else:
                 df_result = calculate_absorbance(df_reference, df_sample, df_dark, path_length_cm)

    if df_result is not None:
        st.success("Calculations complete.")

        # Determine data range, but use user range for plotting if valid
        data_min_wl = df_result['Nanometers'].min()
        data_max_wl = df_result['Nanometers'].max()
        # Use user range if valid, otherwise fall back to data range
        final_plot_x_range = plot_x_range_user if plot_x_range_user else [data_min_wl, data_max_wl]


        # --- Apply Smoothing (if enabled and valid) ---
        smoothed_absorbance_values = None
        smoothing_error = False
        if apply_smoothing:
            # ... (smoothing logic remains the same) ...
            with st.spinner("Applying Savitzky-Golay smoothing..."):
                if sg_window_length % 2 != 0 and sg_poly_order < sg_window_length and len(df_result['Absorbance']) >= sg_window_length:
                    smoothed_absorbance_values = apply_savitzky_golay(
                        df_result['Absorbance'].values, sg_window_length, sg_poly_order
                    )
                    if smoothed_absorbance_values is None:
                        smoothing_error = True
                        st.warning("Smoothing failed during application.")
                    else:
                        df_result['Absorbance_Smoothed'] = smoothed_absorbance_values
                        st.info(f"Applied Savitzky-Golay smoothing (Window: {sg_window_length}, Order: {sg_poly_order}).")
                else:
                    smoothing_error = True
                    st.warning("Smoothing skipped due to invalid parameters or insufficient data.")


        # --- Find Peaks (on full spectrum) ---
        absorption_peaks_info = None
        peak_data_label = "Raw Absorbance"
        st.session_state['peak_source_label'] = peak_data_label

        with st.spinner("Finding absorption peaks..."):
            absorbance_for_peaks = df_result['Absorbance'].values
            if apply_smoothing and not smoothing_error and peak_source == "Smoothed Absorbance":
                 if smoothed_absorbance_values is not None:
                     absorbance_for_peaks = smoothed_absorbance_values
                     peak_data_label = "Smoothed Absorbance"
                     st.session_state['peak_source_label'] = peak_data_label
                 else: st.warning("Smoothed peaks requested, but smoothing failed. Using raw data.")
            elif apply_smoothing and peak_source == "Smoothed Absorbance" and smoothing_error:
                 st.warning("Smoothed peaks requested, but smoothing failed. Using raw data.")

            # Find peaks on the *full* selected spectrum
            absorption_peaks_info = find_absorption_peaks(
                df_result['Nanometers'].values, absorbance_for_peaks,
                peak_height, peak_distance, peak_prominence
            )
            if absorption_peaks_info is None: st.warning("Peak finding failed.")


        # --- Display Plots in Tabs (using final_plot_x_range) ---
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Absorbance Plot", "📊 Results Table", "📉 Counts Plots", "ℹ️ Peak Details"])

        with tab1: # Absorbance Plot
            st.plotly_chart(
                plot_absorbance(
                    df_result, path_length_mm, absorption_peaks_info,
                    smoothed_absorbance_values, final_plot_x_range # Pass user/data range
                ), use_container_width=True
            )

        with tab2: # Results Table (Full Data)
            st.markdown("#### Calculated Data (Full Range)")
            # ... (table display logic remains the same) ...
            display_df = df_result.copy()
            cols_to_show = ['Nanometers', 'Absorbance', 'Absorbance_per_cm']
            if 'Absorbance_Smoothed' in display_df.columns and display_df['Absorbance_Smoothed'].notna().any():
                 cols_to_show.insert(2, 'Absorbance_Smoothed')
            if 'Dark_Counts' in display_df.columns:
                 cols_to_show.extend(['Reference_Counts', 'Sample_Counts', 'Dark_Counts', 'Reference_Corrected', 'Sample_Corrected'])
            else:
                 cols_to_show.extend(['Reference_Counts', 'Sample_Counts', 'Reference_Corrected', 'Sample_Corrected'])
            display_df = display_df[cols_to_show]
            display_df.rename(columns={ 'Absorbance_per_cm': f'Absorbance (1cm Path)', 'Absorbance_Smoothed': 'Absorbance (Smoothed)', 'Reference_Counts': 'Ref Counts (Raw)', 'Sample_Counts': 'Sample Counts (Raw)', 'Dark_Counts': 'Dark Counts', 'Reference_Corrected': 'Ref Counts (Corrected)', 'Sample_Corrected': 'Sample Counts (Corrected)' }, inplace=True)
            st.dataframe(display_df, height=400, use_container_width=True)
            create_download_button(display_df, "Download Full Results", "absorbance_results_full.csv", "results_full")


        with tab3: # Counts Plots (using final_plot_x_range)
            st.markdown("#### Raw Counts")
            st.plotly_chart(plot_raw_counts(df_result, df_result.get('Dark_Counts'), final_plot_x_range), use_container_width=True)
            st.markdown("#### Dark-Corrected Counts")
            st.plotly_chart(plot_corrected_counts(df_result, final_plot_x_range), use_container_width=True)

        with tab4: # Peak Details (Filtered by final_plot_x_range)
            st.markdown(f"#### Peak Detection Summary ({peak_data_label})")
            if absorption_peaks_info is not None and len(absorption_peaks_info[0]) > 0:
                peak_indices, properties = absorption_peaks_info
                # Create DataFrame with all peaks first
                all_peaks_df = pd.DataFrame({
                    'Wavelength (nm)': df_result['Nanometers'].values[peak_indices],
                    'Absorbance (AU)': absorbance_for_peaks[peak_indices],
                    'Height (AU)': properties.get('peak_heights', np.nan),
                    'Prominence (AU)': properties.get('prominences', np.nan),
                    'Color Region': [get_color_name(wl) for wl in df_result['Nanometers'].values[peak_indices]]
                })
                all_peaks_df = all_peaks_df.sort_values(by='Wavelength (nm)')

                # Filter the DataFrame based on the selected range
                min_r, max_r = final_plot_x_range
                filtered_peaks_df = all_peaks_df[
                    (all_peaks_df['Wavelength (nm)'] >= min_r) &
                    (all_peaks_df['Wavelength (nm)'] <= max_r)
                ]

                st.markdown(f"Displaying peaks within **{min_r:.1f} nm** to **{max_r:.1f} nm** range:")
                st.dataframe(filtered_peaks_df, use_container_width=True)
                # Download button for the filtered peaks
                create_download_button(filtered_peaks_df, "Download Filtered Peak List", "absorption_peaks_filtered.csv", "peaks_filtered")
            elif absorption_peaks_info is not None:
                st.info("No peaks found matching the specified criteria in the full spectrum.")
            else:
                st.warning("Peak finding was not performed or encountered an error.")

    
    elif reference_files and sample_files: # Files uploaded, but df_result is None
        st.error("Calculation could not be completed. Check file formats and wavelength consistency.")

else:
    st.info("⬅️ Upload Reference (Blank) and Sample files to begin.")
