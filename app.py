import streamlit as st
import numpy as np
import pandas as pd
import io
import plotly.graph_objects as go
from scipy.signal import find_peaks
from functools import lru_cache

# Set page configuration
st.set_page_config(page_title="Spectrophotometry Absorbance Calculator", layout="wide")
st.title("Spectrophotometry Absorbance Calculator")
st.write("Upload reference (blank) and sample files to calculate absorbance according to Beer-Lambert Law.")

# Define constants
SPECTRUM_MIN = 340  # nm
SPECTRUM_MAX = 800  # nm
EPSILON = 1e-10  # Small value to prevent division by zero or log of zero

# Function to read a dataframe from a file - with caching to improve performance
@st.cache_data
def read_spectral_file(file):
    try:
        return pd.read_csv(file, delim_whitespace=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Function to create a small preview plot for uploaded files
def create_preview_plot(df, title, color):
    if df is None:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Nanometers'],
        y=df['Counts'],
        mode='lines',
        line=dict(color=color)
    ))
    fig.update_layout(
        title=title,
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Wavelength (nm)",
        yaxis_title="Counts"
    )
    return fig

# Function to average multiple dataframes with validation
def average_dataframes(dataframes):
    if not dataframes:
        return None
    
    if any(df is None for df in dataframes):
        st.error("One or more files could not be processed.")
        return None
    
    # Check if all dataframes have the same wavelength points
    wavelengths = dataframes[0]['Nanometers'].values
    for df in dataframes[1:]:
        if not np.array_equal(df['Nanometers'].values, wavelengths):
            st.error("Wavelength values in files don't match. Please ensure all files have the same wavelength points.")
            return None
    
    # Create a new dataframe with the wavelengths
    result = pd.DataFrame({'Nanometers': wavelengths})
    
    # Average the counts
    counts_arrays = [df['Counts'].values for df in dataframes]
    result['Counts'] = np.mean(counts_arrays, axis=0)
    
    return result

# Function to process uploaded files with progress indication
def process_file_uploads(files, file_type="data"):
    if not files:
        return None
    
    dataframes = []
    
    # Show a progress bar for file processing
    progress_text = f"Processing {file_type} files..."
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        df = read_spectral_file(file)
        if df is not None:
            dataframes.append(df)
            file.seek(0)  # Reset file position for potential later use
        progress_bar.progress((i + 1) / len(files))
    
    progress_bar.empty()  # Remove progress bar when done
    
    if not dataframes:
        return None
        
    return average_dataframes(dataframes)

# Function to get visible spectrum color based on wavelength (nm) - optimized with caching
@st.cache_data
def wavelength_to_rgb(wavelength):
    """Convert wavelength in nm to RGB color using cached computation."""
    # Define color ranges
    if 380 <= wavelength < 440:  # Violet
        R, G, B = (440 - wavelength) / 60, 0.0, 1.0
    elif 440 <= wavelength < 490:  # Blue
        R, G, B = 0.0, (wavelength - 440) / 50, 1.0
    elif 490 <= wavelength < 510:  # Cyan
        R, G, B = 0.0, 1.0, (510 - wavelength) / 20
    elif 510 <= wavelength < 580:  # Green
        R, G, B = (wavelength - 510) / 70, 1.0, 0.0
    elif 580 <= wavelength < 645:  # Yellow to Orange
        R, G, B = 1.0, (645 - wavelength) / 65, 0.0
    elif 645 <= wavelength <= 780:  # Red
        R, G, B = 1.0, 0.0, 0.0
    elif 340 <= wavelength < 380:  # Near UV (deep purple)
        R, G, B = 0.4, 0.0, 0.8
    elif 780 < wavelength <= 800:  # Near IR (deep red)
        R, G, B = 0.8, 0.0, 0.0
    else:  # Outside spectrum range
        R, G, B = 0.5, 0.5, 0.5
    
    # Apply intensity adjustment for edges of visible spectrum
    if 380 <= wavelength <= 780:
        if wavelength < 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / 40
        elif wavelength > 700:
            factor = 0.3 + 0.7 * (780 - wavelength) / 80
        else:
            factor = 1.0
        
        R, G, B = R * factor, G * factor, B * factor
    
    return (R, G, B)

# Function to convert wavelength to visible color name
@st.cache_data
def get_color_name(wavelength):
    if 380 <= wavelength < 450:
        return "Violet"
    elif 450 <= wavelength < 485:
        return "Blue"
    elif 485 <= wavelength < 500:
        return "Cyan"
    elif 500 <= wavelength < 565:
        return "Green"
    elif 565 <= wavelength < 590:
        return "Yellow"
    elif 590 <= wavelength < 625:
        return "Orange"
    elif 625 <= wavelength <= 780:
        return "Red"
    elif wavelength < 380:
        return "Near UV"
    else:
        return "Near IR"

# Function to find absorption peaks
def find_absorption_peaks(wavelengths, absorbances, min_height=0.1, min_distance=15, min_prominence=0.05):
    """Find absorption peaks in spectral data."""
    # Convert min_distance from nm to number of data points
    avg_spacing = np.mean(np.diff(wavelengths))
    distance_points = int(min_distance / avg_spacing) if avg_spacing > 0 else 15
    
    # Find peaks on the absorbance data
    peak_indices, _ = find_peaks(
        absorbances, 
        height=min_height, 
        distance=distance_points,
        prominence=min_prominence
    )
    
    # Return wavelengths and absorbances at peak locations
    return [(wavelengths[i], absorbances[i]) for i in peak_indices]

# Function to create a base plot configuration
def create_base_plot(title, x_label, y_label):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(range=[SPECTRUM_MIN, SPECTRUM_MAX]),
        hovermode='closest',
        height=600
    )
    return fig

# Set up the UI layout
st.subheader("File Upload")
col1, col2 = st.columns(2)

# File upload sections
with col1:
    st.markdown("### Reference (Blank) Files")
    reference_files = st.file_uploader("Upload reference/blank measurements", type=["txt", "csv"], key="reference", accept_multiple_files=True)
    
    if reference_files:
        st.info(f"{len(reference_files)} reference files uploaded. These will be averaged.")
        
        # Preview first file
        if len(reference_files) > 0:
            try:
                df_reference_preview = read_spectral_file(reference_files[0])
                reference_files[0].seek(0)
                if df_reference_preview is not None:
                    st.plotly_chart(create_preview_plot(df_reference_preview, "First Reference File Preview", "blue"), use_container_width=True)
            except Exception as e:
                st.error(f"Error previewing reference file: {e}")

with col2:
    st.markdown("### Sample Files")
    sample_files = st.file_uploader("Upload sample measurements", type=["txt", "csv"], key="sample", accept_multiple_files=True)
    
    if sample_files:
        st.info(f"{len(sample_files)} sample files uploaded. These will be averaged.")
        
        # Preview first file
        if len(sample_files) > 0:
            try:
                df_sample_preview = read_spectral_file(sample_files[0])
                sample_files[0].seek(0)
                if df_sample_preview is not None:
                    st.plotly_chart(create_preview_plot(df_sample_preview, "First Sample File Preview", "green"), use_container_width=True)
            except Exception as e:
                st.error(f"Error previewing sample file: {e}")

# Optional dark spectrum for background correction
dark_expander = st.expander("Advanced: Dark Spectrum Correction (Optional)")
with dark_expander:
    st.markdown("""
    If you have a dark spectrum measurement (detector response with no light), 
    upload it below for more accurate absorbance calculation.
    """)
    dark_files = st.file_uploader("Upload dark spectrum files (optional)", type=["txt", "csv"], key="dark", accept_multiple_files=True)
    
    if dark_files:
        st.info(f"{len(dark_files)} dark files uploaded. These will be averaged.")
        
        # Preview first file
        if len(dark_files) > 0:
            try:
                df_dark_preview = read_spectral_file(dark_files[0])
                dark_files[0].seek(0)
                if df_dark_preview is not None:
                    st.plotly_chart(create_preview_plot(df_dark_preview, "First Dark Spectrum Preview", "black"), use_container_width=True)
            except Exception as e:
                st.error(f"Error previewing dark file: {e}")

# Add path length input (default 10mm)
path_length = st.number_input("Path length (mm):", min_value=0.1, max_value=100.0, value=10.0, step=0.1)
path_length_cm = path_length / 10  # Convert to cm for calculations

# Add peak detection settings
peak_settings_expander = st.expander("Peak Detection Settings")
with peak_settings_expander:
    col1, col2, col3 = st.columns(3)
    with col1:
        peak_height = st.number_input("Minimum peak height:", min_value=0.01, max_value=5.0, value=0.1, step=0.01, 
                                     help="Minimum height required for a peak to be identified")
    with col2:
        peak_distance = st.number_input("Minimum peak distance (nm):", min_value=5, max_value=100, value=15, step=5,
                                       help="Minimum distance between peaks in nanometers")
    with col3:
        peak_prominence = st.number_input("Minimum peak prominence:", min_value=0.01, max_value=2.0, value=0.05, step=0.01,
                                         help="Required prominence of peaks (higher values give fewer peaks)")

# Process the uploaded files
if reference_files and sample_files:
    try:
        with st.spinner("Processing files..."):
            # Process files using our optimized function
            df_reference = process_file_uploads(reference_files, "reference")
            if df_reference is None:
                st.error("Could not process reference files.")
                st.stop()
                
            df_sample = process_file_uploads(sample_files, "sample")
            if df_sample is None:
                st.error("Could not process sample files.")
                st.stop()
            
            # Process dark files if provided
            if dark_files:
                df_dark = process_file_uploads(dark_files, "dark")
                if df_dark is None:
                    st.error("Could not process dark files.")
                    st.stop()
                
                dark_counts = df_dark['Counts']
            else:
                # If no dark file, assume zero dark counts
                dark_counts = pd.Series(0, index=range(len(df_reference)))
            
            # Verify wavelength alignment between reference and sample
            if not np.array_equal(df_reference['Nanometers'], df_sample['Nanometers']):
                st.error("Wavelength values in reference and sample files don't match.")
                st.stop()
                
            # Create results dataframe
            df_result = pd.DataFrame({
                'Nanometers': df_reference['Nanometers'],
                'Reference_Counts': df_reference['Counts'],
                'Sample_Counts': df_sample['Counts']
            })
            
            # Apply dark correction
            reference_corrected = (df_reference['Counts'] - dark_counts).clip(lower=EPSILON)
            sample_corrected = (df_sample['Counts'] - dark_counts).clip(lower=EPSILON)
            
            df_result['Reference_Corrected'] = reference_corrected
            df_result['Sample_Corrected'] = sample_corrected
            
            # Calculate absorbance: A = log10(I₀/I)
            df_result['Absorbance'] = np.log10(reference_corrected / sample_corrected)
            
            # Calculate absorbance adjusted for path length
            df_result['Absorbance_per_cm'] = df_result['Absorbance'] / path_length_cm
            
            # Add averaging info
            averaging_info = f"""
            ### Averaging Information
            - {len(reference_files)} reference files averaged
            - {len(sample_files)} sample files averaged
            """
            if dark_files:
                averaging_info += f"- {len(dark_files)} dark files averaged"
            
            st.markdown(averaging_info)
            
            # Show the processed data
            st.subheader("Processed Data")
            st.dataframe(df_result)
            
            # Create interactive plots with Plotly
            st.subheader("Visualization")
            tab1, tab2, tab3 = st.tabs(["Raw Counts", "Corrected Counts", "Absorbance"])
            
            # Filter data to the required wavelength range for visualization
            df_plot = df_result[(df_result['Nanometers'] >= SPECTRUM_MIN) & 
                                (df_result['Nanometers'] <= SPECTRUM_MAX)].copy()
            
            with tab1:
                fig1 = create_base_plot(
                    title=f'Raw Counts vs Wavelength ({SPECTRUM_MIN}-{SPECTRUM_MAX} nm)',
                    x_label='Wavelength (nm)',
                    y_label='Counts'
                )
                
                fig1.add_trace(go.Scatter(
                    x=df_plot['Nanometers'], 
                    y=df_plot['Reference_Counts'],
                    mode='lines',
                    name='Reference',
                    line=dict(color='blue'),
                    hovertemplate='Wavelength: %{x:.1f} nm<br>Counts: %{y:.2f}<extra></extra>'
                ))
                
                fig1.add_trace(go.Scatter(
                    x=df_plot['Nanometers'], 
                    y=df_plot['Sample_Counts'],
                    mode='lines',
                    name='Sample',
                    line=dict(color='green'),
                    hovertemplate='Wavelength: %{x:.1f} nm<br>Counts: %{y:.2f}<extra></extra>'
                ))
                
                if dark_files:
                    # Filter dark counts to the same range
                    dark_counts_plot = dark_counts.iloc[df_plot.index]
                    fig1.add_trace(go.Scatter(
                        x=df_plot['Nanometers'], 
                        y=dark_counts_plot,
                        mode='lines',
                        name='Dark',
                        line=dict(color='black'),
                        hovertemplate='Wavelength: %{x:.1f} nm<br>Counts: %{y:.2f}<extra></extra>'
                    ))
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                fig2 = create_base_plot(
                    title=f'Dark-Corrected Counts vs Wavelength ({SPECTRUM_MIN}-{SPECTRUM_MAX} nm)',
                    x_label='Wavelength (nm)',
                    y_label='Counts (Dark Corrected)'
                )
                
                fig2.add_trace(go.Scatter(
                    x=df_plot['Nanometers'], 
                    y=df_plot['Reference_Corrected'],
                    mode='lines',
                    name='Reference (Corrected)',
                    line=dict(color='blue'),
                    hovertemplate='Wavelength: %{x:.1f} nm<br>Counts: %{y:.2f}<extra></extra>'
                ))
                
                fig2.add_trace(go.Scatter(
                    x=df_plot['Nanometers'], 
                    y=df_plot['Sample_Corrected'],
                    mode='lines',
                    name='Sample (Corrected)',
                    line=dict(color='green'),
                    hovertemplate='Wavelength: %{x:.1f} nm<br>Counts: %{y:.2f}<extra></extra>'
                ))
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                # Create the absorbance plot
                fig3 = create_base_plot(
                    title=f'Absorbance vs Wavelength ({SPECTRUM_MIN}-{SPECTRUM_MAX} nm, path length = {path_length} mm)',
                    x_label='Wavelength (nm)',
                    y_label='Absorbance'
                )
                
                wavelengths = df_plot['Nanometers'].values
                absorbances = df_plot['Absorbance'].values
                
                # Create colored segments for the entire range - optimized to reduce iterations
                # Group wavelengths in chunks to reduce number of traces
                chunk_size = max(1, len(wavelengths) // 100)  # Limit to around 100 segments
                for j in range(0, len(wavelengths) - chunk_size, chunk_size):
                    end_idx = min(j + chunk_size, len(wavelengths) - 1)
                    chunk_wavelengths = wavelengths[j:end_idx+1]
                    chunk_absorbances = absorbances[j:end_idx+1]
                    
                    # Get color based on average wavelength of chunk
                    avg_wavelength = np.mean(chunk_wavelengths)
                    rgb = wavelength_to_rgb(avg_wavelength)
                    color = f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
                    
                    fig3.add_trace(go.Scatter(
                        x=chunk_wavelengths,
                        y=chunk_absorbances,
                        mode='lines',
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Add a line for hover info
                fig3.add_trace(go.Scatter(
                    x=wavelengths,
                    y=absorbances,
                    mode='lines',
                    name='Absorbance',
                    line=dict(color='rgba(0,0,0,0.2)', width=1),
                    hovertemplate='Wavelength: %{x:.1f} nm<br>Absorbance: %{y:.4f}<extra></extra>'
                ))
                
                # Find absorption peaks
                absorption_peaks = find_absorption_peaks(
                    wavelengths, 
                    absorbances,
                    min_height=peak_height,
                    min_distance=peak_distance,
                    min_prominence=peak_prominence
                )
                
                # Add peak markers and labels
                if absorption_peaks:
                    peak_x = [peak[0] for peak in absorption_peaks]
                    peak_y = [peak[1] for peak in absorption_peaks]
                    
                    # Add peak points
                    fig3.add_trace(go.Scatter(
                        x=peak_x,
                        y=peak_y,
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='diamond'),
                        name='Absorption Peaks',
                        hovertemplate='Peak: %{x:.1f} nm<br>Absorbance: %{y:.4f}<extra></extra>'
                    ))
                    
                    # Add peak labels
                    for x, y in zip(peak_x, peak_y):
                        fig3.add_annotation(
                            x=x, y=y + 0.05,
                            text=f"{x:.1f} nm",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1, 
                            arrowwidth=1,
                            arrowcolor="black",
                            ax=0,
                            ay=-30
                        )
                
                # Show number of peaks found
                peak_count_text = f"{len(absorption_peaks)} absorption peak{'s' if len(absorption_peaks) != 1 else ''} detected"
                fig3.update_layout(title=f"{fig3.layout.title.text}<br><sub>{peak_count_text}</sub>")
                
                # Add a vertical band showing the visible spectrum range
                fig3.add_shape(
                    type="rect",
                    x0=380, y0=0,
                    x1=780, y1=1,
                    yref="paper",
                    fillcolor="rgba(200,200,200,0.005)",
                    line=dict(width=0),
                    layer="below"
                )
                
                # Add annotation for visible spectrum
                fig3.add_annotation(
                    x=(380+780)/2, y=0.03,
                    yref="paper",
                    text="Visible Spectrum (380-780 nm)",
                    showarrow=False,
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Create a table of detected peaks
                if absorption_peaks:
                    st.subheader("Absorption Peaks")
                    
                    # Create a dataframe for the peaks
                    peaks_df = pd.DataFrame(absorption_peaks, columns=['Wavelength (nm)', 'Absorbance'])
                    peaks_df = peaks_df.sort_values(by='Wavelength (nm)')
                    
                    # Add a column for corresponding visible colors
                    peaks_df['Spectral Region'] = peaks_df['Wavelength (nm)'].apply(get_color_name)
                    
                    # Format the dataframe
                    peaks_df['Wavelength (nm)'] = peaks_df['Wavelength (nm)'].round(1)
                    peaks_df['Absorbance'] = peaks_df['Absorbance'].round(4)
                    
                    st.dataframe(peaks_df)
                    
                    # Add download button for peak data
                    csv_peaks = io.StringIO()
                    peaks_df.to_csv(csv_peaks, index=False)
                    peaks_csv_data = csv_peaks.getvalue()
                    
                    st.download_button(
                        label="Download Peaks CSV",
                        data=peaks_csv_data,
                        file_name="absorbance_peaks.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No absorption peaks detected with current settings. Try adjusting the peak detection parameters.")
                
                # Download buttons for the processed data
                st.subheader("Download Results")
                
                # CSV download
                csv_buffer = io.StringIO()
                df_result.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="absorbance_data.csv",
                    mime="text/csv"
                )
                
                # Add information about Beer-Lambert Law
                with st.expander("Beer-Lambert Law Information"):
                    st.write(f"""
                    ### Beer-Lambert Law
                    
                    The Beer-Lambert Law states that absorbance is directly proportional to the concentration of a solution and the path length:
                    
                    **A = ε × c × l**
                        
                    Where:
                    - A is the absorbance (dimensionless)
                    - ε is the molar absorptivity (L·mol⁻¹·cm⁻¹)
                    - c is the concentration (mol·L⁻¹)
                    - l is the path length (cm)
                    
                    For spectrophotometric measurements:
                    - A = log₁₀(I₀/I)
                    - I₀ is the intensity of light passing through the reference
                    - I is the intensity of light passing through the sample
                    
                    In this application:
                    - Path length = {path_length} mm = {path_length_cm} cm
                    - The 'Absorbance' column shows the raw calculated values
                    - The 'Absorbance_per_cm' column normalizes to a 1 cm path length
                    """)
                
                # Add section about averaging multiple scans
                with st.expander("Benefits of Averaging Multiple Scans"):
                    st.write("""
                    ### Why Average Multiple Scans?
                    
                    Averaging multiple scans provides several advantages for spectrophotometric measurements:
                    
                    1. **Improved Signal-to-Noise Ratio (SNR)**: Averaging N scans improves SNR by a factor of √N. For example, averaging 4 scans doubles the SNR.
                    
                    2. **Reduced Random Errors**: Fluctuations due to electronic noise, stray light, or other random factors are minimized.
                    
                    3. **Better Precision**: Averaging produces more reliable and reproducible results, especially for low-concentration samples.
                    
                    4. **Smoother Spectra**: The resulting spectrum has fewer artifacts and is easier to interpret.
                    
                    In this application, you can upload multiple files for reference, sample, and dark measurements. The application automatically averages the spectra before calculating absorbance.
                    """)
        
    except Exception as e:
        st.error(f"Error processing files: {e}")
        st.write("Please ensure your files have the correct format with 'Nanometers' and 'Counts' columns separated by spaces.")

# Instructions for file format
with st.expander("File Format Instructions"):
    st.write("""
    Your input files should be in the following format:
    ```
    Nanometers Counts
    348.7 0.0
    349.2 0.0
    ...
    ```
    
    - The first line should contain the column headers: 'Nanometers' and 'Counts'
    - Each subsequent line should contain the wavelength and count values separated by space(s)
    - All files must have measurements at the same wavelengths
    - The files can be .txt or .csv files
    
    ### Multiple File Upload
    
    - You can now upload multiple files for reference, sample, and dark measurements
    - The application will average the spectra from multiple files
    - This helps reduce noise and improve measurement accuracy
    - All files being averaged must have the same wavelength points
    """)
