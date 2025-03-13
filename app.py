import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import plotly.graph_objects as go
from scipy.signal import find_peaks

# Set up the Streamlit page
st.set_page_config(page_title="Spectrophotometry Absorbance Calculator", layout="wide")
st.title("Spectrophotometry Absorbance Calculator")
st.write("Upload reference (blank) and sample files to calculate absorbance according to Beer-Lambert Law.")

# Create file uploaders for both reference and sample
st.subheader("File Upload")
col1, col2 = st.columns(2)

# Function to create a small preview plot for uploaded files
def create_preview_plot(df, title, color):
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

with col1:
    st.markdown("### Reference (Blank) File")
    reference_file = st.file_uploader("Upload reference/blank measurement", type=["txt", "csv"], key="reference")
    
    # Show preview if file is uploaded
    if reference_file is not None:
        try:
            df_reference_preview = pd.read_csv(reference_file, delim_whitespace=True)
            reference_file.seek(0)  # Reset file position for later reuse
            st.plotly_chart(create_preview_plot(df_reference_preview, "Reference Preview", "blue"), use_container_width=True)
        except Exception as e:
            st.error(f"Error previewing reference file: {e}")

with col2:
    st.markdown("### Sample File")
    sample_file = st.file_uploader("Upload sample measurement", type=["txt", "csv"], key="sample")
    
    # Show preview if file is uploaded
    if sample_file is not None:
        try:
            df_sample_preview = pd.read_csv(sample_file, delim_whitespace=True)
            sample_file.seek(0)  # Reset file position for later reuse
            st.plotly_chart(create_preview_plot(df_sample_preview, "Sample Preview", "green"), use_container_width=True)
        except Exception as e:
            st.error(f"Error previewing sample file: {e}")

# Optional dark spectrum for background correction
dark_expander = st.expander("Advanced: Dark Spectrum Correction (Optional)")
with dark_expander:
    st.markdown("""
    If you have a dark spectrum measurement (detector response with no light), 
    upload it below for more accurate absorbance calculation.
    """)
    dark_file = st.file_uploader("Upload dark spectrum (optional)", type=["txt", "csv"], key="dark")
    
    # Show preview if dark file is uploaded
    if dark_file is not None:
        try:
            df_dark_preview = pd.read_csv(dark_file, delim_whitespace=True)
            dark_file.seek(0)  # Reset file position for later reuse
            st.plotly_chart(create_preview_plot(df_dark_preview, "Dark Spectrum Preview", "black"), use_container_width=True)
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

# Define the visualization range
SPECTRUM_MIN = 340  # nm
SPECTRUM_MAX = 800  # nm

# Function to get visible spectrum color based on wavelength (nm)
def wavelength_to_rgb(wavelength):
    """Convert wavelength in nm to RGB color."""
    if 380 <= wavelength < 440:
        # Violet
        R = (440 - wavelength) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        # Blue
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        # Cyan
        R = 0.0
        G = 1.0
        B = (510 - wavelength) / (510 - 490)
    elif 510 <= wavelength < 580:
        # Green
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        # Yellow to Orange
        R = 1.0
        G = (645 - wavelength) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 780:
        # Red
        R = 1.0
        G = 0.0
        B = 0.0
    elif 340 <= wavelength < 380:
        # Near UV (deep purple)
        R = 0.4
        G = 0.0
        B = 0.8
    elif 780 < wavelength <= 800:
        # Near IR (deep red)
        R = 0.8
        G = 0.0
        B = 0.0
    else:
        # Outside our spectrum range
        R = 0.5
        G = 0.5
        B = 0.5
        
    # Intensity adjustment for edges of visible spectrum
    if 380 <= wavelength <= 780:
        if wavelength < 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif wavelength > 700:
            factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
        else:
            factor = 1.0
            
        R = R * factor
        G = G * factor
        B = B * factor
        
    return (R, G, B)

# New function to find peak indices on a smoothed curve
def find_absorption_peaks(wavelengths, absorbances, min_height=0.1, min_distance=15, min_prominence=0.05):
    """
    Find absorption peaks in spectral data.
    
    Parameters:
    wavelengths: array of wavelength values
    absorbances: array of absorbance values
    min_height: minimum height for a peak to be identified
    min_distance: minimum distance between peaks in data points
    min_prominence: required prominence of peaks
    
    Returns:
    list of (wavelength, absorbance) tuples for each peak found
    """
    # Convert min_distance from nm to number of data points
    avg_spacing = np.mean(np.diff(wavelengths))
    distance_points = int(min_distance / avg_spacing) if avg_spacing > 0 else 15
    
    # Find peaks on the absorbance data
    peak_indices, properties = find_peaks(
        absorbances, 
        height=min_height, 
        distance=distance_points,
        prominence=min_prominence
    )
    
    # Return wavelengths and absorbances at peak locations
    peaks = [(wavelengths[i], absorbances[i]) for i in peak_indices]
    return peaks

# Process the uploaded files
if reference_file is not None and sample_file is not None:
    try:
        # Read the reference file
        df_reference = pd.read_csv(reference_file, delim_whitespace=True)
        # Read the sample file
        df_sample = pd.read_csv(sample_file, delim_whitespace=True)
        
        # Check if dark file is provided
        if dark_file is not None:
            df_dark = pd.read_csv(dark_file, delim_whitespace=True)
            dark_counts = df_dark['Counts']
        else:
            # If no dark file, assume zero dark counts
            dark_counts = pd.Series(0, index=range(len(df_reference)))
        
        # Verify that the wavelengths match between reference and sample
        if not np.array_equal(df_reference['Nanometers'], df_sample['Nanometers']):
            st.error("Wavelength values in reference and sample files don't match. Please ensure both files have the same wavelength points.")
        else:
            # Create a new DataFrame for results
            df_result = pd.DataFrame()
            df_result['Nanometers'] = df_reference['Nanometers']
            df_result['Reference_Counts'] = df_reference['Counts']
            df_result['Sample_Counts'] = df_sample['Counts']
            
            # Apply dark correction if available
            reference_corrected = df_reference['Counts'] - dark_counts
            sample_corrected = df_sample['Counts'] - dark_counts
            
            # Ensure no negative or zero values after dark correction
            epsilon = 1e-10
            reference_corrected = reference_corrected.clip(lower=epsilon)
            sample_corrected = sample_corrected.clip(lower=epsilon)
            
            df_result['Reference_Corrected'] = reference_corrected
            df_result['Sample_Corrected'] = sample_corrected
            
            # Calculate absorbance: A = log10(I₀/I)
            df_result['Absorbance'] = np.log10(reference_corrected / sample_corrected)
            
            # Calculate absorbance adjusted for path length
            df_result['Absorbance_per_cm'] = df_result['Absorbance'] / path_length_cm
            
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
                fig1 = go.Figure()
                
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
                
                if dark_file is not None:
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
                
                fig1.update_layout(
                    title=f'Raw Counts vs Wavelength ({SPECTRUM_MIN}-{SPECTRUM_MAX} nm)',
                    xaxis_title='Wavelength (nm)',
                    yaxis_title='Counts',
                    xaxis=dict(range=[SPECTRUM_MIN, SPECTRUM_MAX]),
                    hovermode='closest',
                    height=600
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                fig2 = go.Figure()
                
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
                
                fig2.update_layout(
                    title=f'Dark-Corrected Counts vs Wavelength ({SPECTRUM_MIN}-{SPECTRUM_MAX} nm)',
                    xaxis_title='Wavelength (nm)',
                    yaxis_title='Counts (Dark Corrected)',
                    xaxis=dict(range=[SPECTRUM_MIN, SPECTRUM_MAX]),
                    hovermode='closest',
                    height=600
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                # Create a colorful spectral line for absorbance
                fig3 = go.Figure()
                
                # Create segments of the wavelength range to color appropriately
                wavelengths = df_plot['Nanometers'].tolist()
                absorbances = df_plot['Absorbance'].tolist()
                
                # Create colored segments for the entire range
                for j in range(len(wavelengths) - 1):
                    w1, w2 = wavelengths[j], wavelengths[j+1]
                    a1, a2 = absorbances[j], absorbances[j+1]
                    
                    # Get color based on average wavelength of segment
                    avg_wavelength = (w1 + w2) / 2
                    rgb = wavelength_to_rgb(avg_wavelength)
                    color = f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'
                    
                    fig3.add_trace(go.Scatter(
                        x=[w1, w2],
                        y=[a1, a2],
                        mode='lines',
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Add a line for hover info
                fig3.add_trace(go.Scatter(
                    x=df_plot['Nanometers'],
                    y=df_plot['Absorbance'],
                    mode='lines',
                    name='Absorbance',
                    line=dict(color='rgba(0,0,0,0.2)', width=1),
                    hovertemplate='Wavelength: %{x:.1f} nm<br>Absorbance: %{y:.4f}<extra></extra>'
                ))
                
                # Find absorption peaks
                absorption_peaks = find_absorption_peaks(
                    df_plot['Nanometers'].values, 
                    df_plot['Absorbance'].values,
                    min_height=peak_height,
                    min_distance=peak_distance,
                    min_prominence=peak_prominence
                )
                
                # Add peak markers and labels
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
                for i, (x, y) in enumerate(zip(peak_x, peak_y)):
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
                
                fig3.update_layout(
                    title=f'Absorbance vs Wavelength ({SPECTRUM_MIN}-{SPECTRUM_MAX} nm, path length = {path_length} mm)<br><sub>{peak_count_text}</sub>',
                    xaxis_title='Wavelength (nm)',
                    yaxis_title='Absorbance',
                    xaxis=dict(range=[SPECTRUM_MIN, SPECTRUM_MAX]),
                    hovermode='closest',
                    height=600
                )
                
                # Add a vertical band showing the visible spectrum range
                fig3.add_shape(
                    type="rect",
                    x0=380, y0=0,
                    x1=780, y1=1,
                    yref="paper",
                    fillcolor="rgba(200,200,200,0.1)",
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
    - Both reference and sample files must have measurements at the same wavelengths
    - The files can be .txt or .csv files
    """)