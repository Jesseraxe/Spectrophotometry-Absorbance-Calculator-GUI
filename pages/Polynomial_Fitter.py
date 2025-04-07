import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks # May need this later for peak finding
# Consider adding numpy.polyfit later

st.set_page_config(page_title="Polynomial Fitter", layout="wide")

st.title("📈 Polynomial Fitter for Absorbance Peaks")

st.write("""
Upload one or more 'Full Results' CSV files (exported from the Absorbance Calculator)
to analyze the linearity of absorbance peaks using polynomial fitting.
""")

uploaded_files = st.file_uploader(
    "Upload Absorbance Data CSV Files",
    type=["csv"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} file(s)...")
    # Placeholder for file reading and processing logic
    all_data = {}
    error_files = []

    for uploaded_file in uploaded_files:
        try:
            # Make sure to handle potential header issues or format variations
            # Assuming the CSVs have 'Wavelength (nm)' and 'Absorbance (AU)' columns
            df = pd.read_csv(uploaded_file)
            if 'Wavelength (nm)' not in df.columns or 'Absorbance (AU)' not in df.columns:
                 st.warning(f"File '{uploaded_file.name}' is missing required columns ('Wavelength (nm)', 'Absorbance (AU)'). Skipping.")
                 error_files.append(uploaded_file.name)
                 continue
            # Basic validation - ensure numeric types
            df['Wavelength (nm)'] = pd.to_numeric(df['Wavelength (nm)'], errors='coerce')
            df['Absorbance (AU)'] = pd.to_numeric(df['Absorbance (AU)'], errors='coerce')
            df.dropna(subset=['Wavelength (nm)', 'Absorbance (AU)'], inplace=True)
            if df.empty:
                st.warning(f"File '{uploaded_file.name}' contained no valid numeric data after cleaning. Skipping.")
                error_files.append(uploaded_file.name)
                continue

            all_data[uploaded_file.name] = df.sort_values(by='Wavelength (nm)').reset_index(drop=True)
        except Exception as e:
            st.error(f"Error reading file '{uploaded_file.name}': {e}")
            error_files.append(uploaded_file.name)

    # Remove error files from the list of processable files
    valid_files = [f for f in uploaded_files if f.name not in error_files]

    if not all_data:
        st.warning("No valid data could be loaded from the uploaded files.")
    else:
        st.success(f"Successfully loaded data from {len(all_data)} file(s).")

        # --- Input Concentrations ---
        st.subheader("Concentrations")
        st.write("Enter the analyte concentration (PPM) for each uploaded file.")
        concentrations = {}
        # Use columns for better layout if many files are uploaded
        num_files = len(all_data)
        num_cols = min(num_files, 4) # Max 4 columns
        cols = st.columns(num_cols)
        file_counter = 0
        for filename in all_data.keys():
            col_index = file_counter % num_cols
            with cols[col_index]:
                 # Use a default value like 1.0 or 0.0, make it clear it needs input
                 concentrations[filename] = st.number_input(
                    f"Conc. for {filename}",
                    min_value=0.0,
                    value=1.0, # Or 0.0, depending on expected workflow
                    step=0.1,
                    format="%.2f",
                    key=f"conc_{filename}", # Unique key per file
                    help=f"Enter the concentration corresponding to the data in {filename}"
                )
            file_counter += 1

        # --- Next Steps: Peak Selection UI ---
        st.subheader("1. Select Peak Wavelength")
        # We need a way for the user to specify which peak to analyze across all datasets.
        # Option 1: Automatically find the highest peak in the first file?
        # Option 2: Let the user input a target wavelength.
        # Let's start with user input.

        target_wavelength = st.number_input(
            "Enter Target Peak Wavelength (nm)",
            min_value=0.0,
            value=400.0, # Default guess
            step=1.0,
            format="%.1f",
            help="Enter the approximate wavelength of the peak you want to analyze across all datasets."
        )

        # --- Find Peak Absorbance for each concentration ---
        peak_data = {}
        analysis_results = []
        valid_data_for_fitting = True

        for filename, df in all_data.items():
            # Find the index of the wavelength closest to the target
            closest_wl_index = (df['Wavelength (nm)'] - target_wavelength).abs().idxmin()
            actual_wl = df.loc[closest_wl_index, 'Wavelength (nm)']
            peak_abs = df.loc[closest_wl_index, 'Absorbance (AU)']
            concentration = concentrations.get(filename) # Get concentration entered by user

            if concentration is None:
                st.warning(f"Concentration not found for file {filename}. Skipping this file for fitting.")
                valid_data_for_fitting = False
                continue # Skip if concentration wasn't entered or found

            analysis_results.append({
                'Filename': filename,
                'Target Wavelength (nm)': target_wavelength,
                'Actual Wavelength (nm)': actual_wl,
                'Peak Absorbance (AU)': peak_abs,
                'Concentration': concentration
            })

        if not analysis_results:
             st.warning("No valid data points (with concentrations) found near the target wavelength to perform fitting.")
             valid_data_for_fitting = False

        # --- Perform Polynomial Fitting ---
        st.subheader("2. Polynomial Fit Results")

        if valid_data_for_fitting and len(analysis_results) > 1: # Need at least 2 points for fitting
            results_df = pd.DataFrame(analysis_results)

            # Prepare data for numpy polyfit
            x_conc = results_df['Concentration'].values
            y_abs = results_df['Peak Absorbance (AU)'].values

            # Calculate Linear Fit (Degree 1)
            coeffs_deg1 = np.polyfit(x_conc, y_abs, 1)
            poly_deg1 = np.poly1d(coeffs_deg1)
            y_pred_deg1 = poly_deg1(x_conc)

            # Calculate R-squared for Degree 1
            ss_res_deg1 = np.sum((y_abs - y_pred_deg1) ** 2)
            ss_tot_deg1 = np.sum((y_abs - np.mean(y_abs)) ** 2)
            r_squared_deg1 = 1 - (ss_res_deg1 / ss_tot_deg1) if ss_tot_deg1 > 0 else 0

            st.markdown("#### Linear Fit (Degree 1)")
            st.metric("R² (Linear)", f"{r_squared_deg1:.4f}")
            st.write(f"Equation: Absorbance = {coeffs_deg1[0]:.4f} * Concentration + {coeffs_deg1[1]:.4f}")

            # --- Prepare for Plotting ---
            plot_fig = go.Figure()

            # Add scatter points (Concentration vs. Absorbance)
            plot_fig.add_trace(go.Scatter(
                x=x_conc,
                y=y_abs,
                mode='markers',
                name='Data Points',
                marker=dict(size=8),
                text=[f"File: {row['Filename']}<br>Conc: {row['Concentration']:.2f}<br>Abs: {row['Peak Absorbance (AU)']:.4f}" 
                      for index, row in results_df.iterrows()],
                hoverinfo='text'
            ))

            # Add linear fit line
            # Generate smooth line for plotting
            conc_range = np.linspace(x_conc.min(), x_conc.max(), 100)
            abs_fit_line = poly_deg1(conc_range)
            plot_fig.add_trace(go.Scatter(
                x=conc_range,
                y=abs_fit_line,
                mode='lines',
                name=f'Linear Fit (R²={r_squared_deg1:.4f})',
                line=dict(color='red', dash='dash')
            ))

            plot_fig.update_layout(
                title="Concentration vs. Peak Absorbance and Linear Fit",
                xaxis_title="Concentration (Units as entered)",
                yaxis_title="Peak Absorbance (AU)",
                hovermode='closest'
            )

            st.plotly_chart(plot_fig, use_container_width=True)

             # Display the extracted peak data
            with st.expander("View Peak Data Used for Fitting"):
                st.dataframe(results_df[['Filename', 'Concentration', 'Actual Wavelength (nm)', 'Peak Absorbance (AU)']])

        elif len(analysis_results) <= 1:
             st.warning(f"Need at least 2 data points with concentrations to perform fitting. Found {len(analysis_results)}.")
        
        # Display loaded data (optional, maybe in an expander) - Moved this down
        with st.expander("View Full Loaded Data"):
            for filename, df in all_data.items():
                st.markdown(f"**{filename}**")
                st.dataframe(df) # Show full dataframe


else:
    st.info("Upload CSV files containing absorbance data and enter concentrations to begin analysis.")

# --- TODO ---
# 1. Implement logic to find the actual peak closest to `target_wavelength` in each dataset.
# 2. Define a window around the peak for fitting.
# 3. Implement polynomial fitting (e.g., degrees 1-5) for the windowed data.
# 4. Calculate R-squared for each fit.
# 5. Determine the 'best' linear fit (e.g., highest R^2 for degree 1, or specific criteria).
# 6. Display fitting results (coefficients, R^2) in a table.
# 7. Add plots showing the data window, the peak, and the fitted polynomials.
