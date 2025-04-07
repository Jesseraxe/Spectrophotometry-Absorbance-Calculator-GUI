import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide") # Use wide layout for better plot visibility

st.title("Spectrum Viewer and Analyzer")

uploaded_file = st.file_uploader("Choose a spectrum file (CSV or TXT)", type=['csv', 'txt'])

if uploaded_file is not None:
    try:
        # Attempt to read the file using pandas
        # Try common separators and handle potential header issues
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['Wavelength/Wavenumber', 'Intensity'])
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.warning("Attempting to read without header...")
            uploaded_file.seek(0) # Reset file pointer
            df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['Wavelength/Wavenumber', 'Intensity'])

        # Ensure data is numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        if df.empty or df.shape[1] != 2:
            st.error("Could not find two numeric columns for Wavelength/Wavenumber and Intensity. Please ensure the file format is correct.")
        else:
            st.success(f"Successfully loaded file: {uploaded_file.name}")

            col1, col2 = st.columns([1, 2]) # Adjust column widths as needed

            with col1:
                st.subheader("Data Preview")
                st.dataframe(df.head())

                # Basic statistics (optional)
                st.subheader("Basic Statistics")
                st.write(df.describe())

            with col2:
                st.subheader("Spectrum Plot")
                # Create an interactive plot using Plotly
                fig = px.line(df, x=df.columns[0], y=df.columns[1], title=f"Spectrum: {uploaded_file.name}",
                              labels={'x': df.columns[0], 'y': df.columns[1]})
                fig.update_layout(xaxis_title=df.columns[0], yaxis_title='Intensity')
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred processing the file: {e}")
        st.error("Please ensure the file is a valid CSV or TXT file with two numeric columns (Wavelength/Wavenumber, Intensity).")

else:
    st.info("Upload a spectrum file to view its details and plot.")

# Remove the old placeholder lines
# st.write("This is a new blank page.") 