import streamlit as st
import subprocess

st.title("Stock Predictor")

start_button = st.button("Start Stock Predictor")

if start_button:
    command = ["python", "-u", "src/stock_predictor/main.py"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    while process.poll() is None:
        output = process.stdout.readline()
        if not output:
            continue
        st.write(output.strip())
