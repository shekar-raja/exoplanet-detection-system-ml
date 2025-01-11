import streamlit as st
import pandas as pd
import torch

model = torch.load("models/mlp_exoplanet_classifier.pth")

st.write("""
# My first app
Hello *world!*
""")

# df = pd.read_csv("dataset/exoTest.csv")
# st.line_chart(df)