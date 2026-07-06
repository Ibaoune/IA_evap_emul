import os

models = ["XGB", "MLP", "LSTM", "MLR-RF", "AutoEncoder", "ConvLSTM", "RCAN"]

content = """
conf_test_available_data.yaml
This is a first lightweight test using ONLY the data currently available on the HPC (LMDZ predictors, LAI, Albedo, Evaporation target).
The 'soil moisture', 'tsol', and 'soiltile' predictors are INTENTIONALLY EXCLUDED for now because their NetCDF files have not been transferred/extracted yet.
The goal of this test is to quickly validate the pipeline (imports, data loading, preprocessing, output writing) without crashing on missing data.
It is extremely small (short period, few epochs/trees) and should be run on CPU first before launching any heavy experiments.
"""

for m in models:
    path = f"configs/{m}/doc.txt"
    with open(path, "a") as f:
        f.write("\n" + content)
