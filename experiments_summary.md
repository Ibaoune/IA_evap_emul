# Experiments Summary

| Model | Config file | Data used | Missing optional data | Job script | Status | Log path | Result path | Comment |
|---|---|---|---|---|---|---|---|---|
| XGB | `conf_test_available_data.yaml` | LMDZ, LAI, Albedo, Evap Target | `soil_moisture`, `tsol`, `soiltile` | `job_cpu.sh` | Success | `logs/evap_cpu_*.out` | `results/xgb/conf_test_available_data/` | First validation test |
| MLP | `conf_test_available_data.yaml` | LMDZ, LAI, Albedo, Evap Target | `soil_moisture`, `tsol`, `soiltile` | `job_cpu.sh` | Success | - | `results/mlp/conf_test_available_data/` | |
| LSTM | `conf_test_available_data.yaml` | LMDZ, LAI, Albedo, Evap Target | `soil_moisture`, `tsol`, `soiltile` | `job_cpu.sh` | Success | - | `results/lstm/conf_test_available_data/` | |
| MLR-RF | `conf_test_available_data.yaml` | LMDZ, LAI, Albedo, Evap Target | `soil_moisture`, `tsol`, `soiltile` | `job_cpu.sh` | Success | - | `results/mlr_rf/conf_test_available_data/` | |
| AutoEncoder | `conf_test_available_data.yaml` | LMDZ, LAI, Albedo, Evap Target | `soil_moisture`, `tsol`, `soiltile` | `job_cpu.sh` | Success | - | `results/autoencoder/conf_test_available_data/` | |
| ConvLSTM | `conf_test_available_data.yaml` | LMDZ, LAI, Albedo, Evap Target | `soil_moisture`, `tsol`, `soiltile` | `job_cpu.sh` | Success | - | `results/convlstm/conf_test_available_data/` | |
| RCAN | `conf_test_available_data.yaml` | LMDZ, LAI, Albedo, Evap Target | `soil_moisture`, `tsol`, `soiltile` | `job_cpu.sh` | Success | - | `results/rcan/conf_test_available_data/` | |
