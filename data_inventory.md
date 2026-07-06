# Comprehensive Data Inventory

Based on an exhaustive script inspection of `/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data`, here is the detailed breakdown of what exists, what is usable, and what is genuinely missing.

### 1. Data Already Available and Usable

| Dataset / Variable | Existing? | File path | Variable name | Units | Lat/Lon extent | Resolution | Comment |
|---|---|---|---|---|---|---|---|
| Target Evaporation | Yes | `.../data/models/evap_staging/evap_corrigée_clean.nc` | `evap_corrigée` | N/A | Lon: -18.8 to 0.95, Lat: 20.0 to 39.9 | 60x68 | Target output directly available |
| LMDZ 2.5° Predictors | Yes | `.../data/models/lmdz/r250/` | `vitv`, `temp`, `geop`, `rhum`, `vitu` | m/s, K, m2/s2, -, m/s | Lon: -25 to 5, Lat: 16.5 to 43.1 | 13x22 | Readily available |
| LMDZ 3.5° Predictors | Yes | `.../data/models/lmdz/r35/` | `vitv`, `temp`, `precip`, `geop`, `rhum`, `vitu` | m/s, K, kg/(s*m2), etc. | Lon: -25.7 to 5.9, Lat: 16.2 to 43.9 | 95x92 | Readily available |
| LAI | Yes | `.../data/models/evap_staging/LAImean-hist1.nc` | `LAImean` | 1 | Lon: -25.7 to 5.9, Lat: 16.2 to 43.9 | 95x92 | Fully extracted |
| Albedo | Yes | `.../data/models/evap_staging/albedo_glob-hist1.nc` | `albedo_glob` | 1 | Lon: -25.7 to 5.9, Lat: 16.2 to 43.9 | 95x92 | Fully extracted |
| Near-surface temp (t2m) | Yes | `.../data/models/evap_staging/t2m-hist1.nc` | `t2m` | K | Lon: -25 to 5, Lat: 16.5 to 43.1 | 13x22 | Fully extracted |
| Wind & Pressure (surface) | Yes | `.../data/models/evap_staging/` | `u10m`, `v10m`, `pres`, `shumsurf` | m/s, Pa, kg/kg | Lon: -25 to 5, Lat: 16.5 to 43.1 | 13x22 | Available |
| Radiation (Net LW/SW) | Yes | `.../data/models/evap_staging/` | `LWnet`, `SWnet` | N/A | Lon: -25 to 5, Lat: 16.5 to 43.1 | 13x22 | Available |
| Morocco Shapefile | Yes | `.../data/shapefiles/Morocco_shpfile/DA_REGIONS_12R.shp` | N/A | N/A | N/A | N/A | Available |
| ERA5 Reanalysis | Yes | `.../data/reanalysis/era5/` & `.../data/data_to_remove_after_check/` | `t`, `u`, `v`, `q`, `z`, `tp` | K, m/s, kg/kg, m2/s2 | Lon: -20 to 0, Lat: 20 to 40 | 81x81 | Highly detailed pressure/surface vars available for 1979-2020 |

### 2. Data Genuinely Missing (Requires `.rar` Extraction or Re-upload)

| Dataset | File | Missing variables | Impact |
|---|---|---|---|
| Soil Moisture | `soil moisture.rar` | `mrso` | Critical for temporal dynamics in `ConvLSTM` / `LSTM` |
| Soil Temperature | `tsol_1979_2014.rar` | `tsol` | Needed for surface heat flux predictions |
| Land Surface Tiles | `soiltile.rar` | soil types / classes | Needed for specific static spatial models |

*(Note: The `LMDZOR_250.rar` and `LMDZOR_35.rar` mentioned previously do not need extraction because their contents were already found cleanly extracted under `data/models/lmdz/r250/` and `r35/`!)*

### Configuration Update Summary
I have updated `configs/common.yaml` to securely point to the found LMDZ, shapefiles, and target evaporation folders. It is now only the 3 missing datasets (soil moisture, tsol, soiltile) that are pointing to the `evap_extracted` target folders, pending your manual extraction.
