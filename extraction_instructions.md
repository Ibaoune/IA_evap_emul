# Data Extraction Instructions

## Issue: Missing Extraction Tools
I have checked the current HPC environment for the availability of extraction tools (`7za`, `7z`, `unrar`, `unar`, `bsdtar`). Unfortunately, none of these tools are installed in the `PATH`, nor are they available through the SLURM `module` system (I checked `module avail 7zip`, `p7zip`, and `unrar`).

Therefore, the `.rar` data archives cannot be extracted programmatically on this specific node. 

## Prepared Extraction Folders
To prepare the environment, I have already created a clean target directory structure for the extraction at:
`/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_extracted/`

Inside it, the following subfolders are ready:
- `soil_moisture/`
- `LMDZOR_250/`
- `LMDZOR_35/`
- `tsol_1979_2014/`
- `soiltile/`

## Required Manual Commands
You must log into an HPC node or a local machine that has `7z`, `7za`, or `unrar` installed, and run the following exact commands to extract the staging `.rar` files into the newly created folders.

If you have `unrar` available, use these commands:
```bash
unrar x "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/soil moisture.rar" "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_extracted/soil_moisture/"

unrar x "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/LMDZOR_250.rar" "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_extracted/LMDZOR_250/"

unrar x "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/LMDZOR_35.rar" "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_extracted/LMDZOR_35/"

unrar x "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/tsol_1979_2014.rar" "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_extracted/tsol_1979_2014/"

unrar x "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/soiltile.rar" "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_extracted/soiltile/"
```

If you only have `7za` or `7z`, replace `unrar x` with `7za x` and use the `-o` flag for the output directory like this:
```bash
7za x "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/soil moisture.rar" -o"/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_extracted/soil_moisture"
# ... repeat for others
```

## Next Steps
Once the extraction is successfully finished:
1. Review the generated `data_inventory.md` file in the project repository to see the expected variables.
2. Run `python scripts/check_data_paths.py --config configs/train/common.yaml` to rigorously validate that the NetCDF paths and dimensions are fully intact.
3. If `check_data_paths.py` reports success, you are cleared to queue the training jobs using SLURM.
