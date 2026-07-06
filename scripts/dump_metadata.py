import os
import glob
import xarray as xr

base_dir = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data"

nc_files = glob.glob(os.path.join(base_dir, "**/*.nc"), recursive=True)
nc_files += glob.glob(os.path.join(base_dir, "**/*.nc4"), recursive=True)

with open("scratch_inventory.txt", "w") as f:
    for file in nc_files:
        f.write(f"File: {file}\n")
        try:
            ds = xr.open_dataset(file)
            f.write(f"  Dimensions: {dict(ds.dims)}\n")
            f.write(f"  Coordinates: {list(ds.coords)}\n")
            f.write(f"  Variables: {list(ds.data_vars)}\n")
            for coord in ["lon", "longitude", "lat", "latitude", "time", "level", "plev"]:
                if coord in ds:
                    try:
                        f.write(f"    {coord}: min={ds[coord].values.min()} max={ds[coord].values.max()}\n")
                    except Exception as e:
                        f.write(f"    {coord}: Could not compute min/max ({e})\n")
            
            # Print units for variables
            for var in ds.data_vars:
                units = ds[var].attrs.get("units", "N/A")
                f.write(f"    Var [{var}] units: {units}\n")
                
            ds.close()
        except Exception as e:
            f.write(f"  Error opening: {e}\n")
        f.write("\n")
