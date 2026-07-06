import os
import yaml
import glob
try:
    import xarray as xr
except ImportError:
    xr = None
    print("WARNING: xarray is not installed. NetCDF inspection will be skipped.")

def load_config(path="configs/common.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def check_paths():
    print("=" * 40)
    print("Data Paths Validation Script")
    print("=" * 40)
    
    cfg = load_config()
    paths = cfg.get("paths", {})
    
    missing_dirs = []
    
    for key, path in paths.items():
        if "dir" in key: continue # skip save dirs
        
        print(f"\nChecking [{key}]: {path}")
        if os.path.isdir(path):
            print("  [OK] Directory exists.")
            nc_files = glob.glob(os.path.join(path, "*.nc"))
            if not nc_files:
                print("  [WARNING] Directory is empty or contains no .nc files! (Pending extraction?)")
            else:
                print(f"  Found {len(nc_files)} NetCDF files.")
                # Try opening the first one
                if xr is not None:
                    test_file = nc_files[0]
                    try:
                        ds = xr.open_dataset(test_file)
                        print(f"  [INSPECT] Successfully opened {os.path.basename(test_file)}")
                        print(f"            Dimensions: {list(ds.dims.keys())}")
                        print(f"            Variables: {list(ds.data_vars.keys())}")
                    except Exception as e:
                        print(f"  [ERROR] Failed to open {os.path.basename(test_file)}: {e}")
        else:
            print("  [FAIL] Path does not exist or is not a directory.")
            missing_dirs.append(key)
            
    if missing_dirs:
        print("\n" + "=" * 40)
        print("VALIDATION FAILED!")
        print("The following paths from common.yaml are missing:")
        for k in missing_dirs:
            print(f" - {k}")
        print("\nPlease extract the `.rar` data using instructions in `extraction_instructions.md`.")
    else:
        print("\n" + "=" * 40)
        print("VALIDATION SUCCESSFUL!")
        print("All data directories exist. You may proceed with SLURM jobs.")

if __name__ == "__main__":
    check_paths()
