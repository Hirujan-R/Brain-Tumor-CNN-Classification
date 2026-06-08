from pathlib import Path
import h5py


def print_group(name, obj):
    print(f"{name}: {type(obj)}")

CVIND_PATH = Path("data/splits/cvind.mat")

with h5py.File(CVIND_PATH, "r") as f:
    print("\n=== TOP LEVEL KEYS ===")
    print(list(f.keys()))

    print("\n=== FULL STRUCTURE ===")
    f.visititems(print_group)

    print("\n=== DATA PREVIEW ===")

    for key in f.keys():
        obj = f[key]

        if isinstance(obj, h5py.Dataset):
            print(f"\nKEY: {key}")
            print(f"Shape: {obj.shape}")
            print(f"Dtype: {obj.dtype}")

            try:
                data = obj[:]
                print(data[:20])
            except Exception as e:
                print(e)