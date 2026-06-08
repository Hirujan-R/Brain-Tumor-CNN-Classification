# from pathlib import Path

# import h5py
# import numpy as np
# import pandas as pd


# INDEX_PATH = Path("data/ingested/index.csv")
# CVIND_PATH = Path("data/splits/cvind.mat")


# def load_folds():

#     with h5py.File(CVIND_PATH, "r") as f:

#         folds = np.array(
#             f["cvind"]
#         )

#     folds = folds.squeeze()
#     folds = folds.astype(int)

#     return folds


# def main():

#     df = pd.read_csv(INDEX_PATH)

#     folds = load_folds()

#     df["fold"] = folds

#     leaking_patients = []

#     for pid, group in df.groupby("patient_id"):

#         unique_folds = group["fold"].unique()

#         if len(unique_folds) > 1:

#             leaking_patients.append(
#                 {
#                     "pid": pid,
#                     "folds": unique_folds.tolist(),
#                     "samples": len(group),
#                 }
#             )

#     print(
#         f"Patients: {df['patient_id'].nunique()}"
#     )

#     print(
#         f"Leakage cases: {len(leaking_patients)}"
#     )

#     if leaking_patients:

#         print(
#             "\nFirst 10 leakage examples:"
#         )

#         for item in leaking_patients[:10]:

#             print(item)

#     else:

#         print(
#             "\nNo patient leakage detected."
#         )

# import pandas as pd

# df = pd.read_csv("data/ingested/index.csv")

# print(df.columns.tolist())

# for col in df.columns:
#     print(col, df[col].nunique())

# if __name__ == "__main__":
#     main()

import h5py
from pathlib import Path

sample = next(
    Path("data/interim/extracted_mat").rglob("*.mat")
)

print(sample)

with h5py.File(sample, "r") as f:

    print(list(f.keys()))

    cjdata = f["cjdata"]

    print(list(cjdata.keys()))

    pid_obj = cjdata["PID"]

    print("PID shape:", pid_obj.shape)
    print("PID dtype:", pid_obj.dtype)

    print("PID raw:", pid_obj[()])