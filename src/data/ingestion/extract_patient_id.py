import numpy as np
import h5py

def decode_pid(pid_array):
    pid_array = np.array(pid_array).squeeze()
    return "".join([chr(int(x)) for x in pid_array]).strip()

def extract_patient_id(filepath: str) -> str:
    """
    Extracts and decodes patient ID from MATLAB .mat file
    """

    with h5py.File(filepath, "r") as f:
        cjdata = f["cjdata"]

        pid_raw = cjdata["PID"][()]

        patient_id = decode_pid(pid_raw)

        return patient_id