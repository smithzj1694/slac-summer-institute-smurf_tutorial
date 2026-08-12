import numpy as np
import pandas as pd
import os

def load_iq_csv(filename, time_col="time (ms)", iq_col="IQ"):
    """
    Load a fake SQUAT I/Q timestream from a CSV saved as:
        df = pd.DataFrame({"time (ms)": t, "IQ": iq})
        df.to_csv(filename, index=False)

    The IQ column is stored by pandas as complex-valued strings like
    "(1.23-0.45j)", so we parse them back to a complex numpy array.

    Returns
    -------
    times_ms : np.ndarray (float)
        Time array in milliseconds.
    iq : np.ndarray (complex)
        Complex I/Q timestream.
    """
    df = pd.read_csv(filename)
    times_ms = df[time_col].to_numpy(dtype=float)

    raw = df[iq_col].to_numpy()
    if np.iscomplexobj(raw):
        iq = raw.astype(complex)
    else:
        # Strip parentheses/whitespace, then parse each entry to complex.
        def _parse(s):
            return complex(str(s).strip().lstrip("(").rstrip(")").replace(" ", ""))
        iq = np.array([_parse(s) for s in raw], dtype=complex)

    return times_ms, iq

def read_file(filepath, filename):
    readin = np.load(os.path.join(filepath, filename), allow_pickle=True)
    return readin['data'].item()

def unwrap_phases(data, force_line_delay_val=None, verbose=True):
    ## Set up output containers, unwrap phases (i.e. remove 2pi jumps)
    corrected_phases = np.zeros(len(data["phases"]-1))
    unwrapped = np.unwrap(data["phases"])
    ## Give user the option to manually set a line delay
    ## If no value is supplied, calculate the line delay from the data
    if force_line_delay_val is None:
        line_delay = np.mean(unwrapped[1:]-unwrapped[:-1])/(data["freqs"][1:]-data["freqs"][:-1])
        line_delay = np.mean(line_delay)
        if verbose: print("Calculated line delay:", line_delay)
    else:
         if verbose: print("Manually set line delay:", force_line_delay_val)
         line_delay = force_line_delay_val
    for n, phase in enumerate(unwrapped):
            corrected_phases[n] = phase - (data["freqs"][n] - data["freqs"][0])*line_delay
    return corrected_phases, line_delay