
import numpy as np
from app.helpers.polars_hyper_plugin import *
import polars as pl
from typing import KeysView

def _e_default(x):
    return x() if callable(x) else x

def ensure_list(value, *, default=list, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, list): return value
    if isinstance(value, tuple): return list(value)
    if isinstance(value, KeysView): return list(value)
    return [value]

def ensure_bool(value, *, default=False, deep=True, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, bool): return value
    if deep and isinstance(value, (tuple, list)):
        return type(value)([ensure_bool(x, default=default, deep=True, allow_none=allow_none) for x in value])
    try: return bool(value)
    except Exception: return _e_default(default)

def ensure_int(value, *, default=0, deep=True, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, int): return value
    if deep and isinstance(value, (tuple, list)):
        return type(value)([ensure_int(x, default=default, deep=True, allow_none=allow_none) for x in value])
    try: return int(value)
    except Exception: return _e_default(default)

def ensure_float(value, *, default=0, deep=True, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, float): return value
    if deep and isinstance(value, (tuple, list)):
        return type(value)([ensure_float(x, default=default, deep=True, allow_none=allow_none) for x in value])
    try: return float(value)
    except Exception: return _e_default(default)

def ensure_numeric(value, *, default=0, deep=True, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, bool): return int(value)
    if isinstance(value, (int, float)): return value
    if deep and isinstance(value, (tuple, list)):
        return type(value)([ensure_numeric(x, default=default, deep=True, allow_none=allow_none) for x in value])
    try: return float(value)
    except Exception: return _e_default(default)

def ensure_str(value, *, default=str, deep=True, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, str): return value
    if deep and isinstance(value, (tuple, list)):
        return type(value)([ensure_str(x, default=default, deep=True, allow_none=allow_none) for x in value])
    try: return str(value)
    except Exception: return _e_default(default)

def ensure_upper(value, *, default=str, deep=True):
    return ensure_str(value, default=default, deep=deep, allow_none=False).upper()

def ensure_lower(value, *, default=str, deep=True):
    return ensure_str(value, default=default, deep=deep, allow_none=False).lower()

def ensure_camel(value, *, default=str, deep=True):
    return ensure_str(value, default=default, deep=deep, allow_none=False).upper()

def ensure_tuple(value, *, default=tuple, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, tuple): return value
    if isinstance(value, list): return tuple(value)
    return (value,)

def ensure_set(value, *, default=set, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, set): return value
    if isinstance(value, (list, tuple)): return set(value)
    return {value}

def ensure_dict(value, *, default=dict, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, dict): return value
    try: return dict(value)
    except Exception: return _e_default(default)

def ensure_bytes(value, *, default=bytes, allow_none=False):
    if value is None: return _e_default(default) if not allow_none else None
    if isinstance(value, bytes): return value
    try: return ensure_str(value, default=default).encode('utf-8')
    except Exception: return _e_default(default)

def ensure_uint32_numpy(arr: np.ndarray):
    if arr.size == 0: return arr.astype(np.uint32, copy=False)
    return arr if arr.dtype == np.uint32 else arr.astype(np.uint32, copy=False)

def ensure_lazy(df):
    if df is None: return None
    if isinstance(df, pl.LazyFrame): return df
    if isinstance(df, pl.DataFrame): return df.lazy()
    if isinstance(df, (dict, list, pl.Series)):
        try: return pl.LazyFrame(df)
        except Exception: return None
    return None


