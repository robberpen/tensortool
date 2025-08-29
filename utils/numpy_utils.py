#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pickle
import sys
from typing import Tuple

# ----------------------
# Helpers
# ----------------------
def parse_shape(s: str) -> Tuple[int, ...]:
    # Parse "H,W,C" or "C,H,W" etc. into a tuple of ints
    try:
        shp = tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
        if len(shp) == 0:
            raise ValueError
        return shp
    except Exception:
        raise argparse.ArgumentTypeError("Invalid --shape format. Example: --shape 640,640,3")

def dtype_from_str(name: str, little: bool) -> np.dtype:
    # Map string to numpy dtype with endian control for >8-bit types
    mapping = {
        "int8": np.int8, "uint8": np.uint8,
        "int16": np.int16, "uint16": np.uint16,
        "int32": np.int32, "uint32": np.uint32,
        "float32": np.float32, "float64": np.float64,
    }
    if name not in mapping:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {name}")
    dt = np.dtype(mapping[name])
    # Only apply byteorder for items with size > 1 byte
    if dt.itemsize > 1:
        dt = dt.newbyteorder("<" if little else ">")
    return dt

def validate_layout(arr: np.ndarray, layout: str):
    # Enforce C=3 for HWC/CHW transforms as requested
    if arr.ndim != 3:
        return
    if layout == "CHW" and arr.shape[2] != 3:
        raise ValueError("Expect HWC with C=3 before transposing to CHW.")
    if layout == "HWC" and arr.shape[0] != 3:
        raise ValueError("Expect CHW with C=3 before transposing to HWC.")

def transpose_layout(arr: np.ndarray, layout: str) -> np.ndarray:
    # HWC <-> CHW transforms
    if arr.ndim != 3:
        return arr
    if layout == "CHW" and arr.shape[2] == 3:
        return np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    if layout == "HWC" and arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    return arr

# ----------------------
# CLI
# ----------------------
EPILOG = r"""
Usage examples:

# 1) Load .npy, transpose to CHW (C=3), convert to int8 (typical DSP int8), save as pickle
numpy_utils.py --load ./face-640x640.npy --as CHW --type int8 --save ./face-640x640.pkl

# 2) Load .npy, transpose to CHW, convert to int32, save as RAW (little-endian)
numpy_utils.py --load ./face-640x640.npy --as CHW --type int32 --save ./face-640x640.raw --raw --little

# 3) Load RAW back (little-endian), shape=CHW(3,640,640), view first 16 elements, save as .npy(pickle)
numpy_utils.py --load_raw ./face-640x640.raw --shape 3,640,640 --raw_dtype int8 --little --show 16 --save ./restored.pkl

# 4) Convert RAW HWC(640,640,3) -> CHW then int8 -> RAW big-endian
numpy_utils.py --load_raw ./img.hwcrgb --shape 640,640,3 --raw_dtype uint8 --as CHW --type int8 --save ./img.chw.i8.be --raw
"""

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Numpy utilities: load/transform/save .npy or RAW binary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG
    )

    # Input (mutually exclusive: .npy vs raw)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--load", type=str,
                     help="Path to input .npy file (pickle saved by numpy/save)")
    src.add_argument("--load_raw", type=str,
                     help="Path to input RAW binary file")

    parser.add_argument("--shape", type=parse_shape,
                        help="Required when --load_raw is used. Example: --shape 640,640,3")

    parser.add_argument("--raw_dtype",
                        choices=["int8","uint8","int16","uint16","int32","uint32","float32","float64"],
                        help="RAW element dtype for --load_raw (default int8 if omitted)")

    parser.add_argument("--show", type=int, default=0,
                        help="Show first <num> elements after all transforms")

    # Output group
    out_group = parser.add_argument_group("Output options")
    out_group.add_argument("--save", type=str,
                           help="Output path. Default saves as pickle if provided")
    out_group.add_argument("--as", dest="layout", choices=["HWC", "CHW"], default="HWC",
                           help="Transpose layout between HWC <-> CHW (C=3). Default HWC")
    out_group.add_argument("--type",
                           choices=["int8","int32","uint8","float32"],
                           default="int8",
                           help="Convert dtype (default int8). Note: int8 typical for DSP.")
    out_group.add_argument("--raw", action="store_true",
                           help="Save as RAW binary instead of pickle")
    out_group.add_argument("--little", action="store_true",
                           help="When RAW is involved (load or save), use Little-Endian. Default Big-Endian")

    return parser

# ----------------------
# Main
# ----------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load
    if args.load_raw:
        if args.shape is None:
            parser.error("--shape is required when using --load_raw")
        # Infer dtype for reading RAW
        in_dtype_name = args.raw_dtype or "int8"
        dt = dtype_from_str(in_dtype_name, little=args.little)
        # Read flat, then reshape
        raw = np.fromfile(args.load_raw, dtype=dt)
        try:
            arr = raw.reshape(args.shape)
        except Exception as e:
            parser.error(f"Cannot reshape RAW to {args.shape}: {e}")
        print(f"Loaded RAW: path={args.load_raw}, shape={arr.shape}, dtype={arr.dtype}")
    else:
        arr = np.load(args.load, allow_pickle=True)
        print(f"Loaded NPY: path={args.load}, shape={arr.shape}, dtype={arr.dtype}")

    # Layout transpose (C must be 3 when applicable)
    try:
        validate_layout(arr, args.layout)
        arr = transpose_layout(arr, args.layout)
        if arr.ndim == 3:
            print(f"Layout -> {args.layout}: shape={arr.shape}")
    except Exception as e:
        parser.error(str(e))

    # Convert dtype (post-layout)
    target_dt = dtype_from_str(args.type, little=args.little)
    # Keep sign/scale awareness for int8 workflows on DSPs
    arr = arr.astype(target_dt, copy=False)
    print(f"Dtype -> {arr.dtype}")

    # Preview
    if args.show > 0:
        flat = arr.ravel()
        n = min(args.show, flat.size)
        print(f"First {n} elements: {flat[:n]}")

    # Save
    if args.save:
        if args.raw:
            # Ensure endian as requested
            if arr.dtype.itemsize > 1:
                arr = arr.astype(arr.dtype.newbyteorder("<" if args.little else ">"), copy=False)
            endian_tag = "LE" if args.little else "BE"
            print(f"Saving RAW ({endian_tag}) -> {args.save}")
            arr.tofile(args.save)
        else:
            print(f"Saving pickle (.npy-like via pickle) -> {args.save}")
            with open(args.save, "wb") as f:
                pickle.dump(arr, f)

if __name__ == "__main__":
    main()

