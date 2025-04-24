#!/usr/bin/env python3
import argparse
import sys

try:
    from safetensors import safe_open
except ImportError:
    print("Missing dependency: install with `pip install safetensors`", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="List all layer names in a .safetensors file."
    )
    parser.add_argument(
        "safetensor_file",
        help="Path to the .safetensors file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print count of layers before listing"
    )
    args = parser.parse_args()

    try:
        with safe_open(args.safetensor_file, framework="pt") as f:
            keys = list(f.keys())
    except Exception as e:
        print(f"Error reading '{args.safetensor_file}': {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(keys)} layers in '{args.safetensor_file}':")

    for name in keys:
        print(name)

if __name__ == "__main__":
    main()

