#!/usr/bin/env python3
import argparse
import torch
from safetensors.torch import load_file, save_file

def parse_range(rng: str):
    start, end = rng.split("-", 1)
    return int(start), int(end)

def main():
    p = argparse.ArgumentParser(
        description="Zero out AdaLN style layers in a U-Net LoRA checkpoint"
    )
    p.add_argument("input", help="input .safetensors")
    p.add_argument("output", help="output .safetensors")
    p.add_argument(
        "--stream",
        choices=["double", "single", "dual"],
        default="dual",
        help="which stream_blocks to target",
    )
    p.add_argument(
        "--range",
        required=True,
        help="block index range to zero, e.g. 0-5",
    )
    # Zero out attention layers instead
    p.add_argument(
        "--attention",
        action="store_true",
        help="zero out attention layers instead of AdaLN_modulation.1 layers",
    )
    args = p.parse_args()

    start, end = parse_range(args.range)
    stream_key = f"{args.stream}_stream_blocks"

    # load everything into a dict of tensors
    tensors = load_file(args.input, device="cpu")
    
    count = total_count = 0
    skipped = []
    zeroed = []
    already_zeroed = []

    # zero out just the AdaLN_modulation.1 layers in the given block range
    for key, t in list(tensors.items()):
        total_count += 1
        parts = key.split(".")
        # looking for keys like:
        # diffusion_model.<stream>_stream_blocks.<i>.block.adaLN_modulation.1.<weight|bias|...>
        # Check layer data to see if it's already been zeroed out in a previous run
        if tensors[key].abs().sum() == 0:
            already_zeroed.append(key) 
                
        # Print the "parts" list for debugging
        print(f"Parts: {parts}")
        
        if (
            len(parts) > 5
            and (parts[1] == stream_key or args.stream == "dual")
            and parts[3] == "block"
            and ((args.attention and (parts[4] == "attn1" or parts[4] == "attn2")) or
                 (not args.attention and parts[4] == "adaLN_modulation"))
            and (args.attention or parts[5] == "1")
        ):
            try:
                blk = int(parts[2])
            except ValueError:
                continue
            if start <= blk <= end:
                # zero out the tensor if not already zeroed
                if tensors[key].abs().sum() != 0:
                    print(f"Zeroing out {key}")
                    zeroed.append(key)
                tensors[key] = torch.zeros_like(t)
                count += 1
        else:
            skipped.append(key)
            print(f"Skipping layer {key}")

# diffusion_model.double_stream_blocks.9.block.adaLN_modulation.1.lora_A.weight
# diffusion_model.single_stream_blocks.12.block.adaLN_modulation.1.lora_A.weight

    # write out
    save_file(tensors, args.output)
    print(f"Skipped these layers:")
    for key in skipped:
        print(f"  {key}")
    print(f"These layers are already zeroed:")
    for key in already_zeroed:
        print(f"  {key}")
    print(f"Zeroed out these layers:")
    for key in zeroed:
        print(f"  {key}")
    print(f"Saved with blocks {start}–{end} zeroed out in {args.output}")
    print(f"Zeroed out {count} of {total_count} tensors")


if __name__ == "__main__":
    main()
