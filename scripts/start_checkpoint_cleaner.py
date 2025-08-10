import argparse
import os
import re
import shutil
import time


def clean_up(checkpoint_dir: str, divisor: int = 10000) -> None:
    """Remove excess checkpoints, keeping those whose step count is divisible by divisor.
    Always keeps the latest valid and latest invalid checkpoints.

    Args:
        checkpoint_dir: Directory that contains sub-directories named
            ``step_<step_number>``.
        divisor: Keep checkpoints whose step count is divisible by this number.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(checkpoint_dir):
        print(f"[{timestamp}] Checkpoint directory does not exist: {checkpoint_dir}")
        return

    # Pattern to match checkpoint directory names like ``step_00001000``
    pattern = re.compile(r"step_(\d{8})")

    # Gather valid and invalid checkpoint directories separately
    valid_checkpoints = []  # (step, name)
    invalid_checkpoints = []  # (step, name)

    for name in os.listdir(checkpoint_dir):
        match = pattern.fullmatch(name)
        if not match:
            continue

        path = os.path.join(checkpoint_dir, name)
        step = int(match.group(1))

        is_valid = (
            os.path.isdir(path)
            and os.path.exists(os.path.join(path, ".metadata"))
            and os.path.exists(os.path.join(path, "shared.pth"))
        )

        if is_valid:
            valid_checkpoints.append((step, name))
        else:
            invalid_checkpoints.append((step, name))

    # Sort by step number (ascending)
    valid_checkpoints.sort()
    invalid_checkpoints.sort()

    # Determine which checkpoints to keep
    keep_names = set()

    # Keep checkpoints divisible by divisor
    for step, name in valid_checkpoints:
        if step % divisor == 0:
            keep_names.add(name)

    # Always keep the latest valid checkpoint
    if valid_checkpoints:
        keep_names.add(valid_checkpoints[-1][1])

    # Always keep the latest invalid checkpoint
    if invalid_checkpoints:
        keep_names.add(invalid_checkpoints[-1][1])

    all_checkpoints = valid_checkpoints + invalid_checkpoints
    total_valid = len(valid_checkpoints)
    total_invalid = len(invalid_checkpoints)

    print(f"[{timestamp}] Found {total_valid} valid and {total_invalid} invalid checkpoints")
    print(f"[{timestamp}] Keeping {len(keep_names)} checkpoints (divisible by {divisor}, latest valid, latest invalid)")

    # Delete the rest
    for _step, name in all_checkpoints:
        if name not in keep_names:
            path = os.path.join(checkpoint_dir, name)
            print(f"[{timestamp}] Deleting: {path}")
            shutil.rmtree(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Periodically clean up checkpoints directory.")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint sub-directories"
    )
    parser.add_argument("--interval", type=int, default=5, help="Interval in minutes between cleanup runs (default: 5)")
    parser.add_argument(
        "--divisor",
        type=int,
        default=10000,
        help="Keep checkpoints whose step count is divisible by this number (default: 10000)",
    )
    args = parser.parse_args()

    # Run forever until interrupted
    try:
        while True:
            clean_up(args.checkpoint_dir, args.divisor)
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt. Exiting.")
