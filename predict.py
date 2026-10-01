"""
Command-Line Interface for Traffic Trajectory Model Inference
=============================================================

Usage examples:
  python predict.py --sample-idx 0
  python predict.py --num-samples 5
  python predict.py --input-file data/hybrid_traffic/sequences/test_sequences.npz --sample-idx 42
  python predict.py --input-file data/hybrid_traffic/sequences/test_sequences.npz --num-samples 10 --output predictions/sample_preds.json
"""

import os
import sys
import argparse
import json
import numpy as np

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from inference import TrafficInferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using the trained TCN-Transformer Gated Hybrid model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="checkpoints/feature_scaler.joblib",
        help="Path to fitted feature scaler (.joblib)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/hybrid_traffic/sequences/test_sequences.npz",
        help="Path to NPZ file containing input sequence array 'X'"
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Specific sequence index to predict"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of sequences to predict from input file (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device: cpu or cuda"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("TRAFFIC INTELLIGENCE CLI PREDICTION ENGINE")
    print("=" * 70)
    print(f"Model Checkpoint: {args.checkpoint}")
    print(f"Feature Scaler:   {args.scaler}")
    print(f"Input Sequences:  {args.input_file}")

    # Load sequences
    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input sequence file not found: {args.input_file}")
        sys.exit(1)

    data = np.load(args.input_file, mmap_mode="r")
    X = data["X"]
    track_ids = data["track_ids"] if "track_ids" in data else None

    # Slice sequences
    if args.sample_idx is not None:
        if args.sample_idx < 0 or args.sample_idx >= len(X):
            print(f"[ERROR] Sample index {args.sample_idx} out of range [0, {len(X)-1}]")
            sys.exit(1)
        seqs = X[args.sample_idx : args.sample_idx + 1]
        indices = [args.sample_idx]
    else:
        num = min(args.num_samples, len(X))
        seqs = X[:num]
        indices = list(range(num))

    # Initialize Pipeline
    pipeline = TrafficInferencePipeline(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        device=args.device
    )

    print(f"\nLoaded model from Epoch {pipeline.epoch} (Val Loss: {pipeline.val_loss:.4f})")
    print(f"Running inference on {len(seqs)} sequence(s)...\n")

    predictions = pipeline.predict(seqs)

    # Print results
    for idx, (seq_idx, pred) in enumerate(zip(indices, predictions)):
        tid = track_ids[seq_idx] if track_ids is not None else "N/A"
        print("-" * 70)
        print(f"Sample #{idx+1} [Sequence Index: {seq_idx} | Track ID: {tid}]")
        print("-" * 70)
        print(f"  * Congestion Level:     {pred['congestion_level']['label']:<15} (Conf: {pred['congestion_level']['confidence']*100:.1f}%)")
        print(f"  * Congestion Score:     {pred['congestion_score']:<15.2f} / 100")
        print(f"  * Risk Level:           {pred['risk_level']['label']:<15} (Conf: {pred['risk_level']['confidence']*100:.1f}%)")
        print(f"  * Approaching Status:   {pred['is_approaching']['label']:<15} (Threat: {pred['approach_threat_score']:.2f}/100)")
        print(f"  * Motion State:         {pred['motion_state']['label']:<15} (Conf: {pred['motion_state']['confidence']*100:.1f}%)")
        print(f"  * Maneuver Type:        {pred['maneuver_type']['label']:<15} (Conf: {pred['maneuver_type']['confidence']*100:.1f}%)")
        print(f"  * Infraction Status:    {pred['has_infraction']['label']:<15} (Type: {pred['infraction_type']['label']})")
        if "predicted_next_displacement" in pred:
            disp = pred["predicted_next_displacement"]
            print(f"  * Next Displacement:    dx={disp['dx']:+.2f} px, dy={disp['dy']:+.2f} px")

    # Save output if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"\n[OUTPUT] Predictions saved to: {args.output}")

    print("\n" + "=" * 70)
    print("Inference completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
