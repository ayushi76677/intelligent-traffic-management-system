"""
Phase 6 Real-Time Performance Benchmarking Script
=================================================

Measures empirical latencies, throughput, and memory consumption across
all stages of the real-time hybrid traffic intelligence pipeline:
1. Feature extraction latency (FeatureAdapter)
2. Sequence buffering latency (SequenceBuffer)
3. Preprocessing / Scaling latency (StandardScaler)
4. Model forward pass latency across batch sizes (1, 4, 8, 16, 32)
5. Multi-task output decoding latency
6. Hierarchical traffic aggregation latency (TrafficAggregator)
7. Total end-to-end per-frame latency across real video tracking data
8. Approximate FPS and memory footprint
"""

import os
import sys
import time
import json
import tracemalloc
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.hybrid_traffic.feature_adapter import LiveFeatureAdapter
from ml.hybrid_traffic.sequence_buffer import TrackSequenceBuffer
from ml.hybrid_traffic.realtime_inference import RealTimeTrafficPredictor
from ml.hybrid_traffic.traffic_aggregator import TrafficIntelligenceAggregator
from traffic_intelligence import TrafficIntelligenceEngine


def benchmark_pipeline(num_warmup: int = 20, num_benchmark_runs: int = 100):
    print("=" * 70)
    print("PHASE 6 REAL-TIME HYBRID PIPELINE BENCHMARK")
    print("=" * 70)

    tracemalloc.start()
    t_start_mem, _ = tracemalloc.get_traced_memory()

    # Instantiate modules
    t0 = time.perf_counter()
    adapter = LiveFeatureAdapter()
    buffer = TrackSequenceBuffer(sequence_length=20, feature_adapter=adapter)
    predictor = RealTimeTrafficPredictor(device="cpu")
    aggregator = TrafficIntelligenceAggregator()
    engine = TrafficIntelligenceEngine()
    init_time_s = time.perf_counter() - t0
    print(f"[*] Pipeline initialized in {init_time_s:.2f} seconds.")

    # 1. Feature Adapter Latency
    sample_obs = {"x1": 150.0, "y1": 250.0, "x2": 220.0, "y2": 370.0, "frame": 30, "time": 1.0}
    prev_obs = {"x1": 148.0, "y1": 248.0, "x2": 218.0, "y2": 368.0, "frame": 29, "time": 0.967}
    
    # Warmup
    for _ in range(num_warmup):
        adapter.compute_observation_features(sample_obs, prev_obs)

    adapter_times = []
    for _ in range(num_benchmark_runs):
        t0 = time.perf_counter()
        adapter.compute_observation_features(sample_obs, prev_obs)
        adapter_times.append((time.perf_counter() - t0) * 1000.0)

    avg_adapter_ms = float(np.mean(adapter_times))
    std_adapter_ms = float(np.std(adapter_times))
    print(f"[1] Feature Adapter Latency: {avg_adapter_ms:.4f} ms ± {std_adapter_ms:.4f} ms per observation")

    # 2. Sequence Buffer Update Latency
    test_buf = TrackSequenceBuffer(sequence_length=20, feature_adapter=adapter)
    buf_times = []
    for f in range(num_benchmark_runs):
        obs = {"track_id": 99, "x1": 100 + f, "y1": 200, "x2": 160 + f, "y2": 280, "frame": f, "time": f * 0.033}
        t0 = time.perf_counter()
        test_buf.update(obs)
        buf_times.append((time.perf_counter() - t0) * 1000.0)

    avg_buf_ms = float(np.mean(buf_times))
    std_buf_ms = float(np.std(buf_times))
    print(f"[2] Sequence Buffer Update Latency: {avg_buf_ms:.4f} ms ± {std_buf_ms:.4f} ms per observation")

    # 3. Preprocessing (StandardScaler) Latency across Batch Sizes
    scaling_benchmarks = {}
    for batch_size in [1, 4, 8, 16, 32]:
        dummy_seq = np.random.randn(batch_size, 20, 20).astype(np.float32)
        # Warmup
        for _ in range(num_warmup):
            predictor.preprocess_sequence(dummy_seq)
        
        times = []
        for _ in range(num_benchmark_runs):
            t0 = time.perf_counter()
            predictor.preprocess_sequence(dummy_seq)
            times.append((time.perf_counter() - t0) * 1000.0)
        
        scaling_benchmarks[batch_size] = {
            "mean_ms": round(float(np.mean(times)), 3),
            "std_ms": round(float(np.std(times)), 3)
        }
        print(f"[3] Scaler Preprocessing (Batch={batch_size:<2}): {np.mean(times):.3f} ms")

    # 4. Model Forward Pass Latency across Batch Sizes
    model_benchmarks = {}
    for batch_size in [1, 4, 8, 16, 32]:
        dummy_seq = np.random.randn(batch_size, 20, 20).astype(np.float32)
        tensor_x = predictor.preprocess_sequence(dummy_seq)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                predictor.model(tensor_x)
        
        times = []
        with torch.no_grad():
            for _ in range(num_benchmark_runs):
                t0 = time.perf_counter()
                predictor.model(tensor_x)
                times.append((time.perf_counter() - t0) * 1000.0)
        
        model_benchmarks[batch_size] = {
            "mean_ms": round(float(np.mean(times)), 3),
            "std_ms": round(float(np.std(times)), 3),
            "per_item_ms": round(float(np.mean(times)) / batch_size, 3)
        }
        print(f"[4] Model Forward Pass (Batch={batch_size:<2}): {np.mean(times):.3f} ms total ({np.mean(times)/batch_size:.3f} ms/item)")

    # 5. Output Decoding Latency
    sample_tensor = predictor.preprocess_sequence(np.random.randn(1, 20, 20).astype(np.float32))
    with torch.no_grad():
        raw_out = predictor.model(sample_tensor)
    
    dec_times = []
    for _ in range(num_benchmark_runs):
        t0 = time.perf_counter()
        predictor._decode_predictions(raw_out, idx=0)
        dec_times.append((time.perf_counter() - t0) * 1000.0)
    
    avg_dec_ms = float(np.mean(dec_times))
    print(f"[5] Multi-Head Output Decoding Latency: {avg_dec_ms:.4f} ms per vehicle")

    # 6. Hierarchical Aggregation Latency
    sample_dets = engine.frames_data.get(50, [])
    sample_preds = predictor.predict_frame_tracks(50, sample_dets)
    agg_times = []
    for _ in range(num_benchmark_runs):
        t0 = time.perf_counter()
        aggregator.aggregate(50, 50 / 30.0, sample_dets, sample_preds, None)
        agg_times.append((time.perf_counter() - t0) * 1000.0)
    
    avg_agg_ms = float(np.mean(agg_times))
    print(f"[6] Hierarchical Aggregation Latency: {avg_agg_ms:.4f} ms per frame ({len(sample_dets)} vehicles)")

    # 7. End-to-End Per-Frame Real Video Processing (Frames 0 to 100)
    live_predictor = RealTimeTrafficPredictor(device="cpu")
    e2e_frame_times = []
    vehicle_counts = []

    for f in range(100):
        dets = engine.frames_data.get(f, [])
        vehicle_counts.append(len(dets))
        t0 = time.perf_counter()
        f_preds = live_predictor.predict_frame_tracks(f, dets)
        intel = aggregator.aggregate(f, f / 30.0, dets, f_preds, None)
        e2e_frame_times.append((time.perf_counter() - t0) * 1000.0)

    avg_e2e_frame_ms = float(np.mean(e2e_frame_times))
    p95_e2e_frame_ms = float(np.percentile(e2e_frame_times, 95))
    min_e2e_frame_ms = float(np.min(e2e_frame_times))
    max_e2e_frame_ms = float(np.max(e2e_frame_times))
    avg_vehicles_per_frame = float(np.mean(vehicle_counts))
    approx_fps = 1000.0 / avg_e2e_frame_ms if avg_e2e_frame_ms > 0 else 0.0

    print(f"[7] End-to-End Per-Frame Latency: {avg_e2e_frame_ms:.2f} ms (p95: {p95_e2e_frame_ms:.2f} ms, min: {min_e2e_frame_ms:.2f} ms, max: {max_e2e_frame_ms:.2f} ms)")
    print(f"    Average Vehicles/Frame: {avg_vehicles_per_frame:.1f}")
    print(f"    Approximate Processing Throughput: {approx_fps:.1f} FPS")

    # Memory Tracking
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    current_mb = current_mem / (1024 * 1024)
    peak_mb = peak_mem / (1024 * 1024)
    print(f"[8] Memory Usage: Current Traced: {current_mb:.2f} MB, Peak Traced: {peak_mb:.2f} MB")

    results = {
        "device": str(predictor.device),
        "total_parameters": 226937,
        "feature_adapter_latency_ms": round(avg_adapter_ms, 4),
        "sequence_buffer_latency_ms": round(avg_buf_ms, 4),
        "preprocessing_benchmarks": scaling_benchmarks,
        "model_forward_benchmarks": model_benchmarks,
        "output_decoding_latency_ms": round(avg_dec_ms, 4),
        "aggregation_latency_ms": round(avg_agg_ms, 4),
        "end_to_end_per_frame_mean_ms": round(avg_e2e_frame_ms, 2),
        "end_to_end_per_frame_p95_ms": round(p95_e2e_frame_ms, 2),
        "end_to_end_per_frame_min_ms": round(min_e2e_frame_ms, 2),
        "end_to_end_per_frame_max_ms": round(max_e2e_frame_ms, 2),
        "approximate_fps": round(approx_fps, 1),
        "average_vehicles_per_frame": round(avg_vehicles_per_frame, 1),
        "memory_peak_traced_mb": round(peak_mb, 2),
    }

    # Save to JSON
    out_json = os.path.join(BASE_DIR, "reports", "phase6_performance_metrics.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved metrics to {out_json}")

    return results


if __name__ == "__main__":
    benchmark_pipeline()
