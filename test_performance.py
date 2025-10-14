"""Performance test for feature extraction."""

import time
import numpy as np
from src.synthetic_data import SyntheticIncidentGenerator
from src.features import IncidentFeatureExtractor


def test_extraction_performance():
    """Test feature extraction performance."""
    print("Feature Extraction Performance Test")
    print("=" * 40)

    # Setup
    generator = SyntheticIncidentGenerator(seed=42)
    extractor = IncidentFeatureExtractor()

    # Generate test incidents
    incidents = generator.generate_training_dataset(n_samples=100)
    test_incident = incidents[0]

    # Warm up
    for _ in range(10):
        extractor.extract_model_features(test_incident)

    # Time model feature extraction
    print("\nTesting model feature extraction speed...")
    times = []

    for i in range(1000):
        start_time = time.perf_counter()
        features = extractor.extract_model_features(test_incident)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p95_time = np.percentile(times, 95)

    print(f"Average extraction time: {avg_time:.3f}ms")
    print(f"Minimum extraction time: {min_time:.3f}ms")
    print(f"Maximum extraction time: {max_time:.3f}ms")
    print(f"95th percentile time: {p95_time:.3f}ms")

    # Check if meets requirement
    if avg_time < 5.0:
        print(f"[PASS] Average time ({avg_time:.3f}ms) < 5ms target")
    else:
        print(f"[FAIL] Average time ({avg_time:.3f}ms) >= 5ms target")

    # Test batch extraction
    print(f"\nTesting batch extraction...")
    start_time = time.perf_counter()
    batch_features = extractor.extract_batch_features(incidents[:100])
    end_time = time.perf_counter()

    batch_time = (end_time - start_time) * 1000
    per_incident_time = batch_time / 100

    print(f"Batch extraction (100 incidents): {batch_time:.3f}ms")
    print(f"Per incident in batch: {per_incident_time:.3f}ms")

    # Test agent context extraction (can be slower)
    print(f"\nTesting agent context extraction...")
    start_time = time.perf_counter()
    context = extractor.extract_agent_context(test_incident)
    end_time = time.perf_counter()

    context_time = (end_time - start_time) * 1000
    print(f"Agent context extraction: {context_time:.3f}ms")

    print(f"\nFeature vector shape: {features.shape}")
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Context keys: {len(context.keys())}")

    print("\nPerformance test complete!")


if __name__ == "__main__":
    test_extraction_performance()