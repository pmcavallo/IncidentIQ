"""Synthetic incident data generation module."""

import json
import random
import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class IncidentMetrics:
    """Incident metrics data structure."""
    cpu_usage: float
    memory_usage: float
    disk_io_ops: float
    network_latency_ms: float
    response_time_ms: float
    error_rate: float
    connection_pool_usage: float
    throughput_rps: float
    packet_loss_percent: float
    db_query_time_ms: float


@dataclass
class IncidentContext:
    """Incident context and environmental factors."""
    timestamp: str
    business_event: str
    recent_deployments: List[str]
    traffic_multiplier: float
    geographic_distribution: Dict[str, float]
    feature_flags: List[str]
    historical_incidents: List[Dict[str, Any]]


@dataclass
class IncidentPredictions:
    """Expected model and agent predictions."""
    incident_type: str
    severity: str
    root_cause: str
    confidence: float
    recommended_actions: List[str]
    is_edge_case: bool
    edge_case_type: str


@dataclass
class SyntheticIncident:
    """Complete synthetic incident data structure."""
    incident_id: str
    metrics: IncidentMetrics
    context: IncidentContext
    predictions: IncidentPredictions
    ground_truth: Dict[str, Any]


class SyntheticIncidentGenerator:
    """Generator for realistic incident data with specific edge cases."""

    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        random.seed(seed)
        np.random.seed(seed)

        self.incident_types = [
            "database_performance",
            "network_latency",
            "memory_leak",
            "cpu_spike",
            "disk_io"
        ]

        self.severities = ["low", "medium", "high", "critical"]

    def _generate_timestamp(self, days_back: int = 0, hours_back: int = 0) -> str:
        """Generate a realistic timestamp."""
        now = datetime.datetime.now()
        delta = datetime.timedelta(days=days_back, hours=hours_back)
        return (now - delta).isoformat()

    def _generate_baseline_metrics(self) -> IncidentMetrics:
        """Generate baseline healthy metrics."""
        return IncidentMetrics(
            cpu_usage=random.uniform(15, 35),
            memory_usage=random.uniform(40, 60),
            disk_io_ops=random.uniform(100, 300),
            network_latency_ms=random.uniform(5, 15),
            response_time_ms=random.uniform(120, 200),
            error_rate=random.uniform(0.01, 0.05),
            connection_pool_usage=random.uniform(20, 40),
            throughput_rps=random.uniform(800, 1200),
            packet_loss_percent=random.uniform(0.001, 0.01),
            db_query_time_ms=random.uniform(10, 50)
        )

    def generate_edge_case_1_db_actually_network(self) -> SyntheticIncident:
        """
        Edge case 1: Database symptoms but network is root cause (misleading).

        DB metrics appear degraded but DB internals are healthy.
        Network metrics show the real problem.
        """
        incident_id = f"EDGE_1_{random.randint(1000, 9999)}"

        # Misleading DB metrics (appear bad)
        metrics = IncidentMetrics(
            cpu_usage=32.5,  # Normal DB CPU
            memory_usage=55.8,  # Normal DB memory
            disk_io_ops=285.4,  # Normal DB disk I/O
            network_latency_ms=145.7,  # HIGH - real problem
            response_time_ms=850.3,  # HIGH due to network
            error_rate=0.12,  # Elevated due to timeouts
            connection_pool_usage=89.2,  # HIGH - appears like DB issue
            throughput_rps=450.1,  # Low due to network issues
            packet_loss_percent=2.3,  # HIGH - smoking gun
            db_query_time_ms=45.2  # Normal internal DB performance
        )

        # Historical context with similar incidents
        historical_incidents = [
            {
                "incident_id": "HIST_NET_001",
                "date": "2024-08-15",
                "root_cause": "network_switch_failure",
                "symptoms": ["high_connection_pool", "elevated_response_time"],
                "resolution": "replaced_faulty_network_switch"
            },
            {
                "incident_id": "HIST_NET_002",
                "date": "2024-09-22",
                "root_cause": "network_congestion",
                "symptoms": ["packet_loss", "db_connection_timeouts"],
                "resolution": "traffic_rerouting"
            }
        ]

        context = IncidentContext(
            timestamp=self._generate_timestamp(hours_back=2),
            business_event="normal_operations",
            recent_deployments=["user-service-v2.1.3"],
            traffic_multiplier=1.1,
            geographic_distribution={"us-east": 0.4, "us-west": 0.35, "eu": 0.25},
            feature_flags=["enhanced_caching_v2"],
            historical_incidents=historical_incidents
        )

        predictions = IncidentPredictions(
            incident_type="network_latency",  # Correct classification
            severity="high",
            root_cause="network_infrastructure_degradation",
            confidence=0.75,  # Lower confidence due to misleading symptoms
            recommended_actions=[
                "check_network_switch_health",
                "analyze_packet_loss_patterns",
                "verify_network_routing_tables",
                "investigate_isp_connectivity"
            ],
            is_edge_case=True,
            edge_case_type="misleading_db_symptoms"
        )

        ground_truth = {
            "actual_root_cause": "network_switch_intermittent_failure",
            "misleading_symptoms": ["high_connection_pool_usage", "elevated_db_response_time"],
            "correct_indicators": ["packet_loss", "network_latency", "normal_db_internals"],
            "resolution_time_minutes": 45,
            "business_impact": "moderate"
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    def generate_edge_case_2_false_positive_black_friday(self) -> SyntheticIncident:
        """
        Edge case 2: Black Friday false positive (high load is expected).

        Metrics breach thresholds but it's expected high traffic.
        """
        incident_id = f"EDGE_2_{random.randint(1000, 9999)}"

        # Elevated but expected metrics
        metrics = IncidentMetrics(
            cpu_usage=78.5,  # High but expected
            memory_usage=82.1,  # High but expected
            disk_io_ops=2850.7,  # High but expected
            network_latency_ms=12.3,  # Normal
            response_time_ms=520.4,  # Slightly above 500ms threshold
            error_rate=0.04,  # Normal despite high load
            connection_pool_usage=85.6,  # High but handling load well
            throughput_rps=9800.2,  # 12x normal traffic
            packet_loss_percent=0.008,  # Normal
            db_query_time_ms=65.8  # Slightly elevated but acceptable
        )

        # Black Friday context
        context = IncidentContext(
            timestamp="2024-11-29T14:30:00",  # Black Friday afternoon
            business_event="black_friday_peak_shopping",
            recent_deployments=["checkout-service-v3.2.1", "inventory-service-v1.8.0"],
            traffic_multiplier=12.3,  # Expected 12x traffic
            geographic_distribution={"us-east": 0.45, "us-west": 0.4, "eu": 0.15},
            feature_flags=["black_friday_optimizations", "enhanced_checkout_flow"],
            historical_incidents=[
                {
                    "event": "black_friday_2023",
                    "response_times": [480, 495, 510, 525, 530, 518, 502],
                    "traffic_multiplier": 11.8,
                    "incident_count": 0
                },
                {
                    "event": "black_friday_2022",
                    "response_times": [465, 489, 503, 528, 535, 521, 498],
                    "traffic_multiplier": 9.2,
                    "incident_count": 1
                }
            ]
        )

        predictions = IncidentPredictions(
            incident_type="expected_high_load",  # Should classify as non-incident
            severity="low",  # Should be low despite threshold breach
            root_cause="seasonal_traffic_spike",
            confidence=0.92,  # High confidence it's expected
            recommended_actions=[
                "monitor_error_rates_closely",
                "verify_auto_scaling_active",
                "prepare_additional_capacity_if_needed",
                "confirm_business_event_correlation"
            ],
            is_edge_case=True,
            edge_case_type="false_positive_expected_load"
        )

        ground_truth = {
            "is_actual_incident": False,
            "business_context": "black_friday_shopping_event",
            "threshold_breach_reason": "expected_seasonal_traffic",
            "system_health": "performing_within_expected_parameters",
            "historical_comparison": "response_times_consistent_with_previous_years",
            "action_taken": "continued_monitoring_no_intervention"
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    def generate_edge_case_3_novel_feature_flag(self) -> SyntheticIncident:
        """
        Edge case 3: Novel pattern from feature flag deployment.

        Geographic clustering with feature flag correlation.
        """
        incident_id = f"EDGE_3_{random.randint(1000, 9999)}"

        # Geographically clustered problem
        metrics = IncidentMetrics(
            cpu_usage=45.2,  # Normal overall
            memory_usage=68.7,  # Slightly elevated
            disk_io_ops=520.1,  # Normal
            network_latency_ms=18.5,  # Slightly elevated
            response_time_ms=1250.8,  # High in affected region
            error_rate=0.31,  # High overall due to US-East
            connection_pool_usage=55.4,  # Normal
            throughput_rps=850.3,  # Normal
            packet_loss_percent=0.015,  # Normal
            db_query_time_ms=35.2  # Normal
        )

        context = IncidentContext(
            timestamp=self._generate_timestamp(hours_back=1),
            business_event="normal_operations",
            recent_deployments=["recommendation-engine-v4.1.0"],
            traffic_multiplier=1.05,
            geographic_distribution={
                "us-east": 0.93,  # 93% of errors in US-East
                "us-west": 0.04,
                "eu": 0.02,
                "asia": 0.01
            },
            feature_flags=[
                "ml_recommendations_v4",  # Deployed 45 min ago
                "personalized_search_beta",
                "enhanced_product_sorting"
            ],
            historical_incidents=[
                {
                    "incident_id": "HIST_FF_001",
                    "date": "2024-08-12",
                    "feature_flag": "advanced_filtering_v2",
                    "geographic_impact": "us-west",
                    "symptom_pattern": "memory_leak_in_recommendation_service",
                    "resolution": "feature_flag_rollback"
                }
            ]
        )

        predictions = IncidentPredictions(
            incident_type="feature_flag_regression",
            severity="high",
            root_cause="ml_recommendations_v4_regional_deployment_issue",
            confidence=0.88,
            recommended_actions=[
                "rollback_ml_recommendations_v4_in_us_east",
                "analyze_recommendation_service_logs",
                "check_ml_model_inference_performance",
                "investigate_regional_data_dependencies"
            ],
            is_edge_case=True,
            edge_case_type="novel_geographic_feature_correlation"
        )

        ground_truth = {
            "root_cause": "ml_model_incompatibility_with_us_east_data_format",
            "deployment_correlation": {
                "feature_flag": "ml_recommendations_v4",
                "deployment_time": "45_minutes_ago",
                "affected_region": "us-east"
            },
            "novel_pattern": "geographic_clustering_with_ml_feature_flag",
            "resolution": "feature_flag_config_fix_for_regional_data_format",
            "learning": "ml_models_need_regional_data_format_validation"
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    def _generate_clear_case_incident(self, incident_type: str) -> SyntheticIncident:
        """Generate a clear, straightforward incident case."""
        incident_id = f"CLEAR_{incident_type.upper()}_{random.randint(1000, 9999)}"

        # Generate type-specific metrics
        if incident_type == "database_performance":
            metrics = IncidentMetrics(
                cpu_usage=85.7,
                memory_usage=78.2,
                disk_io_ops=3500.5,
                network_latency_ms=8.3,
                response_time_ms=2850.1,
                error_rate=0.08,
                connection_pool_usage=95.2,
                throughput_rps=320.1,
                packet_loss_percent=0.005,
                db_query_time_ms=1250.8
            )
            root_cause = "database_lock_contention"
            actions = ["analyze_query_performance", "check_index_usage", "review_lock_waits"]

        elif incident_type == "network_latency":
            metrics = IncidentMetrics(
                cpu_usage=35.2,
                memory_usage=52.1,
                disk_io_ops=200.3,
                network_latency_ms=285.7,
                response_time_ms=1850.4,
                error_rate=0.15,
                connection_pool_usage=45.8,
                throughput_rps=450.2,
                packet_loss_percent=3.2,
                db_query_time_ms=35.1
            )
            root_cause = "network_infrastructure_degradation"
            actions = ["check_network_hardware", "analyze_routing", "contact_isp"]

        elif incident_type == "memory_leak":
            metrics = IncidentMetrics(
                cpu_usage=68.5,
                memory_usage=92.8,
                disk_io_ops=180.5,
                network_latency_ms=12.1,
                response_time_ms=950.3,
                error_rate=0.12,
                connection_pool_usage=55.2,
                throughput_rps=680.4,
                packet_loss_percent=0.008,
                db_query_time_ms=45.2
            )
            root_cause = "application_memory_leak"
            actions = ["analyze_heap_dumps", "check_object_retention", "restart_service"]

        elif incident_type == "cpu_spike":
            metrics = IncidentMetrics(
                cpu_usage=98.5,
                memory_usage=65.2,
                disk_io_ops=250.8,
                network_latency_ms=15.2,
                response_time_ms=3200.1,
                error_rate=0.25,
                connection_pool_usage=85.7,
                throughput_rps=150.3,
                packet_loss_percent=0.012,
                db_query_time_ms=55.8
            )
            root_cause = "cpu_intensive_process"
            actions = ["identify_cpu_intensive_processes", "analyze_profiler_data", "optimize_algorithms"]

        else:  # disk_io
            metrics = IncidentMetrics(
                cpu_usage=45.8,
                memory_usage=58.7,
                disk_io_ops=8500.2,
                network_latency_ms=10.5,
                response_time_ms=2200.5,
                error_rate=0.18,
                connection_pool_usage=75.3,
                throughput_rps=380.1,
                packet_loss_percent=0.006,
                db_query_time_ms=850.3
            )
            root_cause = "disk_io_bottleneck"
            actions = ["check_disk_utilization", "analyze_io_patterns", "consider_ssd_upgrade"]

        context = IncidentContext(
            timestamp=self._generate_timestamp(hours_back=random.randint(1, 48)),
            business_event="normal_operations",
            recent_deployments=[f"{incident_type.replace('_', '-')}-service-v1.{random.randint(1,9)}.{random.randint(0,9)}"],
            traffic_multiplier=random.uniform(0.8, 1.3),
            geographic_distribution={"us-east": 0.4, "us-west": 0.35, "eu": 0.25},
            feature_flags=[f"optimization_{random.randint(1,5)}"],
            historical_incidents=[]
        )

        predictions = IncidentPredictions(
            incident_type=incident_type,
            severity=random.choice(["medium", "high"]),
            root_cause=root_cause,
            confidence=random.uniform(0.85, 0.95),
            recommended_actions=actions,
            is_edge_case=False,
            edge_case_type="none"
        )

        ground_truth = {
            "incident_type": incident_type,
            "clear_symptoms": True,
            "resolution_straightforward": True
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    def generate_training_dataset(self, n_samples: int = 1000) -> List[SyntheticIncident]:
        """
        Generate balanced training dataset.

        Args:
            n_samples: Total number of samples (default 1000)

        Returns:
            List of synthetic incidents (70% clear cases, 30% edge cases)
        """
        incidents = []

        # Calculate distribution
        n_clear_cases = int(n_samples * 0.7)  # 70%
        n_edge_cases = n_samples - n_clear_cases  # 30%

        # Generate clear cases (distributed across incident types)
        clear_cases_per_type = n_clear_cases // len(self.incident_types)

        for incident_type in self.incident_types:
            for _ in range(clear_cases_per_type):
                incidents.append(self._generate_clear_case_incident(incident_type))

        # Generate remaining clear cases
        remaining_clear = n_clear_cases - (clear_cases_per_type * len(self.incident_types))
        for _ in range(remaining_clear):
            incident_type = random.choice(self.incident_types)
            incidents.append(self._generate_clear_case_incident(incident_type))

        # Generate edge cases (distributed equally)
        edge_cases_per_type = n_edge_cases // 3

        # Edge case 1: DB symptoms but network root cause
        for _ in range(edge_cases_per_type):
            incidents.append(self.generate_edge_case_1_db_actually_network())

        # Edge case 2: Black Friday false positive
        for _ in range(edge_cases_per_type):
            incidents.append(self.generate_edge_case_2_false_positive_black_friday())

        # Edge case 3: Novel feature flag pattern
        remaining_edge = n_edge_cases - (edge_cases_per_type * 2)
        for _ in range(remaining_edge):
            incidents.append(self.generate_edge_case_3_novel_feature_flag())

        # Shuffle the dataset
        random.shuffle(incidents)

        return incidents

    def save_edge_cases_to_json(self, output_dir: str = "demo_data") -> None:
        """
        Save edge cases to JSON files for analysis.

        Args:
            output_dir: Directory to save JSON files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate each edge case
        edge_case_1 = self.generate_edge_case_1_db_actually_network()
        edge_case_2 = self.generate_edge_case_2_false_positive_black_friday()
        edge_case_3 = self.generate_edge_case_3_novel_feature_flag()

        # Save to JSON files
        with open(output_path / "edge_case_1_db_network.json", "w") as f:
            json.dump(asdict(edge_case_1), f, indent=2)

        with open(output_path / "edge_case_2_black_friday.json", "w") as f:
            json.dump(asdict(edge_case_2), f, indent=2)

        with open(output_path / "edge_case_3_feature_flag.json", "w") as f:
            json.dump(asdict(edge_case_3), f, indent=2)

        print(f"Edge cases saved to {output_path}/")
        print("Files created:")
        print("- edge_case_1_db_network.json")
        print("- edge_case_2_black_friday.json")
        print("- edge_case_3_feature_flag.json")


def main():
    """Main function for testing the generator."""
    print("Synthetic Incident Generator")
    print("=" * 40)

    # Initialize generator
    generator = SyntheticIncidentGenerator(seed=42)

    # Generate and save edge cases
    print("\nGenerating edge cases...")
    generator.save_edge_cases_to_json()

    # Generate training dataset sample
    print("\nGenerating training dataset sample (100 incidents)...")
    training_data = generator.generate_training_dataset(n_samples=100)

    # Analysis
    edge_cases = [inc for inc in training_data if inc.predictions.is_edge_case]
    clear_cases = [inc for inc in training_data if not inc.predictions.is_edge_case]

    print(f"\nDataset Analysis:")
    print(f"- Total incidents: {len(training_data)}")
    print(f"- Clear cases: {len(clear_cases)} ({len(clear_cases)/len(training_data)*100:.1f}%)")
    print(f"- Edge cases: {len(edge_cases)} ({len(edge_cases)/len(training_data)*100:.1f}%)")

    # Incident type distribution
    type_counts = {}
    for incident in training_data:
        inc_type = incident.predictions.incident_type
        type_counts[inc_type] = type_counts.get(inc_type, 0) + 1

    print(f"\nIncident Type Distribution:")
    for inc_type, count in sorted(type_counts.items()):
        print(f"- {inc_type}: {count}")

    # Edge case type distribution
    edge_type_counts = {}
    for incident in edge_cases:
        edge_type = incident.predictions.edge_case_type
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

    print(f"\nEdge Case Type Distribution:")
    for edge_type, count in sorted(edge_type_counts.items()):
        print(f"- {edge_type}: {count}")

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()