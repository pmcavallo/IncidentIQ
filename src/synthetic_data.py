"""Synthetic incident data generation for binary classification.

Generates training data with binary labels ('normal', 'incident') and
5 edge cases demonstrating agent value.
"""

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
    """Generator for realistic incident data with binary classification."""

    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        random.seed(seed)
        np.random.seed(seed)

        # Specific root causes (for agent investigation)
        self.incident_root_causes = [
            "database_performance_degradation",
            "network_infrastructure_issue",
            "memory_leak",
            "cpu_intensive_process",
            "disk_io_bottleneck",
            "connection_pool_exhaustion"
        ]

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

    # ========================================================================
    # EDGE CASE 1: FALSE POSITIVE - Black Friday Normal Traffic
    # ========================================================================
    def generate_edge_case_1_false_positive_black_friday(self) -> SyntheticIncident:
        """Edge Case 1: FALSE POSITIVE - Black Friday normal traffic.

        Model: 'incident' (95% confidence) - thinks system degradation
        Reality: Normal Black Friday traffic (12x baseline)
        Agent value: Prevent unnecessary scaling ($47K cost)
        """
        incident_id = f"EDGE_1_FP_BF_{random.randint(1000, 9999)}"

        # High metrics but expected for Black Friday
        metrics = IncidentMetrics(
            cpu_usage=78.5,  # High but expected
            memory_usage=82.1,  # High but expected
            disk_io_ops=2850.7,  # High but expected
            network_latency_ms=12.3,  # Normal
            response_time_ms=520.4,  # Slightly above threshold
            error_rate=0.04,  # Normal despite high load
            connection_pool_usage=85.6,  # High but handling well
            throughput_rps=9800.2,  # 12x normal traffic
            packet_loss_percent=0.008,  # Normal
            db_query_time_ms=65.8  # Slightly elevated but acceptable
        )

        context = IncidentContext(
            timestamp="2024-11-29T14:30:00",  # Black Friday afternoon
            business_event="black_friday_peak_shopping",
            recent_deployments=["checkout-service-v3.2.1", "inventory-service-v1.8.0"],
            traffic_multiplier=12.3,
            geographic_distribution={"us-east": 0.45, "us-west": 0.4, "eu": 0.15},
            feature_flags=["black_friday_optimizations", "enhanced_checkout_flow"],
            historical_incidents=[
                {
                    "event": "black_friday_2023",
                    "response_times": [480, 495, 510, 525, 530, 518, 502],
                    "traffic_multiplier": 11.8,
                    "incident_count": 0
                }
            ]
        )

        predictions = IncidentPredictions(
            incident_type="normal",  # What agents determine
            severity="none",
            root_cause="expected_black_friday_traffic",
            confidence=0.92,
            recommended_actions=[
                "continue_monitoring",
                "verify_auto_scaling_active",
                "no_intervention_required"
            ],
            is_edge_case=True,
            edge_case_type="false_positive_expected_event"
        )

        ground_truth = {
            "actual_label": "normal",  # Binary label
            "actual_root_cause": "expected_black_friday_traffic",
            "why_tricky": "Metrics exceed typical thresholds, but within expected range for Black Friday",
            "model_prediction": "incident",  # What model would predict
            "model_confidence": 0.95,
            "model_would_do": "Scale infrastructure immediately (cost: $47K)",
            "agent_finding": "Normal Black Friday traffic pattern, metrics consistent with historical data",
            "agent_recommendation": "Continue monitoring, no intervention needed",
            "outcome": "Prevented $47K unnecessary cloud scaling costs"
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    # ========================================================================
    # EDGE CASE 2: FALSE NEGATIVE - Gradual Memory Leak
    # ========================================================================
    def generate_edge_case_2_false_negative_memory_leak(self) -> SyntheticIncident:
        """Edge Case 2: FALSE NEGATIVE - Gradual memory leak missed.

        Model: 'normal' (88% confidence) - current metrics look fine
        Reality: Progressive memory leak that will cause failure in 2 hours
        Agent value: Catch early warning signs before outage
        """
        incident_id = f"EDGE_2_FN_ML_{random.randint(1000, 9999)}"

        # Metrics currently normal but trending dangerously
        metrics = IncidentMetrics(
            cpu_usage=28.5,  # Normal
            memory_usage=67.2,  # Creeping up (was 45% 2hr ago)
            disk_io_ops=220.4,  # Normal
            network_latency_ms=9.5,  # Normal
            response_time_ms=185.3,  # Normal
            error_rate=0.03,  # Normal
            connection_pool_usage=35.7,  # Normal
            throughput_rps=950.2,  # Normal
            packet_loss_percent=0.005,  # Normal
            db_query_time_ms=38.1  # Normal
        )

        context = IncidentContext(
            timestamp=self._generate_timestamp(hours_back=4),
            business_event="normal_operations",
            recent_deployments=["user-profile-service-v2.4.1"],  # 6 hours ago
            traffic_multiplier=1.05,
            geographic_distribution={"us-east": 0.4, "us-west": 0.35, "eu": 0.25},
            feature_flags=["enhanced_caching_v3"],
            historical_incidents=[
                {
                    "incident_id": "HIST_ML_001",
                    "date": "2024-07-15",
                    "symptom": "gradual_memory_increase",
                    "resolution": "cache_ttl_configuration_fix"
                }
            ]
        )

        predictions = IncidentPredictions(
            incident_type="incident",  # What agents determine
            severity="high",
            root_cause="cache_memory_leak_from_deployment",
            confidence=0.82,
            recommended_actions=[
                "analyze_memory_growth_trend",
                "review_cache_configuration",
                "prepare_service_restart",
                "rollback_if_pattern_continues"
            ],
            is_edge_case=True,
            edge_case_type="false_negative_early_warning"
        )

        ground_truth = {
            "actual_label": "incident",  # Binary label
            "actual_root_cause": "cache_ttl_misconfiguration_causing_memory_leak",
            "why_tricky": "Current metrics look normal, requires trend analysis to spot danger",
            "model_prediction": "normal",  # What model would predict
            "model_confidence": 0.88,
            "model_would_do": "No action, metrics appear healthy",
            "agent_finding": "Memory increasing 3.5% per hour since deployment, will hit 95% in 2 hours",
            "agent_recommendation": "Immediate rollback of user-profile-service-v2.4.1",
            "outcome": "Prevented production outage, caught issue 2 hours before critical threshold"
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    # ========================================================================
    # EDGE CASE 3: WRONG ROOT CAUSE - DB Symptoms, Network Cause
    # ========================================================================
    def generate_edge_case_3_wrong_root_cause_db_network(self) -> SyntheticIncident:
        """Edge Case 3: WRONG ROOT CAUSE - Database symptoms, network cause.

        Model: 'incident' (91% confidence) - correct label, wrong diagnosis
        Reality: Network issue affecting DB connections, DB itself healthy
        Agent value: Prevent wasted time on wrong fix
        """
        incident_id = f"EDGE_3_WC_DBN_{random.randint(1000, 9999)}"

        # Misleading DB metrics (caused by network)
        metrics = IncidentMetrics(
            cpu_usage=32.5,  # Normal DB CPU
            memory_usage=55.8,  # Normal DB memory
            disk_io_ops=285.4,  # Normal DB disk
            network_latency_ms=145.7,  # HIGH - smoking gun
            response_time_ms=850.3,  # HIGH due to network
            error_rate=0.12,  # Elevated due to timeouts
            connection_pool_usage=89.2,  # HIGH - appears like DB issue
            throughput_rps=450.1,  # Low due to network issues
            packet_loss_percent=2.3,  # HIGH - real problem
            db_query_time_ms=45.2  # Normal internal DB performance
        )

        context = IncidentContext(
            timestamp=self._generate_timestamp(hours_back=2),
            business_event="normal_operations",
            recent_deployments=["database-config-update-v3.1.2"],
            traffic_multiplier=1.1,
            geographic_distribution={"us-east": 0.4, "us-west": 0.35, "eu": 0.25},
            feature_flags=["enhanced_db_caching"],
            historical_incidents=[
                {
                    "incident_id": "HIST_NET_001",
                    "date": "2024-08-15",
                    "symptoms": ["high_connection_pool", "elevated_response_time"],
                    "actual_cause": "network_switch_failure"
                }
            ]
        )

        predictions = IncidentPredictions(
            incident_type="incident",  # Correct label
            severity="high",
            root_cause="network_infrastructure_degradation",  # Correct cause
            confidence=0.85,
            recommended_actions=[
                "check_network_switch_health",
                "analyze_packet_loss_patterns",
                "verify_network_routing",
                "do_not_restart_database"
            ],
            is_edge_case=True,
            edge_case_type="misleading_symptoms_wrong_root_cause"
        )

        ground_truth = {
            "actual_label": "incident",  # Binary label (correct)
            "actual_root_cause": "network_switch_intermittent_failure",
            "why_tricky": "DB metrics misleading - high connection pool suggests DB issue, but network is culprit",
            "model_prediction": "incident",  # Correct label
            "model_confidence": 0.91,
            "model_would_do": "Restart database, scale DB resources (45min downtime, $0 fix)",
            "agent_finding": "DB internals healthy (45ms query time), network packet loss 2.3% is real problem",
            "agent_recommendation": "Replace failing network switch, do NOT touch database",
            "outcome": "Prevented 45min unnecessary DB restart, fixed in 15min by replacing switch"
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    # ========================================================================
    # EDGE CASE 4: NOVEL PATTERN - Feature Flag Interaction
    # ========================================================================
    def generate_edge_case_4_novel_pattern_feature_flag(self) -> SyntheticIncident:
        """Edge Case 4: NOVEL PATTERN - Feature flag interaction.

        Model: 'incident' (68% confidence) - low confidence, uncertain
        Reality: Complex interaction between 2 feature flags causing memory leak in 2% of traffic
        Agent value: Identify novel pattern through correlation analysis
        """
        incident_id = f"EDGE_4_NP_FF_{random.randint(1000, 9999)}"

        # Subtle, unusual metric pattern
        metrics = IncidentMetrics(
            cpu_usage=45.2,  # Normal overall
            memory_usage=68.7,  # Slightly elevated
            disk_io_ops=520.1,  # Normal
            network_latency_ms=18.5,  # Slightly elevated
            response_time_ms=1250.8,  # High in affected segment
            error_rate=0.08,  # Moderate
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
                "us-east": 0.93,  # 93% of errors concentrated here
                "us-west": 0.04,
                "eu": 0.02,
                "asia": 0.01
            },
            feature_flags=[
                "ml_recommendations_v4",  # Deployed 45 min ago
                "personalized_search_beta",  # Enabled 2 weeks ago
                "enhanced_product_sorting"
            ],
            historical_incidents=[]
        )

        predictions = IncidentPredictions(
            incident_type="incident",  # Agents identify this
            severity="medium",
            root_cause="feature_flag_interaction_memory_leak",
            confidence=0.78,
            recommended_actions=[
                "disable_ml_recommendations_v4_for_personalized_search_users",
                "analyze_memory_patterns_in_us_east",
                "investigate_flag_interaction_logic",
                "prepare_targeted_rollback"
            ],
            is_edge_case=True,
            edge_case_type="novel_pattern_feature_correlation"
        )

        ground_truth = {
            "actual_label": "incident",  # Binary label
            "actual_root_cause": "ml_recommendations_v4_memory_leak_when_combined_with_personalized_search",
            "why_tricky": "Novel pattern - never seen this flag combination, affects only 2% of users",
            "model_prediction": "incident",  # Correct label but...
            "model_confidence": 0.68,  # Low confidence
            "model_would_do": "Generic incident response, broad rollback affecting all users",
            "agent_finding": "Memory leak ONLY for users with both ml_recommendations_v4 AND personalized_search_beta",
            "agent_recommendation": "Targeted fix: disable ml_recommendations_v4 for 2% affected segment, keep for 98%",
            "outcome": "Surgical fix affecting 2% vs broad rollback affecting 100% of users"
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    # ========================================================================
    # EDGE CASE 5: CASCADE EARLY DETECTION
    # ========================================================================
    def generate_edge_case_5_cascade_early_detection(self) -> SyntheticIncident:
        """Edge Case 5: CASCADE EARLY DETECTION - Subtle cross-service pattern.

        Model: 'normal' (82% confidence) - metrics within bounds
        Reality: Early signs of cascade failure pattern (auth → API → DB)
        Agent value: Detect multi-service correlation before cascade
        """
        incident_id = f"EDGE_5_CE_CS_{random.randint(1000, 9999)}"

        # Metrics individually normal, pattern is key
        metrics = IncidentMetrics(
            cpu_usage=42.8,  # Normal
            memory_usage=58.3,  # Normal
            disk_io_ops=280.5,  # Normal
            network_latency_ms=22.5,  # Slightly elevated
            response_time_ms=285.7,  # Slightly elevated
            error_rate=0.06,  # Slightly elevated
            connection_pool_usage=52.3,  # Normal
            throughput_rps=780.4,  # Slightly low
            packet_loss_percent=0.012,  # Normal
            db_query_time_ms=58.2  # Slightly elevated
        )

        context = IncidentContext(
            timestamp=self._generate_timestamp(hours_back=1),
            business_event="normal_operations",
            recent_deployments=["auth-service-v3.2.0"],  # 3 hours ago
            traffic_multiplier=0.95,  # Slightly declining
            geographic_distribution={"us-east": 0.4, "us-west": 0.35, "eu": 0.25},
            feature_flags=["enhanced_auth_validation"],
            historical_incidents=[
                {
                    "incident_id": "HIST_CAS_001",
                    "date": "2024-06-22",
                    "pattern": "auth_slowdown_cascade",
                    "services_affected": ["auth", "api-gateway", "database"],
                    "outcome": "30min full outage"
                }
            ]
        )

        predictions = IncidentPredictions(
            incident_type="incident",  # Agents identify early
            severity="high",
            root_cause="auth_service_slowdown_cascade_pattern",
            confidence=0.79,
            recommended_actions=[
                "rollback_auth_service_v3.2.0_immediately",
                "monitor_api_gateway_connection_pool",
                "prepare_database_connection_scaling",
                "alert_on_call_engineer"
            ],
            is_edge_case=True,
            edge_case_type="cross_service_cascade_early_warning"
        )

        ground_truth = {
            "actual_label": "incident",  # Binary label
            "actual_root_cause": "auth_service_validation_causing_connection_buildup",
            "why_tricky": "Individual metrics normal, danger is in cross-service correlation pattern",
            "model_prediction": "normal",  # Misses pattern
            "model_confidence": 0.82,
            "model_would_do": "No action, all metrics within acceptable ranges",
            "agent_finding": "Auth latency +40ms → API connections +15% → DB query queue forming. Classic cascade pattern.",
            "agent_recommendation": "Immediate auth-service rollback to prevent cascade failure",
            "outcome": "Prevented full cascade failure, caught 45min before critical threshold"
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    # ========================================================================
    # NORMAL EXAMPLES (30% of dataset)
    # ========================================================================
    def _generate_normal_example(self) -> SyntheticIncident:
        """Generate normal (non-incident) examples.

        Examples: maintenance windows, load tests, gradual scaling, temp spikes
        """
        incident_id = f"NORMAL_{random.randint(1000, 9999)}"

        normal_types = [
            "scheduled_maintenance",
            "load_test",
            "gradual_scaling",
            "temporary_spike",
            "expected_high_traffic"
        ]

        normal_type = random.choice(normal_types)

        if normal_type == "scheduled_maintenance":
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(20, 40),
                memory_usage=random.uniform(45, 65),
                disk_io_ops=random.uniform(150, 400),
                network_latency_ms=random.uniform(8, 18),
                response_time_ms=random.uniform(180, 280),
                error_rate=random.uniform(0.02, 0.06),
                connection_pool_usage=random.uniform(25, 50),
                throughput_rps=random.uniform(700, 1100),
                packet_loss_percent=random.uniform(0.005, 0.015),
                db_query_time_ms=random.uniform(20, 60)
            )
            business_event = "scheduled_database_maintenance"
            root_cause = "planned_maintenance_window"

        elif normal_type == "load_test":
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(60, 85),
                memory_usage=random.uniform(65, 85),
                disk_io_ops=random.uniform(1500, 3000),
                network_latency_ms=random.uniform(10, 20),
                response_time_ms=random.uniform(300, 500),
                error_rate=random.uniform(0.03, 0.08),
                connection_pool_usage=random.uniform(60, 85),
                throughput_rps=random.uniform(5000, 8000),
                packet_loss_percent=random.uniform(0.008, 0.02),
                db_query_time_ms=random.uniform(40, 80)
            )
            business_event = "performance_load_testing"
            root_cause = "planned_load_test_execution"

        elif normal_type == "gradual_scaling":
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(50, 70),
                memory_usage=random.uniform(55, 75),
                disk_io_ops=random.uniform(500, 1000),
                network_latency_ms=random.uniform(12, 22),
                response_time_ms=random.uniform(250, 400),
                error_rate=random.uniform(0.03, 0.07),
                connection_pool_usage=random.uniform(50, 70),
                throughput_rps=random.uniform(2000, 4000),
                packet_loss_percent=random.uniform(0.008, 0.018),
                db_query_time_ms=random.uniform(35, 70)
            )
            business_event = "gradual_traffic_increase"
            root_cause = "expected_business_growth"

        elif normal_type == "temporary_spike":
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(65, 80),
                memory_usage=random.uniform(60, 75),
                disk_io_ops=random.uniform(800, 1500),
                network_latency_ms=random.uniform(10, 18),
                response_time_ms=random.uniform(280, 450),
                error_rate=random.uniform(0.04, 0.08),
                connection_pool_usage=random.uniform(55, 75),
                throughput_rps=random.uniform(3000, 5000),
                packet_loss_percent=random.uniform(0.009, 0.018),
                db_query_time_ms=random.uniform(45, 75)
            )
            business_event = "promotional_campaign"
            root_cause = "temporary_marketing_driven_traffic"

        else:  # expected_high_traffic
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(70, 85),
                memory_usage=random.uniform(65, 80),
                disk_io_ops=random.uniform(1200, 2500),
                network_latency_ms=random.uniform(12, 20),
                response_time_ms=random.uniform(350, 520),
                error_rate=random.uniform(0.04, 0.09),
                connection_pool_usage=random.uniform(65, 85),
                throughput_rps=random.uniform(6000, 10000),
                packet_loss_percent=random.uniform(0.01, 0.02),
                db_query_time_ms=random.uniform(50, 85)
            )
            business_event = "seasonal_high_traffic_period"
            root_cause = "expected_seasonal_traffic_pattern"

        context = IncidentContext(
            timestamp=self._generate_timestamp(hours_back=random.randint(1, 48)),
            business_event=business_event,
            recent_deployments=[],
            traffic_multiplier=random.uniform(1.2, 3.5),
            geographic_distribution={"us-east": 0.4, "us-west": 0.35, "eu": 0.25},
            feature_flags=[],
            historical_incidents=[]
        )

        predictions = IncidentPredictions(
            incident_type="normal",
            severity="none",
            root_cause=root_cause,
            confidence=random.uniform(0.80, 0.95),
            recommended_actions=["continue_monitoring", "no_action_required"],
            is_edge_case=False,
            edge_case_type="none"
        )

        ground_truth = {
            "actual_label": "normal",
            "actual_root_cause": root_cause,
            "is_expected_behavior": True
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    # ========================================================================
    # INCIDENT EXAMPLES (70% of dataset)
    # ========================================================================
    def _generate_incident_example(self) -> SyntheticIncident:
        """Generate clear incident examples (non-edge cases)."""
        incident_id = f"INCIDENT_{random.randint(1000, 9999)}"

        incident_cause = random.choice(self.incident_root_causes)

        if incident_cause == "database_performance_degradation":
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(75, 90),
                memory_usage=random.uniform(70, 85),
                disk_io_ops=random.uniform(3000, 5000),
                network_latency_ms=random.uniform(8, 15),
                response_time_ms=random.uniform(2000, 3500),
                error_rate=random.uniform(0.10, 0.25),
                connection_pool_usage=random.uniform(85, 98),
                throughput_rps=random.uniform(200, 450),
                packet_loss_percent=random.uniform(0.005, 0.015),
                db_query_time_ms=random.uniform(800, 1500)
            )
            severity = "high"
            actions = ["analyze_slow_queries", "check_index_usage", "review_db_locks"]

        elif incident_cause == "network_infrastructure_issue":
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(30, 50),
                memory_usage=random.uniform(45, 65),
                disk_io_ops=random.uniform(150, 400),
                network_latency_ms=random.uniform(200, 400),
                response_time_ms=random.uniform(1500, 2800),
                error_rate=random.uniform(0.15, 0.35),
                connection_pool_usage=random.uniform(40, 65),
                throughput_rps=random.uniform(300, 600),
                packet_loss_percent=random.uniform(2.5, 5.0),
                db_query_time_ms=random.uniform(30, 70)
            )
            severity = "critical"
            actions = ["check_network_hardware", "analyze_routing_tables", "contact_isp"]

        elif incident_cause == "memory_leak":
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(55, 75),
                memory_usage=random.uniform(88, 98),
                disk_io_ops=random.uniform(180, 350),
                network_latency_ms=random.uniform(10, 20),
                response_time_ms=random.uniform(800, 1500),
                error_rate=random.uniform(0.12, 0.28),
                connection_pool_usage=random.uniform(50, 70),
                throughput_rps=random.uniform(500, 800),
                packet_loss_percent=random.uniform(0.008, 0.018),
                db_query_time_ms=random.uniform(40, 80)
            )
            severity = "high"
            actions = ["analyze_heap_dumps", "check_object_retention", "restart_service"]

        elif incident_cause == "cpu_intensive_process":
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(92, 99),
                memory_usage=random.uniform(55, 75),
                disk_io_ops=random.uniform(200, 500),
                network_latency_ms=random.uniform(12, 22),
                response_time_ms=random.uniform(2500, 4000),
                error_rate=random.uniform(0.20, 0.40),
                connection_pool_usage=random.uniform(75, 90),
                throughput_rps=random.uniform(100, 300),
                packet_loss_percent=random.uniform(0.01, 0.025),
                db_query_time_ms=random.uniform(50, 100)
            )
            severity = "critical"
            actions = ["identify_cpu_processes", "analyze_profiler_data", "kill_runaway_process"]

        elif incident_cause == "disk_io_bottleneck":
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(40, 60),
                memory_usage=random.uniform(50, 70),
                disk_io_ops=random.uniform(7000, 12000),
                network_latency_ms=random.uniform(10, 18),
                response_time_ms=random.uniform(1800, 3200),
                error_rate=random.uniform(0.15, 0.30),
                connection_pool_usage=random.uniform(65, 85),
                throughput_rps=random.uniform(250, 500),
                packet_loss_percent=random.uniform(0.008, 0.018),
                db_query_time_ms=random.uniform(600, 1200)
            )
            severity = "high"
            actions = ["check_disk_utilization", "analyze_io_patterns", "migrate_to_ssd"]

        else:  # connection_pool_exhaustion
            metrics = IncidentMetrics(
                cpu_usage=random.uniform(50, 70),
                memory_usage=random.uniform(60, 80),
                disk_io_ops=random.uniform(400, 800),
                network_latency_ms=random.uniform(15, 30),
                response_time_ms=random.uniform(3000, 5000),
                error_rate=random.uniform(0.25, 0.45),
                connection_pool_usage=random.uniform(95, 100),
                throughput_rps=random.uniform(150, 350),
                packet_loss_percent=random.uniform(0.01, 0.025),
                db_query_time_ms=random.uniform(80, 150)
            )
            severity = "critical"
            actions = ["increase_pool_size", "check_connection_leaks", "restart_services"]

        context = IncidentContext(
            timestamp=self._generate_timestamp(hours_back=random.randint(1, 48)),
            business_event="normal_operations",
            recent_deployments=[f"service-v{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}"],
            traffic_multiplier=random.uniform(0.9, 1.5),
            geographic_distribution={"us-east": 0.4, "us-west": 0.35, "eu": 0.25},
            feature_flags=[],
            historical_incidents=[]
        )

        predictions = IncidentPredictions(
            incident_type="incident",
            severity=severity,
            root_cause=incident_cause,
            confidence=random.uniform(0.85, 0.95),
            recommended_actions=actions,
            is_edge_case=False,
            edge_case_type="none"
        )

        ground_truth = {
            "actual_label": "incident",
            "actual_root_cause": incident_cause,
            "clear_symptoms": True
        }

        return SyntheticIncident(
            incident_id=incident_id,
            metrics=metrics,
            context=context,
            predictions=predictions,
            ground_truth=ground_truth
        )

    # ========================================================================
    # HELPER METHODS FOR EVALUATION SYSTEM
    # ========================================================================
    def generate_edge_case_incident(self) -> SyntheticIncident:
        """Generate a random edge case incident for evaluation."""
        edge_case_generators = [
            self.generate_edge_case_1_false_positive_black_friday,
            self.generate_edge_case_2_false_negative_memory_leak,
            self.generate_edge_case_3_wrong_root_cause_db_network,
            self.generate_edge_case_4_novel_pattern_feature_flag,
            self.generate_edge_case_5_cascade_early_detection
        ]
        generator_func = random.choice(edge_case_generators)
        return generator_func()

    def generate_standard_incident(self) -> SyntheticIncident:
        """Generate a random standard (non-edge case) incident for evaluation."""
        # 50/50 split between incident and normal
        if random.random() < 0.7:
            return self._generate_incident_example()
        else:
            return self._generate_normal_example()

    # ========================================================================
    # TRAINING DATASET GENERATION
    # ========================================================================
    def generate_training_dataset(self, n_samples: int = 1000) -> List[SyntheticIncident]:
        """Generate balanced training dataset for binary classification.

        Args:
            n_samples: Total number of samples (default 1000)

        Returns:
            List of synthetic incidents (70% incident, 30% normal)
        """
        incidents = []

        # Calculate distribution: 70% incident, 30% normal
        n_incidents = int(n_samples * 0.7)
        n_normal = n_samples - n_incidents

        # Generate incident examples (70%)
        for _ in range(n_incidents):
            incidents.append(self._generate_incident_example())

        # Generate normal examples (30%)
        for _ in range(n_normal):
            incidents.append(self._generate_normal_example())

        # Shuffle the dataset
        random.shuffle(incidents)

        return incidents

    # ========================================================================
    # EDGE CASE JSON GENERATION
    # ========================================================================
    def save_edge_cases_to_json(self, output_dir: str = "demo_data") -> None:
        """Save 5 edge cases to JSON files for Streamlit demo.

        Args:
            output_dir: Directory to save JSON files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate all 5 edge cases
        edge_case_1 = self.generate_edge_case_1_false_positive_black_friday()
        edge_case_2 = self.generate_edge_case_2_false_negative_memory_leak()
        edge_case_3 = self.generate_edge_case_3_wrong_root_cause_db_network()
        edge_case_4 = self.generate_edge_case_4_novel_pattern_feature_flag()
        edge_case_5 = self.generate_edge_case_5_cascade_early_detection()

        # Save to JSON files
        edge_cases = [
            (edge_case_1, "edge_case_1_false_positive_black_friday.json"),
            (edge_case_2, "edge_case_2_false_negative_memory_leak.json"),
            (edge_case_3, "edge_case_3_wrong_root_cause_db_network.json"),
            (edge_case_4, "edge_case_4_novel_pattern_feature_flag.json"),
            (edge_case_5, "edge_case_5_cascade_early_detection.json")
        ]

        for edge_case, filename in edge_cases:
            filepath = output_path / filename
            with open(filepath, "w") as f:
                json.dump(asdict(edge_case), f, indent=2)

        print(f"\n[SUCCESS] Edge cases saved to {output_path}/")
        print(f"\nFiles created:")
        for _, filename in edge_cases:
            print(f"  - {filename}")
        print(f"\nEdge Case Summary:")
        print(f"  1. FALSE POSITIVE: Black Friday traffic (prevent $47K unnecessary scaling)")
        print(f"  2. FALSE NEGATIVE: Memory leak (catch 2 hours before outage)")
        print(f"  3. WRONG ROOT CAUSE: DB symptoms, network cause (prevent 45min wasted fix)")
        print(f"  4. NOVEL PATTERN: Feature flag interaction (surgical fix for 2% vs 100%)")
        print(f"  5. CASCADE EARLY DETECTION: Cross-service pattern (prevent full outage)")


def main():
    """Main function for testing the generator."""
    print("=" * 60)
    print("BINARY CLASSIFICATION - Synthetic Incident Generator")
    print("=" * 60)

    # Initialize generator
    generator = SyntheticIncidentGenerator(seed=42)

    # Generate and save edge cases
    print("\n[1/3] Generating 5 edge cases for demo...")
    generator.save_edge_cases_to_json()

    # Generate training dataset sample
    print("\n[2/3] Generating training dataset sample (100 incidents)...")
    training_data = generator.generate_training_dataset(n_samples=100)

    # Analysis
    incidents = [inc for inc in training_data if inc.ground_truth.get('actual_label') == 'incident']
    normals = [inc for inc in training_data if inc.ground_truth.get('actual_label') == 'normal']
    edge_cases = [inc for inc in training_data if inc.predictions.is_edge_case]

    print(f"\n[3/3] Dataset Analysis:")
    print(f"  Total samples: {len(training_data)}")
    print(f"  Incidents: {len(incidents)} ({len(incidents)/len(training_data)*100:.1f}%)")
    print(f"  Normal: {len(normals)} ({len(normals)/len(training_data)*100:.1f}%)")
    print(f"  Edge cases: {len(edge_cases)} ({len(edge_cases)/len(training_data)*100:.1f}%)")

    # Binary label distribution
    binary_labels = {}
    for incident in training_data:
        label = incident.ground_truth.get('actual_label', 'unknown')
        binary_labels[label] = binary_labels.get(label, 0) + 1

    print(f"\nBinary Label Distribution:")
    for label, count in sorted(binary_labels.items()):
        print(f"  {label}: {count} ({count/len(training_data)*100:.1f}%)")

    print(f"\n{'=' * 60}")
    print("[SUCCESS] Generation complete!")
    print("[SUCCESS] Ready for binary classification training")
    print("=" * 60)


if __name__ == "__main__":
    main()
