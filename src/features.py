"""Feature engineering module for incident data."""

import numpy as np
import datetime
import os
import sys
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our synthetic data structures
from src.synthetic_data import SyntheticIncident, IncidentMetrics, IncidentContext


class IncidentFeatureExtractor:
    """
    Feature extractor for incident data supporting both ML models and AI agents.

    Provides fast feature extraction for LightGBM models and rich context
    extraction for agent reasoning.
    """

    def __init__(self):
        """Initialize the feature extractor."""
        # Define feature names for consistency
        self.model_feature_names = [
            'cpu_usage',
            'memory_usage',
            'response_time_ms',
            'error_rate',
            'request_rate',
            'hour_of_day',
            'day_of_week',
            'is_business_hours',
            'cpu_delta',
            'response_time_delta',
            'error_rate_delta',
            'db_unhealthy',
            'cache_unhealthy',
            'queue_high',
            'pattern_similarity_score'
        ]

        # Business hours thresholds
        self.business_hours_start = 8  # 8 AM
        self.business_hours_end = 18   # 6 PM
        self.business_days = [0, 1, 2, 3, 4]  # Monday-Friday (0=Monday)

        # Health thresholds for binary features
        self.db_unhealthy_threshold = 0.8  # 80% connection pool usage
        self.cache_unhealthy_threshold = 0.15  # 15% error rate
        self.queue_high_threshold = 1000.0  # 1000 ms response time

    def extract_model_features(self, incident: SyntheticIncident) -> np.ndarray:
        """
        Fast feature extraction for LightGBM model.

        Target: <5ms extraction time.

        Args:
            incident: SyntheticIncident object

        Returns:
            np.ndarray: Feature vector for ML model
        """
        metrics = incident.metrics
        context = incident.context

        # Parse timestamp for temporal features
        timestamp = datetime.datetime.fromisoformat(context.timestamp.replace('Z', '+00:00'))

        # Base metrics features
        cpu_usage = metrics.cpu_usage
        memory_usage = metrics.memory_usage
        response_time_ms = metrics.response_time_ms
        error_rate = metrics.error_rate
        request_rate = metrics.throughput_rps  # Using throughput as request rate

        # Temporal features
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        is_business_hours = float(self._is_business_hours(timestamp))

        # Delta features (rate of change approximation)
        # Using simplified delta calculation based on current metrics vs expected baseline
        cpu_delta = self._calculate_delta(metrics, 'cpu_usage')
        response_time_delta = self._calculate_delta(metrics, 'response_time_ms')
        error_rate_delta = self._calculate_delta(metrics, 'error_rate')

        # Health indicator features
        db_unhealthy = float(metrics.connection_pool_usage > self.db_unhealthy_threshold * 100)
        cache_unhealthy = float(metrics.error_rate > self.cache_unhealthy_threshold)
        queue_high = float(metrics.response_time_ms > self.queue_high_threshold)

        # Pattern similarity score
        pattern_similarity_score = self._pattern_similarity_score(metrics)

        # Construct feature vector
        features = np.array([
            cpu_usage,
            memory_usage,
            response_time_ms,
            error_rate,
            request_rate,
            hour_of_day,
            day_of_week,
            is_business_hours,
            cpu_delta,
            response_time_delta,
            error_rate_delta,
            db_unhealthy,
            cache_unhealthy,
            queue_high,
            pattern_similarity_score
        ], dtype=np.float32)

        return features

    def extract_agent_context(self, incident: SyntheticIncident) -> Dict[str, Any]:
        """
        Rich context extraction for agent reasoning.

        Can be slower but provides comprehensive context for decision making.

        Args:
            incident: SyntheticIncident object

        Returns:
            Dict: Rich context for agent reasoning
        """
        # Generate human-readable incident summary
        incident_summary = self._summarize_incident(incident)

        # Extract timestamp and parse for readability
        timestamp = datetime.datetime.fromisoformat(incident.context.timestamp.replace('Z', '+00:00'))

        # Determine service name from deployments or generate default
        service_name = "unknown_service"
        if incident.context.recent_deployments:
            # Extract service name from deployment string
            deployment = incident.context.recent_deployments[0]
            service_name = deployment.split('-')[0] if '-' in deployment else deployment

        # Determine severity based on metrics and predictions
        severity = self._determine_severity(incident)

        # Prepare metrics in a structured format
        metrics_dict = asdict(incident.metrics)

        # Add computed features for context
        computed_features = {
            'is_business_hours': self._is_business_hours(timestamp),
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.strftime('%A'),
            'cpu_delta': self._calculate_delta(incident.metrics, 'cpu_usage'),
            'response_time_delta': self._calculate_delta(incident.metrics, 'response_time_ms'),
            'error_rate_delta': self._calculate_delta(incident.metrics, 'error_rate'),
            'db_health_status': 'unhealthy' if incident.metrics.connection_pool_usage > 80 else 'healthy',
            'cache_health_status': 'unhealthy' if incident.metrics.error_rate > 0.15 else 'healthy',
            'queue_status': 'high' if incident.metrics.response_time_ms > 1000 else 'normal',
            'pattern_similarity_score': self._pattern_similarity_score(incident.metrics)
        }

        # Historical context
        historical_context = []
        for hist_incident in incident.context.historical_incidents:
            historical_context.append({
                'summary': self._format_historical_incident(hist_incident),
                'relevance_score': self._calculate_historical_relevance(incident, hist_incident)
            })

        # Business context
        business_context = {
            'event': incident.context.business_event,
            'traffic_multiplier': incident.context.traffic_multiplier,
            'geographic_distribution': incident.context.geographic_distribution,
            'feature_flags': incident.context.feature_flags,
            'recent_deployments': incident.context.recent_deployments
        }

        return {
            'incident_id': incident.incident_id,
            'incident_summary': incident_summary,
            'timestamp': timestamp.isoformat(),
            'service_name': service_name,
            'severity': severity,
            'metrics': metrics_dict,
            'computed_features': computed_features,
            'context': {
                'business': business_context,
                'historical': historical_context,
                'predictions': asdict(incident.predictions) if hasattr(incident, 'predictions') else {},
                'ground_truth': incident.ground_truth if hasattr(incident, 'ground_truth') else {}
            }
        }

    def _is_business_hours(self, timestamp: datetime.datetime) -> bool:
        """
        Check if timestamp falls within business hours.

        Business hours: Monday-Friday 8AM-6PM

        Args:
            timestamp: datetime object

        Returns:
            bool: True if within business hours
        """
        # Check if it's a weekday (Monday=0, Sunday=6)
        if timestamp.weekday() not in self.business_days:
            return False

        # Check if it's within business hours
        hour = timestamp.hour
        return self.business_hours_start <= hour < self.business_hours_end

    def _calculate_delta(self, metrics: IncidentMetrics, key: str) -> float:
        """
        Calculate delta (rate of change) for a metric.

        Since we don't have historical data, we estimate delta based on
        deviation from expected baseline values.

        Args:
            metrics: IncidentMetrics object
            key: Metric key name

        Returns:
            float: Estimated delta value
        """
        # Define baseline "normal" values for comparison
        baselines = {
            'cpu_usage': 25.0,           # Normal CPU around 25%
            'memory_usage': 50.0,        # Normal memory around 50%
            'response_time_ms': 150.0,   # Normal response time 150ms
            'error_rate': 0.03,          # Normal error rate 3%
            'network_latency_ms': 10.0,  # Normal latency 10ms
            'throughput_rps': 1000.0     # Normal throughput 1000 RPS
        }

        current_value = getattr(metrics, key, 0.0)
        baseline_value = baselines.get(key, current_value)

        # Calculate percentage change from baseline
        if baseline_value > 0:
            delta = (current_value - baseline_value) / baseline_value
        else:
            delta = 0.0

        return float(delta)

    def _pattern_similarity_score(self, metrics: IncidentMetrics) -> float:
        """
        Calculate pattern similarity score based on metric patterns.

        This score represents how similar the current metrics pattern is
        to known incident patterns.

        Args:
            metrics: IncidentMetrics object

        Returns:
            float: Similarity score between 0-1
        """
        # Define characteristic patterns for different incident types
        patterns = {
            'database_performance': {
                'cpu_usage': 80, 'memory_usage': 75, 'response_time_ms': 2000,
                'connection_pool_usage': 90, 'db_query_time_ms': 800
            },
            'network_latency': {
                'network_latency_ms': 200, 'packet_loss_percent': 2.0,
                'response_time_ms': 1500, 'throughput_rps': 400
            },
            'memory_leak': {
                'memory_usage': 90, 'cpu_usage': 70, 'response_time_ms': 800
            },
            'cpu_spike': {
                'cpu_usage': 95, 'response_time_ms': 3000, 'throughput_rps': 200
            },
            'disk_io': {
                'disk_io_ops': 8000, 'response_time_ms': 2000, 'cpu_usage': 50
            }
        }

        current_metrics = asdict(metrics)
        max_similarity = 0.0

        # Calculate similarity to each pattern
        for pattern_name, pattern_values in patterns.items():
            similarities = []

            for metric_name, expected_value in pattern_values.items():
                if metric_name in current_metrics:
                    current_value = current_metrics[metric_name]

                    # Calculate normalized similarity (closer to expected = higher similarity)
                    if expected_value > 0:
                        # Use inverse of relative difference
                        diff = abs(current_value - expected_value) / expected_value
                        similarity = max(0.0, 1.0 - diff)
                    else:
                        similarity = 1.0 if current_value == 0 else 0.0

                    similarities.append(similarity)

            # Average similarity for this pattern
            if similarities:
                pattern_similarity = np.mean(similarities)
                max_similarity = max(max_similarity, pattern_similarity)

        return float(max_similarity)

    def _summarize_incident(self, incident: SyntheticIncident) -> str:
        """
        Generate human-readable incident summary.

        Args:
            incident: SyntheticIncident object

        Returns:
            str: Human-readable incident summary
        """
        metrics = incident.metrics
        context = incident.context

        # Identify primary symptoms
        symptoms = []

        if metrics.cpu_usage > 80:
            symptoms.append(f"high CPU usage ({metrics.cpu_usage:.1f}%)")

        if metrics.memory_usage > 80:
            symptoms.append(f"high memory usage ({metrics.memory_usage:.1f}%)")

        if metrics.response_time_ms > 1000:
            symptoms.append(f"slow response times ({metrics.response_time_ms:.0f}ms)")

        if metrics.error_rate > 0.1:
            symptoms.append(f"elevated error rate ({metrics.error_rate:.1%})")

        if metrics.network_latency_ms > 50:
            symptoms.append(f"network latency ({metrics.network_latency_ms:.1f}ms)")

        if metrics.connection_pool_usage > 80:
            symptoms.append(f"database connection pool strain ({metrics.connection_pool_usage:.1f}%)")

        # Format timestamp
        timestamp = datetime.datetime.fromisoformat(context.timestamp.replace('Z', '+00:00'))

        # Build summary
        symptoms_text = ", ".join(symptoms[:3]) if symptoms else "no clear symptoms"

        summary = f"Incident detected at {timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
        summary += f"during {context.business_event} with {symptoms_text}. "

        if context.recent_deployments:
            summary += f"Recent deployment: {context.recent_deployments[0]}. "

        if context.traffic_multiplier > 1.5:
            summary += f"Traffic is {context.traffic_multiplier:.1f}x normal levels. "

        if len(context.historical_incidents) > 0:
            summary += f"Found {len(context.historical_incidents)} similar historical incidents."

        return summary

    def _determine_severity(self, incident: SyntheticIncident) -> str:
        """
        Determine incident severity based on metrics and context.

        Args:
            incident: SyntheticIncident object

        Returns:
            str: Severity level (low, medium, high, critical)
        """
        metrics = incident.metrics
        severity_score = 0

        # Score based on various factors
        if metrics.cpu_usage > 90:
            severity_score += 3
        elif metrics.cpu_usage > 70:
            severity_score += 2
        elif metrics.cpu_usage > 50:
            severity_score += 1

        if metrics.memory_usage > 90:
            severity_score += 3
        elif metrics.memory_usage > 80:
            severity_score += 2

        if metrics.response_time_ms > 3000:
            severity_score += 3
        elif metrics.response_time_ms > 1000:
            severity_score += 2
        elif metrics.response_time_ms > 500:
            severity_score += 1

        if metrics.error_rate > 0.2:
            severity_score += 3
        elif metrics.error_rate > 0.1:
            severity_score += 2
        elif metrics.error_rate > 0.05:
            severity_score += 1

        # Business context can elevate severity
        if incident.context.business_event == "black_friday_peak_shopping":
            severity_score += 1

        # Map score to severity levels
        if severity_score >= 8:
            return "critical"
        elif severity_score >= 5:
            return "high"
        elif severity_score >= 2:
            return "medium"
        else:
            return "low"

    def _format_historical_incident(self, hist_incident: Dict[str, Any]) -> str:
        """
        Format historical incident for human consumption.

        Args:
            hist_incident: Historical incident dictionary

        Returns:
            str: Formatted description
        """
        if 'incident_id' in hist_incident:
            summary = f"Incident {hist_incident['incident_id']}"
            if 'date' in hist_incident:
                summary += f" on {hist_incident['date']}"
            if 'root_cause' in hist_incident:
                summary += f": {hist_incident['root_cause']}"
            if 'resolution' in hist_incident:
                summary += f" (resolved by {hist_incident['resolution']})"
        else:
            # Handle other historical formats (like Black Friday data)
            summary = f"Historical event: {hist_incident.get('event', 'unknown')}"
            if 'traffic_multiplier' in hist_incident:
                summary += f" with {hist_incident['traffic_multiplier']:.1f}x traffic"

        return summary

    def _calculate_historical_relevance(self, current_incident: SyntheticIncident,
                                      hist_incident: Dict[str, Any]) -> float:
        """
        Calculate relevance score between current and historical incident.

        Args:
            current_incident: Current SyntheticIncident
            hist_incident: Historical incident dictionary

        Returns:
            float: Relevance score 0-1
        """
        relevance_score = 0.0

        # Check for similar symptoms
        if 'symptoms' in hist_incident:
            current_symptoms = set()
            metrics = current_incident.metrics

            if metrics.connection_pool_usage > 80:
                current_symptoms.add('high_connection_pool')
            if metrics.response_time_ms > 1000:
                current_symptoms.add('elevated_response_time')
            if metrics.packet_loss_percent > 1.0:
                current_symptoms.add('packet_loss')
            if metrics.network_latency_ms > 100:
                current_symptoms.add('network_latency')

            hist_symptoms = set(hist_incident['symptoms'])
            if current_symptoms and hist_symptoms:
                overlap = len(current_symptoms.intersection(hist_symptoms))
                total = len(current_symptoms.union(hist_symptoms))
                relevance_score += (overlap / total) * 0.6

        # Check for similar business context
        if 'event' in hist_incident:
            if hist_incident['event'] == current_incident.context.business_event:
                relevance_score += 0.3

        # Check for similar traffic patterns
        if 'traffic_multiplier' in hist_incident:
            current_traffic = current_incident.context.traffic_multiplier
            hist_traffic = hist_incident['traffic_multiplier']
            traffic_similarity = 1.0 - min(1.0, abs(current_traffic - hist_traffic) / max(current_traffic, hist_traffic))
            relevance_score += traffic_similarity * 0.1

        return min(1.0, relevance_score)

    def get_feature_names(self) -> List[str]:
        """
        Get the names of model features.

        Returns:
            List[str]: Feature names in order
        """
        return self.model_feature_names.copy()

    def extract_batch_features(self, incidents: List[SyntheticIncident]) -> np.ndarray:
        """
        Extract features for a batch of incidents efficiently.

        Args:
            incidents: List of SyntheticIncident objects

        Returns:
            np.ndarray: Feature matrix (n_incidents, n_features)
        """
        features_list = []
        for incident in incidents:
            features = self.extract_model_features(incident)
            features_list.append(features)

        return np.vstack(features_list)


def main():
    """Test the feature extractor."""
    print("Incident Feature Extractor Test")
    print("=" * 40)

    # Import and create test data
    try:
        from .synthetic_data import SyntheticIncidentGenerator
    except ImportError:
        from synthetic_data import SyntheticIncidentGenerator

    generator = SyntheticIncidentGenerator(seed=42)
    extractor = IncidentFeatureExtractor()

    # Generate test incidents
    print("\nGenerating test incidents...")
    incidents = generator.generate_training_dataset(n_samples=10)

    # Test model feature extraction
    print("\nTesting model feature extraction...")
    features = extractor.extract_model_features(incidents[0])
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature names: {extractor.get_feature_names()}")
    print(f"Feature values: {features}")

    # Test batch extraction
    print("\nTesting batch feature extraction...")
    batch_features = extractor.extract_batch_features(incidents[:5])
    print(f"Batch features shape: {batch_features.shape}")

    # Test agent context extraction
    print("\nTesting agent context extraction...")
    context = extractor.extract_agent_context(incidents[0])
    print(f"Context keys: {list(context.keys())}")
    print(f"Incident summary: {context['incident_summary']}")
    print(f"Severity: {context['severity']}")

    print("\nFeature extraction test complete!")


if __name__ == "__main__":
    main()