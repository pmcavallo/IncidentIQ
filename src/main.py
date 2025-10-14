"""FastAPI application for IncidentIQ - AI-powered incident response system."""

import os
import sys
import uuid
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Import project modules using consistent format
from src.model import IncidentClassifier
from src.features import IncidentFeatureExtractor
from src.agents import IncidentAgentSystem
from src.synthetic_data import SyntheticIncident, IncidentMetrics, IncidentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global system components
classifier: Optional[IncidentClassifier] = None
feature_extractor: Optional[IncidentFeatureExtractor] = None
agent_system: Optional[IncidentAgentSystem] = None

# Investigation tracking
investigation_results: Dict[str, Dict] = {}


class IncidentAlert(BaseModel):
    """Incoming incident alert data."""
    service_name: str = Field(..., description="Name of the affected service")
    severity: str = Field(..., description="Alert severity level", pattern="^(low|medium|high|critical)$")
    metrics: Dict[str, float] = Field(..., description="System metrics at time of alert")
    alert_source: str = Field(..., description="Source system that generated the alert")
    description: Optional[str] = Field(None, description="Human-readable alert description")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Alert timestamp")


class IncidentResponse(BaseModel):
    """Response from incident processing."""
    incident_id: str = Field(..., description="Unique incident identifier")
    decision: str = Field(..., description="System decision on how to handle incident")
    classification: str = Field(..., description="Predicted incident type")
    confidence: float = Field(..., description="Confidence in classification (0.0-1.0)")
    action: str = Field(..., description="Action being taken")
    status: str = Field(..., description="Current processing status")
    reasoning: Optional[List[str]] = Field(None, description="Decision reasoning chain")
    recommended_actions: Optional[List[str]] = Field(None, description="Specific actions to take")
    estimated_resolution_time: Optional[str] = Field(None, description="Estimated time to resolve")
    requires_human_review: Optional[bool] = Field(None, description="Whether human review is required")


class ServiceInfo(BaseModel):
    """Service information and status."""
    name: str = "IncidentIQ"
    version: str = "0.1.0"
    description: str = "AI-powered incident response and analysis system"
    status: str = "operational"
    model_loaded: bool = False
    agents_available: bool = False
    features_count: int = 15


class HealthStatus(BaseModel):
    """Health check status."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool = False
    agents_available: bool = False
    components: Dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Starting IncidentIQ system...")
    await initialize_system()
    yield
    # Shutdown
    logger.info("Shutting down IncidentIQ system...")


app = FastAPI(
    title="IncidentIQ",
    description="AI-powered incident response and analysis system",
    version="0.1.0",
    lifespan=lifespan
)


async def initialize_system():
    """Initialize ML models and agent system."""
    global classifier, feature_extractor, agent_system

    logger.info("Initializing system components...")

    try:
        # Initialize feature extractor
        feature_extractor = IncidentFeatureExtractor()
        logger.info("✅ Feature extractor initialized")

        # Initialize and load classifier
        classifier = IncidentClassifier()
        try:
            classifier.load('./models/incident_classifier')
            logger.info("✅ LightGBM model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load trained model: {e}")
            logger.info("System will use fallback predictions")

        # Initialize agent system
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            logger.info("🤖 Initializing agent system with Anthropic API...")
        else:
            logger.info("🤖 Initializing agent system in mock mode (no API key)")

        agent_system = IncidentAgentSystem(anthropic_api_key=anthropic_key)
        logger.info("✅ Agent system initialized")

        logger.info("🚀 IncidentIQ system ready!")

    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise


def convert_alert_to_incident(alert: IncidentAlert, incident_id: str) -> SyntheticIncident:
    """Convert incoming alert to SyntheticIncident format."""

    # Map alert metrics to incident metrics format
    metrics = IncidentMetrics(
        cpu_usage=alert.metrics.get('cpu_usage', 50.0),
        memory_usage=alert.metrics.get('memory_usage', 60.0),
        disk_io_ops=alert.metrics.get('disk_io_ops', 200.0),
        network_latency_ms=alert.metrics.get('network_latency_ms', 100.0),
        response_time_ms=alert.metrics.get('response_time_ms', 500.0),
        error_rate=alert.metrics.get('error_rate', 0.01),
        connection_pool_usage=alert.metrics.get('connection_pool_usage', 70.0),
        throughput_rps=alert.metrics.get('throughput_rps', 1000.0),
        packet_loss_percent=alert.metrics.get('packet_loss_percent', 0.1),
        db_query_time_ms=alert.metrics.get('db_query_time_ms', 50.0)
    )

    # Create context
    context = IncidentContext(
        timestamp=alert.timestamp.isoformat() if alert.timestamp else datetime.utcnow().isoformat(),
        business_event="normal_operations",
        recent_deployments=[],
        traffic_multiplier=1.0,
        geographic_distribution={"us-east": 0.5, "us-west": 0.3, "eu": 0.2},
        feature_flags=[],
        historical_incidents=[]
    )

    # Create synthetic incident
    incident = SyntheticIncident(
        incident_id=incident_id,
        metrics=metrics,
        context=context,
        predictions={
            "incident_type": "unknown",
            "severity": alert.severity,
            "root_cause": "unknown",
            "confidence": 0.5,
            "recommended_actions": [],
            "is_edge_case": True,
            "edge_case_type": "live_alert"
        },
        ground_truth={
            "actual_root_cause": "unknown",
            "resolution_time_minutes": 0,
            "business_impact": "unknown"
        }
    )

    return incident


async def run_agent_investigation(incident_id: str, alert: IncidentAlert):
    """Background task for agent investigation of edge cases."""
    global investigation_results

    logger.info(f"🔍 Starting agent investigation for incident {incident_id}")

    try:
        # Convert alert to incident format
        incident = convert_alert_to_incident(alert, incident_id)

        # Run agent analysis
        result = await agent_system.analyze_incident(incident)

        # Store results immediately
        investigation_results[incident_id] = {
            "status": "completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "alert": alert.dict()
        }

        # Brief delay to ensure results are persisted
        await asyncio.sleep(0.1)

        logger.info(f"✅ Agent investigation completed for {incident_id}")
        logger.info(f"Root cause: {result['root_cause']}, Confidence: {result['confidence']:.2f}")

        # Log actions for monitoring
        if result.get('recommended_actions'):
            logger.info(f"Recommended actions for {incident_id}:")
            for i, action in enumerate(result['recommended_actions'], 1):
                logger.info(f"  {i}. {action}")

        # Check for human review requirement
        if result.get('requires_human_review'):
            logger.warning(f"⚠️ Incident {incident_id} requires human review")
            violations = result.get('governance_violations', [])
            for violation in violations:
                logger.warning(f"  - {violation}")

    except Exception as e:
        logger.error(f"❌ Agent investigation failed for {incident_id}: {e}")

        # Store failure result with fallback analysis
        investigation_results[incident_id] = {
            "status": "completed",  # Mark as completed to prevent infinite waiting
            "result": {
                "root_cause": "investigation_failed",
                "confidence": 0.5,
                "recommended_actions": [
                    "Manual investigation required due to system error",
                    "Check system logs for investigation failure details",
                    "Escalate to on-call engineer"
                ],
                "reasoning_chain": [
                    f"Agent investigation failed: {str(e)}",
                    "Fallback to manual investigation workflow"
                ],
                "requires_human_review": True,
                "governance_violations": ["SYSTEM_ERROR: Agent investigation failed"]
            },
            "timestamp": datetime.utcnow().isoformat(),
            "alert": alert.dict(),
            "error_details": str(e)
        }


@app.get("/", response_model=ServiceInfo)
async def get_service_info():
    """Get service information and status."""
    return ServiceInfo(
        model_loaded=classifier is not None and classifier.is_trained,
        agents_available=agent_system is not None,
        features_count=15
    )


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    components = {}

    # Check classifier
    if classifier is not None:
        components["classifier"] = "loaded" if classifier.is_trained else "not_trained"
    else:
        components["classifier"] = "not_initialized"

    # Check feature extractor
    if feature_extractor is not None:
        components["feature_extractor"] = "ready"
    else:
        components["feature_extractor"] = "not_initialized"

    # Check agent system
    if agent_system is not None:
        components["agent_system"] = "ready"
    else:
        components["agent_system"] = "not_initialized"

    # Determine overall health
    all_ready = (
        classifier is not None and
        feature_extractor is not None and
        agent_system is not None
    )

    return HealthStatus(
        status="healthy" if all_ready else "degraded",
        model_loaded=classifier is not None and classifier.is_trained,
        agents_available=agent_system is not None,
        components=components
    )


@app.post("/incident/alert", response_model=IncidentResponse)
async def handle_incident_alert(alert: IncidentAlert, background_tasks: BackgroundTasks):
    """Main incident handling endpoint."""

    # Generate unique incident ID
    incident_id = f"INC_{int(time.time())}_{str(uuid.uuid4())[:8]}"

    logger.info(f"📥 Received alert for {alert.service_name} (severity: {alert.severity})")
    logger.info(f"Assigned incident ID: {incident_id}")

    # Check system availability
    if not classifier or not feature_extractor or not agent_system:
        raise HTTPException(
            status_code=503,
            detail="System not fully initialized. Check /health endpoint."
        )

    try:
        # Convert alert to incident format for processing
        incident = convert_alert_to_incident(alert, incident_id)

        # Extract features and classify
        if classifier.is_trained:
            features = feature_extractor.extract_model_features(incident)
            predicted_class, confidence, is_edge_case = classifier.predict(features)
        else:
            # Fallback when model not trained
            predicted_class = "unknown"
            confidence = 0.5
            is_edge_case = True

        logger.info(f"🤖 Model prediction: {predicted_class} (confidence: {confidence:.2f}, edge_case: {is_edge_case})")

        # Decision logic based on confidence threshold
        if confidence >= 0.75 and not is_edge_case:
            # High confidence - return standard remediation immediately
            standard_actions = [
                f"Scale {alert.service_name} horizontally",
                "Monitor key metrics for 15 minutes",
                "Check recent deployments for rollback candidates",
                "Enable enhanced logging",
                "Alert on-call team"
            ]

            logger.info(f"✅ High confidence prediction - returning standard remediation")

            return IncidentResponse(
                incident_id=incident_id,
                decision="automated_remediation",
                classification=predicted_class,
                confidence=confidence,
                action="standard_remediation",
                status="resolved",
                reasoning=[
                    f"Model classified as {predicted_class} with {confidence:.2f} confidence",
                    "Confidence above 0.75 threshold - applying standard remediation",
                    "No agent investigation required"
                ],
                recommended_actions=standard_actions,
                estimated_resolution_time="15-30 minutes",
                requires_human_review=False
            )

        else:
            # Low confidence or edge case - queue agent investigation
            logger.info(f"🔍 Low confidence or edge case - queuing agent investigation")

            # Add background task for agent investigation
            background_tasks.add_task(run_agent_investigation, incident_id, alert)

            # Return immediate response
            return IncidentResponse(
                incident_id=incident_id,
                decision="agent_investigation",
                classification=predicted_class,
                confidence=confidence,
                action="investigating",
                status="investigating",
                reasoning=[
                    f"Model classified as {predicted_class} with {confidence:.2f} confidence",
                    "Confidence below 0.75 threshold or edge case detected",
                    "Queued for multi-agent investigation"
                ],
                recommended_actions=[
                    "Monitor situation closely",
                    "Prepare for potential escalation",
                    "Check investigation status periodically"
                ],
                estimated_resolution_time="pending_investigation",
                requires_human_review=None  # Will be determined by agents
            )

    except Exception as e:
        logger.error(f"❌ Error processing alert: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process incident alert: {str(e)}"
        )


@app.get("/incident/{incident_id}/status")
async def get_incident_status(incident_id: str):
    """Get status of ongoing incident investigation."""

    if incident_id not in investigation_results:
        raise HTTPException(
            status_code=404,
            detail=f"Incident {incident_id} not found or investigation not started"
        )

    result = investigation_results[incident_id]

    if result["status"] == "completed":
        investigation = result["result"]
        return {
            "incident_id": incident_id,
            "status": "completed",
            "root_cause": investigation["root_cause"],
            "confidence": investigation["confidence"],
            "requires_human_review": investigation.get("requires_human_review", False),
            "recommended_actions": investigation.get("recommended_actions", []),
            "reasoning_chain": investigation.get("reasoning_chain", []),
            "governance_violations": investigation.get("governance_violations", []),
            "completed_at": result["timestamp"]
        }
    elif result["status"] == "failed":
        return {
            "incident_id": incident_id,
            "status": "failed",
            "error": result["error"],
            "failed_at": result["timestamp"]
        }
    else:
        return {
            "incident_id": incident_id,
            "status": "investigating",
            "message": "Agent investigation in progress"
        }


@app.get("/incidents/active")
async def get_active_incidents():
    """Get list of all active/recent incidents."""
    return {
        "active_investigations": len([r for r in investigation_results.values() if r["status"] == "investigating"]),
        "completed_investigations": len([r for r in investigation_results.values() if r["status"] == "completed"]),
        "failed_investigations": len([r for r in investigation_results.values() if r["status"] == "failed"]),
        "incidents": {
            incident_id: {
                "status": result["status"],
                "timestamp": result["timestamp"],
                "service_name": result["alert"]["service_name"],
                "severity": result["alert"]["severity"]
            }
            for incident_id, result in investigation_results.items()
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )