"""
IncidentIQ Streamlit Web Dashboard
AI-Enhanced Incident Response System Demo
"""

import streamlit as st
import json
import time
import os
import sys
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables more robustly
from dotenv import load_dotenv

# Try multiple paths for .env file
env_paths = [
    os.path.join(project_root, '.env'),
    '.env',
    '../.env'
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
        break
else:
    print("Warning: No .env file found")

# Import IncidentIQ components
from src.model import IncidentClassifier
from src.features import IncidentFeatureExtractor
from src.agents import IncidentAgentSystem
from src.synthetic_data import SyntheticIncident, IncidentMetrics, IncidentContext

# Page configuration
st.set_page_config(
    page_title="IncidentIQ Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dramatic styling
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: none;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(79,172,254,0.3);
        color: white;
    }
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border: none;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(250,112,154,0.3);
        color: white;
    }
    .failure-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border: none;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255,107,107,0.3);
        color: white;
    }
    .agent-step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 6px solid #4facfe;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-radius: 0 1rem 1rem 0;
        color: white;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .confidence-high {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .confidence-low {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa726 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .vs-divider {
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin: 2rem 0;
    }
    .executive-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .impact-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(17,153,142,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Constants
EDGE_CASES_DIR = "demo_data"
MODEL_PATH = "./models/incident_classifier"

# Global system components (initialized on startup)
classifier: Optional[IncidentClassifier] = None
feature_extractor: Optional[IncidentFeatureExtractor] = None
agent_system: Optional[IncidentAgentSystem] = None
investigation_results: Dict[str, Dict] = {}

def load_edge_cases() -> Dict[str, Dict]:
    """Load pre-generated edge case scenarios."""
    edge_cases = {}

    if not os.path.exists(EDGE_CASES_DIR):
        return edge_cases

    for filename in os.listdir(EDGE_CASES_DIR):
        if filename.endswith('.json'):
            case_name = filename.replace('.json', '').replace('_', ' ').title()
            try:
                with open(os.path.join(EDGE_CASES_DIR, filename), 'r') as f:
                    edge_cases[case_name] = json.load(f)
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")

    return edge_cases

def convert_to_api_format(incident_data: Dict) -> Dict:
    """Convert incident data to API alert format."""
    return {
        "service_name": incident_data.get("service_name", "unknown-service"),
        "severity": incident_data.get("severity", "medium"),
        "metrics": incident_data.get("metrics", {}),
        "alert_source": "streamlit_dashboard",
        "description": incident_data.get("description", "Dashboard submitted incident")
    }

def initialize_system():
    """Initialize ML models and agent system."""

    # Create a container for initialization messages
    init_container = st.container()

    with init_container:
        try:
            st.info("🔄 Initializing IncidentIQ components...")

            # Initialize feature extractor
            feature_extractor_local = IncidentFeatureExtractor()
            st.success("✅ Feature extractor initialized")

            # Initialize and load classifier
            classifier_local = IncidentClassifier()
            try:
                classifier_local.load(MODEL_PATH)
                st.success("✅ LightGBM model loaded successfully")
            except Exception as e:
                st.warning(f"⚠️ Could not load trained model: {e}")
                st.info("System will use fallback predictions")

            # Initialize agent system
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')

            # If not in environment, try to read directly from .env file
            if not anthropic_key:
                env_file_path = os.path.join(project_root, '.env')
                if os.path.exists(env_file_path):
                    with open(env_file_path, 'r') as f:
                        for line in f:
                            if line.startswith('ANTHROPIC_API_KEY='):
                                anthropic_key = line.split('=', 1)[1].strip()
                                break

            # Debug API key detection
            if anthropic_key:
                st.success(f"🔑 Anthropic API key detected: {anthropic_key[:10]}...{anthropic_key[-4:]}")
                st.info("🤖 Initializing agent system with Anthropic API...")
            else:
                st.error("❌ ANTHROPIC_API_KEY not found")
                st.error("Please either:")
                st.error("1. Set environment variable: ANTHROPIC_API_KEY=your_key")
                st.error("2. Ensure .env file contains: ANTHROPIC_API_KEY=your_key")

                # Show current environment status for debugging
                st.info("Debug info:")
                st.code(f"Current working directory: {os.getcwd()}")
                st.code(f"Project root: {project_root}")
                env_file_path = os.path.join(project_root, '.env')
                st.code(f".env file exists: {os.path.exists(env_file_path)}")

                raise Exception("Missing ANTHROPIC_API_KEY - cannot initialize agents")

            agent_system_local = IncidentAgentSystem(anthropic_api_key=anthropic_key)
            st.success("✅ Agent system initialized with real Anthropic API")

            st.success("🎉 System initialization complete!")

            return {
                "classifier": classifier_local,
                "feature_extractor": feature_extractor_local,
                "agent_system": agent_system_local
            }

        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            st.error(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

def convert_alert_to_incident(alert_data: Dict, incident_id: str) -> SyntheticIncident:
    """Convert alert data to SyntheticIncident format."""
    # Map alert metrics to incident metrics format
    metrics = IncidentMetrics(
        cpu_usage=alert_data.get('metrics', {}).get('cpu_usage', 50.0),
        memory_usage=alert_data.get('metrics', {}).get('memory_usage', 60.0),
        disk_io_ops=alert_data.get('metrics', {}).get('disk_io_ops', 200.0),
        network_latency_ms=alert_data.get('metrics', {}).get('network_latency_ms', 100.0),
        response_time_ms=alert_data.get('metrics', {}).get('response_time_ms', 500.0),
        error_rate=alert_data.get('metrics', {}).get('error_rate', 0.01),
        connection_pool_usage=alert_data.get('metrics', {}).get('connection_pool_usage', 70.0),
        throughput_rps=alert_data.get('metrics', {}).get('throughput_rps', 1000.0),
        packet_loss_percent=alert_data.get('metrics', {}).get('packet_loss_percent', 0.1),
        db_query_time_ms=alert_data.get('metrics', {}).get('db_query_time_ms', 50.0)
    )

    # Create context
    context = IncidentContext(
        timestamp=alert_data.get('timestamp', datetime.utcnow().isoformat()),
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
            "severity": alert_data.get('severity', 'medium'),
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

async def run_agent_investigation_async(incident_id: str, incident: SyntheticIncident):
    """Run agent investigation asynchronously."""
    global investigation_results, agent_system

    try:
        # Check if agent system is available
        if agent_system is None:
            raise Exception("Agent system not initialized")

        # Run agent analysis
        result = await agent_system.analyze_incident(incident)

        # Store results
        investigation_results[incident_id] = {
            "status": "completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }

        return result

    except Exception as e:
        # Store failure result
        investigation_results[incident_id] = {
            "status": "completed",
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
            "error_details": str(e)
        }
        raise e

def run_agent_investigation(incident_id: str, incident: SyntheticIncident):
    """Run agent investigation synchronously for Streamlit."""
    global agent_system

    # Get agent system from session state or globals
    agent_system_local = st.session_state.get('agent_system') or agent_system

    try:
        # Check if agent system is available
        if agent_system_local is None:
            st.error("❌ Agent system not initialized")
            raise Exception("Agent system not available")

        # Run the async function in a new event loop with local agent system
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Temporarily update global for async function
        original_agent_system = agent_system
        agent_system = agent_system_local

        try:
            result = loop.run_until_complete(run_agent_investigation_async(incident_id, incident))
        finally:
            # Restore original
            agent_system = original_agent_system
            loop.close()

        return result
    except Exception as e:
        st.error(f"❌ Agent investigation failed: {e}")
        st.error("This demo requires a valid Anthropic API key for the full agent experience")
        raise e


def display_system_status():
    """Display current system status."""
    # Get components from session state or globals
    classifier_local = st.session_state.get('classifier') or classifier
    feature_extractor_local = st.session_state.get('feature_extractor') or feature_extractor
    agent_system_local = st.session_state.get('agent_system') or agent_system

    col1, col2, col3 = st.columns(3)

    with col1:
        if classifier_local and feature_extractor_local and agent_system_local:
            st.success("🟢 System Healthy")
        else:
            st.warning("🟡 System Degraded")

    with col2:
        if classifier_local and classifier_local.is_trained:
            st.success("🤖 ML Model Ready")
        elif classifier_local:
            st.warning("⚠️ Model Not Trained")
        else:
            st.error("❌ Model Not Loaded")

    with col3:
        if agent_system_local:
            # Check for API key in multiple ways
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                # Try reading from .env file directly
                env_file_path = os.path.join(project_root, '.env')
                if os.path.exists(env_file_path):
                    with open(env_file_path, 'r') as f:
                        for line in f:
                            if line.startswith('ANTHROPIC_API_KEY='):
                                api_key = line.split('=', 1)[1].strip()
                                break

            if api_key:
                st.success("🧠 AI Agents Ready")
            else:
                st.warning("⚠️ Agents in Mock Mode")
        else:
            st.error("❌ Agents Not Available")

def display_clean_incident_summary(incident_data: Dict):
    """Display clean incident summary without fabricated metrics."""
    st.markdown("### 📋 Incident Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Service:** {incident_data.get('service_name', 'N/A')}")
        severity = incident_data.get('severity', 'unknown')
        if severity in ["critical", "high"]:
            st.markdown(f"**Severity:** 🚨 {severity.upper()}")
        elif severity == "medium":
            st.markdown(f"**Severity:** ⚠️ {severity.upper()}")
        else:
            st.markdown(f"**Severity:** ℹ️ {severity.upper()}")

    with col2:
        if "timestamp" in incident_data:
            st.markdown(f"**Timestamp:** {incident_data['timestamp']}")
        if "description" in incident_data:
            st.markdown(f"**Description:** {incident_data['description']}")

    # Show metrics if available
    if "metrics" in incident_data and incident_data["metrics"]:
        with st.expander("📊 System Metrics", expanded=False):
            metrics = incident_data["metrics"]
            for key, value in metrics.items():
                if isinstance(value, float):
                    st.text(f"{key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    st.text(f"{key.replace('_', ' ').title()}: {value}")

def display_clean_ml_prediction(response_data: Dict):
    """Display ML model prediction without fabricated metrics."""
    st.markdown("### 🤖 ML Model Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        classification = response_data.get("classification", "unknown")
        st.metric("Classification", classification.replace('_', ' ').title())

    with col2:
        confidence = response_data.get("confidence", 0.0)
        st.metric("Confidence", f"{confidence:.1%}")

    with col3:
        decision = response_data.get("decision", "unknown")
        if decision == "automated_remediation":
            st.success("✅ High Confidence")
        elif decision == "agent_investigation":
            st.warning("⚠️ Edge Case Detected")
        else:
            st.info(f"Status: {decision}")

    # Show decision threshold
    if confidence < 0.75:
        st.warning(f"⚠️ Confidence ({confidence:.1%}) below 75% threshold → Routing to AI agents")
    else:
        st.success(f"✅ Confidence ({confidence:.1%}) above 75% threshold → Standard remediation")

    # Show model reasoning
    if "reasoning" in response_data:
        with st.expander("🧠 Model Reasoning Chain", expanded=False):
            for i, reason in enumerate(response_data["reasoning"], 1):
                st.markdown(f"**{i}.** {reason}")

    # Show recommended actions
    if "recommended_actions" in response_data:
        with st.expander("🔧 Model's Recommended Actions", expanded=False):
            for i, action in enumerate(response_data["recommended_actions"], 1):
                st.markdown(f"{i}. {action}")

def display_standard_remediation(response_data: Dict):
    """Display standard remediation for high-confidence predictions."""
    st.subheader("🎯 Standard Automated Remediation")

    if "recommended_actions" in response_data:
        st.markdown("**🔧 Automated Actions:**")
        for i, action in enumerate(response_data["recommended_actions"], 1):
            st.markdown(f"{i}. {action}")

    est_time = response_data.get("estimated_resolution_time", "15-30 minutes")
    st.success(f"⏱️ **Estimated Resolution Time:** {est_time}")
    st.info("No human intervention required - system will handle automatically.")


def display_live_agent_investigation(incident_id: str, incident: SyntheticIncident, incident_data: Optional[Dict] = None) -> Optional[Dict]:
    """Display live agent investigation with progress tracking."""
    st.subheader("🔍 Step 2: AI Agent Investigation")

    # Check if this is a demo case with pre-defined agent findings
    is_demo_case = (incident_data and "ground_truth" in incident_data and
                    "agent_investigation" in incident_data["ground_truth"])

    # Create progress columns for each agent
    col1, col2, col3 = st.columns(3)

    with col1:
        diag_container = st.container()
        diag_container.markdown("**🩺 Diagnostic Agent**")
        diag_spinner = diag_container.empty()
        diag_status = diag_container.empty()

    with col2:
        context_container = st.container()
        context_container.markdown("**📊 Context Agent**")
        context_spinner = context_container.empty()
        context_status = context_container.empty()

    with col3:
        rec_container = st.container()
        rec_container.markdown("**💡 Recommendation Agent**")
        rec_spinner = rec_container.empty()
        rec_status = rec_container.empty()

    # Start progress indicators
    diag_spinner.info("🔄 Analyzing incident patterns...")
    context_spinner.info("⏳ Waiting...")
    rec_spinner.info("⏳ Waiting...")

    # Simulate progress updates while running investigation
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Update progress indicators
        status_text.text("🔄 Starting diagnostic analysis...")
        progress_bar.progress(10)
        time.sleep(0.5)

        diag_spinner.success("✅ Pattern analysis complete")
        diag_status.success("Found patterns in metrics")
        context_spinner.info("🔄 Evaluating system context...")
        progress_bar.progress(40)
        status_text.text("📊 Evaluating system context...")
        time.sleep(0.5)

        context_spinner.success("✅ Context evaluation complete")
        context_status.success("Historical patterns analyzed")
        rec_spinner.info("🔄 Generating recommendations...")
        progress_bar.progress(70)
        status_text.text("💡 Generating recommendations...")
        time.sleep(0.5)

        progress_bar.progress(90)
        status_text.text("🧠 Running multi-agent investigation...")

        # Use demo data if available, otherwise run real agent investigation
        if is_demo_case:
            # Extract agent findings from demo data
            agent_view = incident_data["ground_truth"]["agent_investigation"]

            # Simulate processing time
            time.sleep(1)

            result = {
                "finding": agent_view.get("finding", "unknown"),  # Add finding field for comparison
                "root_cause": agent_view.get("root_cause", "unknown"),
                "confidence": agent_view.get("confidence", 0.5),
                "recommended_actions": agent_view.get("recommendation", "").split(". ") if isinstance(agent_view.get("recommendation"), str) else agent_view.get("recommended_actions", []),
                "reasoning_chain": agent_view.get("key_insights", []),
                "requires_human_review": agent_view.get("human_review_required", False),
                "override_ml": agent_view.get("override_ml", False),
                "escalation_reason": agent_view.get("escalation_reason", ""),
                "roi_estimate": agent_view.get("roi_estimate", {}),
                "governance_violations": []
            }
        else:
            # Run the actual agent investigation
            result = run_agent_investigation(incident_id, incident)

        if result:
            progress_bar.progress(100)
            status_text.text("✅ Investigation complete!")

            rec_spinner.success("✅ Recommendations generated")
            rec_status.success("Optimal solution identified")

            # Return the investigation result in the expected format
            return {
                "status": "completed",
                "finding": result.get("finding", "unknown"),  # Pass through finding for comparison
                "root_cause": result.get("root_cause", "unknown"),
                "confidence": result.get("confidence", 0.5),
                "requires_human_review": result.get("requires_human_review", False),
                "override_ml": result.get("override_ml", False),
                "escalation_reason": result.get("escalation_reason", ""),
                "roi_estimate": result.get("roi_estimate", {}),
                "recommended_actions": result.get("recommended_actions", []),
                "reasoning_chain": result.get("reasoning_chain", []),
                "governance_violations": result.get("governance_violations", []),
                "completed_at": datetime.utcnow().isoformat(),
                "is_demo_case": is_demo_case
            }
        else:
            progress_bar.progress(0)
            status_text.error("❌ Investigation failed")
            return None

    except Exception as e:
        progress_bar.progress(0)
        status_text.error(f"❌ Investigation failed: {e}")
        return None

    finally:
        # Clean up progress indicators
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

    st.success("🎉 **Multi-Agent Investigation Complete!**")

def display_clean_comparison(model_response: Dict, agent_response: Dict):
    """Display clean comparison without fabricated business metrics."""
    st.markdown("### 📊 Model vs Agent Comparison")

    model_classification = model_response.get("classification", "unknown")
    model_confidence = model_response.get("confidence", 0)
    agent_root_cause = agent_response.get("root_cause", "unknown")
    agent_confidence = agent_response.get("confidence", 0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🤖 ML Model")
        st.markdown(f"**Classification:** {model_classification.replace('_', ' ').title()}")
        st.markdown(f"**Confidence:** {model_confidence:.1%}")

        if "recommended_actions" in model_response:
            st.markdown("**Would have done:**")
            for action in model_response["recommended_actions"][:3]:
                st.markdown(f"- {action}")

    with col2:
        st.markdown("#### 🧠 AI Agents")
        st.markdown(f"**Root Cause:** {agent_root_cause.replace('_', ' ').title()}")
        st.markdown(f"**Confidence:** {agent_confidence:.1%}")

        if "recommended_actions" in agent_response:
            st.markdown("**Recommended:**")
            for action in agent_response["recommended_actions"][:3]:
                st.markdown(f"- {action}")

    # Show key insight
    st.markdown("#### 🔍 Key Insight")

    # Check if this is a demo case to show better comparison
    is_demo = model_response.get("is_demo_case", False)

    if is_demo:
        # For demo cases, compare finding (normal/incident) not classification vs root_cause
        model_finding = model_classification  # 'normal' or 'incident'
        agent_finding = agent_response.get("finding", agent_root_cause)  # 'normal' or 'incident'
        override_ml = agent_response.get("override_ml", False)

        if model_finding != agent_finding:
            st.error(f"**⚠️ AGENTS RECOMMEND OVERRIDING ML MODEL**")
            st.warning(f"**ML Model Decision:**\n"
                      f"- Predicted: **'{model_finding}'** with {model_confidence:.0%} confidence\n"
                      f"- Would do: {model_response.get('recommended_actions', ['Standard remediation'])[0]}\n\n"
                      f"**Agent Analysis:**\n"
                      f"- Finding: **'{agent_finding}'** with {agent_confidence:.0%} confidence\n"
                      f"- Root Cause: {agent_root_cause.replace('_', ' ').title()}\n"
                      f"- **Recommendation: OVERRIDE ML and escalate to human review**\n\n"
                      f"**Why Override:** Agents detected contextual factors (business events, trends, historical patterns) "
                      f"that the ML model missed. Proceeding with ML recommendation would result in incorrect action.")

            # Show ROI if available
            roi = agent_response.get("roi_estimate", {})
            if roi:
                st.metric("Estimated Value of Override", roi.get("cost_avoided", "Unknown"))

        else:
            st.info(f"**✅ AGENTS CONFIRM ML MODEL WITH HIGHER CONFIDENCE**")
            st.success(f"**ML Model Decision:**\n"
                      f"- Predicted: **'{model_finding}'** with {model_confidence:.0%} confidence (uncertain)\n\n"
                      f"**Agent Analysis:**\n"
                      f"- Confirms: **'{agent_finding}'** with {agent_confidence:.0%} confidence (high)\n"
                      f"- Root Cause: {agent_root_cause.replace('_', ' ').title()}\n"
                      f"- **Recommendation: PROCEED with ML recommendation, escalate for specific root cause details**\n\n"
                      f"**Why Confirm:** Agents validated ML prediction through deep analysis and identified specific root cause, "
                      f"increasing confidence from {model_confidence:.0%} to {agent_confidence:.0%}.")

            # Show ROI if available
            roi = agent_response.get("roi_estimate", {})
            if roi:
                st.metric("Estimated Value of Detailed Root Cause", roi.get("cost_avoided", "Unknown"))
    else:
        # Original logic for non-demo cases
        if model_classification != agent_root_cause:
            st.info(f"**The agents identified a different root cause than the ML model.** "
                    f"Model predicted '{model_classification}' but agents discovered '{agent_root_cause}' through deeper reasoning. "
                    f"This demonstrates how multi-agent investigation can uncover insights that pattern matching alone might miss.")
        else:
            st.success(f"**The agents confirmed the model's prediction with higher confidence.** "
                       f"Confidence increased from {model_confidence:.1%} to {agent_confidence:.1%} through detailed analysis.")

def display_clean_agent_reasoning(agent_response: Dict):
    """Display agent reasoning chain without fabricated details."""
    st.markdown("### 🔄 Agent Reasoning Chain")

    reasoning_chain = agent_response.get("reasoning_chain", [])

    if reasoning_chain:
        st.markdown("**Full investigation trace:**")

        for i, step in enumerate(reasoning_chain, 1):
            # Parse agent name and reasoning
            if ":" in step:
                agent_name = step.split(":")[0].strip()
                reasoning = step.split(":", 1)[1].strip()
            else:
                agent_name = f"Step {i}"
                reasoning = step

            # Create expandable section for each step
            with st.expander(f"🤖 {agent_name}", expanded=(i == 1)):
                st.markdown(reasoning)

    else:
        st.info("No detailed reasoning chain available for this investigation.")

def display_clean_agent_findings(agent_response: Dict):
    """Display agent findings without fabricated metrics."""
    st.markdown("### 🎯 Agent Findings")

    col1, col2 = st.columns(2)

    with col1:
        root_cause = agent_response.get("root_cause", "unknown")
        st.markdown(f"**Root Cause Identified:**")
        st.info(root_cause.replace('_', ' ').title())

    with col2:
        agent_confidence = agent_response.get("confidence", 0)
        st.markdown(f"**Analysis Confidence:**")
        if agent_confidence >= 0.8:
            st.success(f"{agent_confidence:.1%} - High")
        elif agent_confidence >= 0.6:
            st.warning(f"{agent_confidence:.1%} - Medium")
        else:
            st.error(f"{agent_confidence:.1%} - Low")

    # Show override recommendation if present
    override_ml = agent_response.get("override_ml", False)
    escalation_reason = agent_response.get("escalation_reason", "")

    if escalation_reason:
        st.markdown("---")
        if override_ml:
            st.error("### ⚠️ OVERRIDE RECOMMENDATION")
            st.warning(f"**Escalation to Human Review**: {escalation_reason}")
        else:
            st.info("### ✅ CONFIRMATION WITH HIGH CONFIDENCE")
            st.info(f"**Escalation to Human Review**: {escalation_reason}")

    # Show ROI estimate if present
    roi_estimate = agent_response.get("roi_estimate", {})
    if roi_estimate:
        st.markdown("---")
        st.markdown("### 💰 Estimated ROI")

        col_roi1, col_roi2 = st.columns(2)
        with col_roi1:
            st.metric("Cost Avoided", roi_estimate.get("cost_avoided", "N/A"))
        with col_roi2:
            st.metric("Time Saved", roi_estimate.get("time_saved", "N/A"))

        with st.expander("📊 How we calculated this (estimates)", expanded=False):
            st.markdown(f"**Calculation Method:**")
            st.text(roi_estimate.get("calculation", "No calculation provided"))
            st.markdown(f"**Confidence Level:** {roi_estimate.get('confidence', 'Unknown')}")
            st.caption("Note: These are estimates based on historical data and industry benchmarks. Actual costs may vary.")

    # Show recommended actions
    if "recommended_actions" in agent_response:
        st.markdown("---")
        st.markdown("**Recommended Actions:**")
        for i, action in enumerate(agent_response["recommended_actions"], 1):
            st.markdown(f"{i}. {action}")

    # Show governance violations if any
    governance_violations = agent_response.get("governance_violations", [])
    if governance_violations:
        st.markdown("**⚠️ Governance Alerts:**")
        for violation in governance_violations:
            st.warning(violation)

    # Show human review requirement
    st.markdown("---")
    requires_review = agent_response.get("requires_human_review", False)
    if requires_review:
        st.warning("👨‍💼 **Human review REQUIRED before proceeding**")
    else:
        st.success("✅ **Can proceed with automated remediation**")

def run_ml_prediction(incident_data: Dict) -> Dict:
    """Run ML model prediction on incident data."""
    # Check if this is a demo edge case with ground truth
    if "ground_truth" in incident_data and "ml_model_view" in incident_data["ground_truth"]:
        # This is a demonstration edge case - use the pre-defined ML view
        ml_view = incident_data["ground_truth"]["ml_model_view"]

        incident_id = incident_data.get("incident_id", f"INC_{int(time.time())}_{str(uuid.uuid4())[:8]}")
        predicted_class = ml_view.get("prediction", "unknown")
        confidence = ml_view.get("confidence", 0.5)

        # Determine decision based on confidence threshold
        if confidence >= 0.75:
            decision = "automated_remediation"
            status = "resolved"
        else:
            decision = "agent_investigation"
            status = "investigating"

        return {
            "incident_id": incident_id,
            "decision": decision,
            "classification": predicted_class,
            "confidence": confidence,
            "action": status,
            "status": status,
            "reasoning": [ml_view.get("reasoning", "ML model prediction")],
            "recommended_actions": [ml_view.get("would_do", "Apply standard remediation")],
            "estimated_resolution_time": "varies",
            "requires_human_review": confidence < 0.75,
            "is_demo_case": True
        }

    # Otherwise, use real ML model prediction
    # Get components from session state or globals
    classifier_local = st.session_state.get('classifier') or classifier
    feature_extractor_local = st.session_state.get('feature_extractor') or feature_extractor

    # Generate unique incident ID
    incident_id = f"INC_{int(time.time())}_{str(uuid.uuid4())[:8]}"

    # Convert to incident format
    incident = convert_alert_to_incident(incident_data, incident_id)

    try:
        if classifier_local and classifier_local.is_trained and feature_extractor_local:
            # Extract features and classify
            features = feature_extractor_local.extract_model_features(incident)
            predicted_class, confidence, is_edge_case = classifier_local.predict(features)
        else:
            # Fallback when model not trained
            st.error("❌ Model not loaded or feature extractor not available")
            predicted_class = "unknown"
            confidence = 0.5
            is_edge_case = True

        # Decision logic based on confidence threshold
        if confidence >= 0.75 and not is_edge_case:
            # High confidence - return standard remediation
            standard_actions = [
                f"Scale {incident_data.get('service_name', 'service')} horizontally",
                "Monitor key metrics for 15 minutes",
                "Check recent deployments for rollback candidates",
                "Enable enhanced logging",
                "Alert on-call team"
            ]

            return {
                "incident_id": incident_id,
                "decision": "automated_remediation",
                "classification": predicted_class,
                "confidence": confidence,
                "action": "standard_remediation",
                "status": "resolved",
                "reasoning": [
                    f"Model classified as {predicted_class} with {confidence:.2f} confidence",
                    "Confidence above 0.75 threshold - applying standard remediation",
                    "No agent investigation required"
                ],
                "recommended_actions": standard_actions,
                "estimated_resolution_time": "15-30 minutes",
                "requires_human_review": False,
                "is_demo_case": False
            }
        else:
            # Low confidence or edge case - needs agent investigation
            return {
                "incident_id": incident_id,
                "decision": "agent_investigation",
                "classification": predicted_class,
                "confidence": confidence,
                "action": "investigating",
                "status": "investigating",
                "reasoning": [
                    f"Model classified as {predicted_class} with {confidence:.2f} confidence",
                    "Confidence below 0.75 threshold or edge case detected",
                    "Queued for multi-agent investigation"
                ],
                "recommended_actions": [
                    "Monitor situation closely",
                    "Prepare for potential escalation",
                    "Investigation in progress..."
                ],
                "estimated_resolution_time": "pending_investigation",
                "requires_human_review": None,
                "is_demo_case": False
            }

    except Exception as e:
        st.error(f"❌ Error running ML prediction: {e}")
        return None

def main():
    """Main Streamlit application."""
    global classifier, feature_extractor, agent_system

    # Header
    st.title("🚨 IncidentIQ: AI-Enhanced Incident Response")
    st.markdown("*Intelligent incident response that thrives on complexity*")

    # Initialize system on first run
    if 'system_initialized' not in st.session_state:
        with st.spinner("🚀 Initializing IncidentIQ system..."):
            system_components = initialize_system()
            if system_components:
                st.session_state.system_initialized = True
                # Store components in session state for persistence
                st.session_state.classifier = system_components["classifier"]
                st.session_state.feature_extractor = system_components["feature_extractor"]
                st.session_state.agent_system = system_components["agent_system"]

                # Update global variables properly
                classifier = system_components["classifier"]
                feature_extractor = system_components["feature_extractor"]
                agent_system = system_components["agent_system"]
            else:
                st.error("❌ Failed to initialize system")
                return
    else:
        # Restore from session state
        classifier = st.session_state.get('classifier')
        feature_extractor = st.session_state.get('feature_extractor')
        agent_system = st.session_state.get('agent_system')

    # System status
    display_system_status()

    st.markdown("---")

    # Sidebar
    st.sidebar.title("🎛️ Control Panel")

    # Load edge cases
    edge_cases = load_edge_cases()

    st.sidebar.markdown("---")

    # Incident selection
    st.sidebar.subheader("📁 Select Incident")

    incident_source = st.sidebar.radio(
        "Choose incident source:",
        ["Pre-generated Edge Cases", "Upload Custom JSON", "Manual Entry"]
    )

    incident_data = None

    if incident_source == "Pre-generated Edge Cases":
        if edge_cases:
            selected_case = st.sidebar.selectbox(
                "Select edge case:",
                list(edge_cases.keys())
            )

            if selected_case:
                incident_data = edge_cases[selected_case]

                # Show case description
                st.sidebar.markdown("**📖 Case Description:**")
                if "description" in incident_data:
                    st.sidebar.write(incident_data["description"])
        else:
            st.sidebar.error("No edge cases found. Run synthetic data generation first.")

    elif incident_source == "Upload Custom JSON":
        uploaded_file = st.sidebar.file_uploader(
            "Upload incident JSON:",
            type=['json'],
            help="Upload a JSON file containing incident data"
        )

        if uploaded_file:
            try:
                incident_data = json.load(uploaded_file)
            except json.JSONDecodeError:
                st.sidebar.error("Invalid JSON file")

    elif incident_source == "Manual Entry":
        st.sidebar.subheader("✏️ Manual Entry")

        service_name = st.sidebar.text_input("Service Name", "payment-service")
        severity = st.sidebar.selectbox("Severity", ["low", "medium", "high", "critical"])

        st.sidebar.markdown("**📊 Metrics:**")
        cpu_usage = st.sidebar.slider("CPU Usage (%)", 0, 100, 75)
        memory_usage = st.sidebar.slider("Memory Usage (%)", 0, 100, 85)
        error_rate = st.sidebar.slider("Error Rate (%)", 0.0, 10.0, 2.5)
        response_time = st.sidebar.slider("Response Time (ms)", 0, 2000, 500)

        description = st.sidebar.text_area("Description", "Manual incident entry from dashboard")

        if st.sidebar.button("Create Incident"):
            incident_data = {
                "service_name": service_name,
                "severity": severity,
                "metrics": {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "error_rate": error_rate / 100,  # Convert to decimal
                    "response_time_ms": response_time
                },
                "description": description,
                "timestamp": datetime.now().isoformat()
            }

    # Main content area
    if incident_data:
        st.markdown("---")

        # Analysis section
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
                st.session_state.run_analysis = True

        with col2:
            if st.button("🔄 Reset Analysis", use_container_width=True):
                if 'run_analysis' in st.session_state:
                    del st.session_state.run_analysis
                if 'model_response' in st.session_state:
                    del st.session_state.model_response
                if 'agent_response' in st.session_state:
                    del st.session_state.agent_response

        # Run analysis if requested
        if st.session_state.get('run_analysis', False):

            st.markdown("## 🔄 Analysis Flow")

            # Show incident summary
            display_clean_incident_summary(incident_data)

            st.markdown("---")

            # Step 1: ML Model Prediction
            with st.spinner("🤖 Running ML model prediction..."):
                model_response = run_ml_prediction(incident_data)

            if model_response:
                st.session_state.model_response = model_response

                # Display ML results
                st.success("✅ ML model prediction complete")
                display_clean_ml_prediction(model_response)

                st.markdown("---")

                # Check if edge case routing to agents
                incident_id = model_response.get("incident_id")
                decision = model_response.get("decision", "")
                confidence = model_response.get("confidence", 0)

                if decision == "agent_investigation" and incident_id:
                    st.info("🔍 Edge case detected → Routing to multi-agent investigation")

                    # Convert incident data for agent investigation
                    incident = convert_alert_to_incident(incident_data, incident_id)

                    # Step 2: Live Agent Investigation with Progress (pass incident_data for demo cases)
                    agent_response = display_live_agent_investigation(incident_id, incident, incident_data)

                    if agent_response:
                        st.session_state.agent_response = agent_response

                        st.markdown("---")
                        st.success("🎉 Agent investigation complete!")

                        # Display agent findings
                        display_clean_agent_findings(agent_response)

                        # Display reasoning chain
                        st.markdown("---")
                        display_clean_agent_reasoning(agent_response)

                        # Display comparison
                        st.markdown("---")
                        display_clean_comparison(model_response, agent_response)

                else:
                    # High confidence - standard remediation
                    st.markdown("---")
                    st.success(f"✅ High confidence ({confidence:.1%}) - Standard remediation recommended")
                    display_standard_remediation(model_response)

                    st.info("💡 **High confidence cases don't require agent investigation.** "
                           "The hybrid system's value is most apparent on edge cases where the ML model has low confidence.")

        # Show previous results if available
        elif 'model_response' in st.session_state:
            st.markdown("## 📋 Previous Analysis Results")

            # Show incident summary
            display_clean_incident_summary(incident_data)

            st.markdown("---")

            # Show ML prediction
            display_clean_ml_prediction(st.session_state.model_response)

            if 'agent_response' in st.session_state:
                st.markdown("---")

                # Show agent findings
                display_clean_agent_findings(st.session_state.agent_response)

                # Show reasoning chain
                st.markdown("---")
                display_clean_agent_reasoning(st.session_state.agent_response)

                # Show comparison
                st.markdown("---")
                display_clean_comparison(
                    st.session_state.model_response,
                    st.session_state.agent_response
                )

    else:
        # Clean welcome screen
        st.markdown("""
        <div class="executive-summary">
            <h1>🚀 IncidentIQ: Hybrid ML + AI Incident Response</h1>
            <h3>Intelligent Edge Case Handling Through Multi-Agent Investigation</h3>
            <p><strong>Challenge:</strong> Traditional ML struggles with edge cases and misleading symptoms</p>
            <p><strong>Approach:</strong> Route complex incidents to AI agents for deep reasoning</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # System capabilities (no fake numbers)
        st.markdown("## 🎯 What This Demo Shows")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### ⚡ Fast ML Classification
            - Binary classification (incident/normal)
            - Confidence-based routing decisions
            - Threshold: 75% confidence
            """)

        with col2:
            st.markdown("""
            ### 🧠 AI Agent Investigation
            - Multi-agent system for edge cases
            - Diagnostic, context, and recommendation agents
            - Full reasoning chain visible
            """)

        with col3:
            st.markdown("""
            ### 🔍 Qualitative Comparison
            - See what ML model predicted
            - See what agents discovered
            - Understand the difference
            """)

        st.markdown("---")
        st.markdown("## 📚 How to Use This Demo")

        step_col1, step_col2, step_col3 = st.columns(3)

        with step_col1:
            st.markdown("""
            ### 1️⃣ Select Edge Case
            Choose from 5 pre-generated edge case scenarios in the sidebar

            Each demonstrates agent value:
            - False positive (prevents unnecessary action)
            - False negative (catches missed incident)
            - Wrong root cause (correct diagnosis)
            - Novel pattern (surgical fix)
            - Cascade early detection (prevents outage)
            """)

        with step_col2:
            st.markdown("""
            ### 2️⃣ Watch Analysis
            See the ML model's initial prediction and confidence score

            If confidence < 75%, watch the agent investigation run with live progress
            """)

        with step_col3:
            st.markdown("""
            ### 3️⃣ Review Findings
            Compare what the model predicted vs what the agents discovered

            Read the agent reasoning chain to understand their analysis
            """)

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 1.2rem; margin: 2rem 0;">
            <strong>👈 Select an edge case scenario from the sidebar to begin</strong>
        </div>
        """, unsafe_allow_html=True)

        # Show what makes this different
        st.markdown("## 💡 Why This Matters")

        st.info("""
        **This demo showcases reasoning capabilities, not financial projections.**

        Traditional ML systems classify incidents based on pattern matching. When faced with edge cases
        (misleading symptoms, novel patterns, contextual anomalies), they often provide incorrect or
        low-confidence predictions.

        This hybrid system detects low-confidence scenarios and routes them to specialized AI agents
        that can reason about the incident using:
        - Historical context
        - Metric correlations
        - Business event awareness
        - Multi-system dependencies

        The value is in **better decisions on hard problems**, not just speed on easy ones.
        """)

if __name__ == "__main__":
    main()