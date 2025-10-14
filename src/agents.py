"""Multi-agent incident analysis system using LangGraph."""

import json
import os
import sys
from typing import Dict, List, Any, TypedDict, Optional
from dataclasses import asdict
import asyncio

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_anthropic import ChatAnthropic
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Install with: pip install langgraph langchain-anthropic")

    # Mock classes for testing without LangGraph
    class StateGraph:
        def __init__(self, state_type):
            self.state_type = state_type

        def add_node(self, name, func):
            pass

        def add_edge(self, from_node, to_node):
            pass

        def add_conditional_edges(self, from_node, condition_func, mapping):
            pass

        def set_entry_point(self, node):
            pass

        def compile(self):
            return MockWorkflow()

    class MockWorkflow:
        def __init__(self):
            self.agent_system = None

        async def ainvoke(self, state):
            # Mock workflow that simulates the agent sequence
            if self.agent_system:
                # Run through the agent sequence in mock mode
                state = await self.agent_system.diagnostic_agent(state)
                state = await self.agent_system.context_agent(state)
                state = await self.agent_system.recommendation_agent(state)
                state = await self.agent_system.governance_check(state)
            return state

    END = "END"

    class SystemMessage:
        def __init__(self, content):
            self.content = content

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class ChatAnthropic:
        def __init__(self, **kwargs):
            pass

        async def ainvoke(self, messages):
            class MockResponse:
                content = "Mock LLM response"
            return MockResponse()

# Import project modules
from src.synthetic_data import SyntheticIncident
from src.features import IncidentFeatureExtractor
from src.model import IncidentClassifier


class IncidentState(TypedDict):
    """State definition for incident analysis workflow."""
    # Input data
    incident_id: str
    incident_data: Dict[str, Any]
    context: Dict[str, Any]

    # Model prediction
    model_prediction: Dict[str, Any]

    # Agent results
    diagnostic_result: Optional[Dict[str, Any]]
    context_analysis: Optional[Dict[str, Any]]
    recommendation: Optional[Dict[str, Any]]

    # Final output
    root_cause: Optional[str]
    confidence: Optional[float]
    recommended_actions: Optional[List[str]]
    reasoning_chain: Optional[List[str]]

    # Workflow control
    needs_governance_check: bool
    is_complete: bool
    loop_count: int


class IncidentAgentSystem:
    """Multi-agent system for incident analysis using LangGraph."""

    def __init__(self, anthropic_api_key: Optional[str] = None):
        """Initialize the agent system.

        Args:
            anthropic_api_key: Anthropic API key. If None, will try to get from env.
        """
        if not LANGGRAPH_AVAILABLE:
            print("Running in mock mode. LangGraph not available.")

        # Initialize API key
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

        if not anthropic_api_key:
            print("Warning: No Anthropic API key provided. Agent system will run in mock mode.")
            self.mock_mode = True
            self.llm = None
        else:
            self.mock_mode = False
            # Use Claude Haiku for cost-effectiveness (temperature=0 for consistency)
            self.llm = ChatAnthropic(
                anthropic_api_key=anthropic_api_key,
                model="claude-3-haiku-20240307",
                temperature=0.0,  # For deterministic responses
                max_tokens=2000
            )

        # Initialize ML components
        self.feature_extractor = IncidentFeatureExtractor()
        self.classifier = IncidentClassifier()

        # Try to load trained model
        try:
            self.classifier.load('./models/incident_classifier')
        except Exception as e:
            print(f"Warning: Could not load trained model: {e}")

        # Build workflow graph
        self.workflow = self._build_agent_graph()

    def _build_agent_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(IncidentState)

        # Add nodes
        workflow.add_node("diagnostic_agent", self.diagnostic_agent)
        workflow.add_node("context_agent", self.context_agent)
        workflow.add_node("recommendation_agent", self.recommendation_agent)
        workflow.add_node("governance_check", self.governance_check)

        # Define workflow edges
        workflow.set_entry_point("diagnostic_agent")
        workflow.add_edge("diagnostic_agent", "context_agent")
        workflow.add_edge("context_agent", "recommendation_agent")
        workflow.add_edge("recommendation_agent", "governance_check")

        # Conditional edge from governance
        workflow.add_conditional_edges(
            "governance_check",
            self._should_continue,
            {
                "continue": "diagnostic_agent",  # Loop back for refinement
                "end": END
            }
        )

        # Compile workflow with recursion limit
        try:
            compiled_workflow = workflow.compile(
                config={
                    "recursion_limit": 10  # Limit to prevent infinite loops
                }
            )
        except TypeError:
            # Fallback for older LangGraph versions
            compiled_workflow = workflow.compile()

        # For mock mode, set agent system reference
        if hasattr(compiled_workflow, 'agent_system'):
            compiled_workflow.agent_system = self
        return compiled_workflow

    def _should_continue(self, state: IncidentState) -> str:
        """Determine if workflow should continue or end."""
        if state.get("is_complete", False):
            return "end"

        # Add loop counter to prevent infinite recursion
        loop_count = state.get("loop_count", 0)
        if loop_count >= 3:  # Maximum 3 iterations
            print(f"[WARNING] Maximum loop iterations reached ({loop_count}), forcing completion")
            state["is_complete"] = True
            state["reasoning_chain"].append(f"Governance: Maximum analysis iterations reached ({loop_count}), proceeding with current results")
            return "end"

        state["loop_count"] = loop_count + 1
        return "continue"

    async def diagnostic_agent(self, state: IncidentState) -> IncidentState:
        """Diagnostic agent: Investigates root cause and proposes hypothesis."""
        print("[DIAGNOSTIC] Investigating root cause...")

        incident_data = state["incident_data"]
        model_pred = state["model_prediction"]

        if self.mock_mode:
            # Mock response for testing
            diagnostic_result = {
                "root_cause_hypothesis": model_pred["predicted_class"],
                "confidence": float(model_pred["confidence"]),
                "evidence": [
                    f"Model prediction: {model_pred['predicted_class']} ({model_pred['confidence']:.2f})",
                    "Metrics analysis shows anomalous patterns",
                    "Historical incident patterns match current symptoms"
                ],
                "reasoning": f"Based on model prediction and metric analysis, this appears to be a {model_pred['predicted_class']} incident.",
                "conflicting_signals": []
            }
        else:
            # Real LLM call
            diagnostic_prompt = f"""You are a diagnostic agent for incident analysis.

MODEL PREDICTION: {model_pred['predicted_class']} (confidence: {model_pred['confidence']:.2f})

INCIDENT METRICS:
{json.dumps(incident_data.get('metrics', {}), indent=2)}

HISTORICAL CONTEXT:
{json.dumps(incident_data.get('context', {}).get('historical_incidents', []), indent=2)}

TASK: Investigate if the model prediction is correct. Look for:
1. Supporting evidence in the metrics
2. Conflicting signals that might indicate a different root cause
3. Patterns matching historical incidents

Provide your analysis in this format:
ROOT_CAUSE: [your hypothesis]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [list of supporting evidence]
REASONING: [your reasoning process]
CONFLICTING_SIGNALS: [any contradictory evidence]
"""

            try:
                messages = [
                    SystemMessage(content="You are an expert incident diagnostic agent."),
                    HumanMessage(content=diagnostic_prompt)
                ]
                response = await self.llm.ainvoke(messages)

                # Parse LLM response (simplified parsing)
                response_text = response.content
                diagnostic_result = self._parse_diagnostic_response(response_text, model_pred)

            except Exception as e:
                print(f"Error in diagnostic agent: {e}")
                # Fallback to model prediction
                diagnostic_result = {
                    "root_cause_hypothesis": model_pred["predicted_class"],
                    "confidence": float(model_pred["confidence"]),
                    "evidence": [f"Model prediction: {model_pred['predicted_class']}"],
                    "reasoning": "Fallback to model prediction due to LLM error",
                    "conflicting_signals": []
                }

        state["diagnostic_result"] = diagnostic_result
        state["reasoning_chain"] = [f"Diagnostic: {diagnostic_result['reasoning']}"]

        return state

    async def context_agent(self, state: IncidentState) -> IncidentState:
        """Context agent: Adds business/operational context and challenges findings."""
        print("[CONTEXT] Analyzing business context...")

        diagnostic = state["diagnostic_result"]
        incident_data = state["incident_data"]
        context = incident_data.get("context", {})

        if self.mock_mode:
            # Mock context analysis
            context_analysis = {
                "business_impact": "moderate",
                "expected_behavior": False,
                "context_factors": [
                    f"Business event: {context.get('business_event', 'normal_operations')}",
                    f"Recent deployments: {context.get('recent_deployments', [])}",
                    f"Traffic multiplier: {context.get('traffic_multiplier', 1.0)}"
                ],
                "diagnosis_adjustment": "confirmed",
                "adjusted_confidence": diagnostic["confidence"],
                "reasoning": "Context supports diagnostic findings. No conflicting business events detected."
            }
        else:
            # Real LLM call
            context_prompt = f"""You are a context analysis agent.

DIAGNOSTIC HYPOTHESIS: {diagnostic['root_cause_hypothesis']} (confidence: {diagnostic['confidence']:.2f})
DIAGNOSTIC REASONING: {diagnostic['reasoning']}

BUSINESS CONTEXT:
- Business Event: {context.get('business_event', 'normal_operations')}
- Recent Deployments: {context.get('recent_deployments', [])}
- Traffic Multiplier: {context.get('traffic_multiplier', 1.0)}
- Geographic Distribution: {context.get('geographic_distribution', {})}
- Feature Flags: {context.get('feature_flags', [])}

TASK: Evaluate if the business context changes the diagnosis:
1. Is this expected behavior given the context?
2. Do deployments/events explain the symptoms?
3. Should confidence be adjusted based on context?

Provide your analysis in this format:
EXPECTED_BEHAVIOR: [yes/no and why]
CONTEXT_FACTORS: [relevant context elements]
DIAGNOSIS_ADJUSTMENT: [confirmed/challenged/modified]
ADJUSTED_CONFIDENCE: [0.0-1.0]
REASONING: [your reasoning]
"""

            try:
                messages = [
                    SystemMessage(content="You are an expert context analysis agent."),
                    HumanMessage(content=context_prompt)
                ]
                response = await self.llm.ainvoke(messages)

                context_analysis = self._parse_context_response(response.content, diagnostic)

            except Exception as e:
                print(f"Error in context agent: {e}")
                # Fallback
                context_analysis = {
                    "business_impact": "moderate",
                    "expected_behavior": False,
                    "context_factors": ["Error in context analysis"],
                    "diagnosis_adjustment": "confirmed",
                    "adjusted_confidence": diagnostic["confidence"],
                    "reasoning": "Fallback due to LLM error"
                }

        state["context_analysis"] = context_analysis
        state["reasoning_chain"].append(f"Context: {context_analysis['reasoning']}")

        return state

    async def recommendation_agent(self, state: IncidentState) -> IncidentState:
        """Recommendation agent: Synthesizes diagnostic + context analysis into specific actions."""
        print("[RECOMMENDATION] Generating action plan...")

        diagnostic = state["diagnostic_result"]
        context_analysis = state["context_analysis"]

        if self.mock_mode:
            # Mock recommendations with enhanced details
            root_cause = diagnostic["root_cause_hypothesis"]
            if context_analysis["diagnosis_adjustment"] == "modified":
                root_cause = f"context_modified_{root_cause}"

            # Calculate final confidence based on diagnostic and context
            final_confidence = min(
                diagnostic["confidence"] * 0.7 + context_analysis["adjusted_confidence"] * 0.3,
                1.0
            )

            recommendations = {
                "root_cause": root_cause,
                "confidence": final_confidence,
                "immediate_actions": [
                    f"Investigate {root_cause} components immediately",
                    "Monitor key metrics every 30 seconds",
                    "Prepare rollback plan if deployment-related",
                    "Alert on-call team for incident response"
                ],
                "mitigation_steps": [
                    "Scale affected services horizontally",
                    "Route traffic away from affected regions",
                    "Enable enhanced monitoring and logging",
                    "Implement circuit breakers if not present"
                ],
                "prevention_measures": [
                    "Add alerting for similar patterns",
                    "Review deployment procedures",
                    "Update runbooks with lessons learned",
                    "Conduct post-incident review"
                ],
                "estimated_resolution_time": "45-90 minutes",
                "estimated_cost": "$500-2000 (infrastructure scaling)",
                "business_impact": context_analysis["business_impact"],
                "priority": "high" if final_confidence > 0.8 else "medium"
            }
        else:
            # Real LLM call with enhanced prompt
            recommendation_prompt = f"""You are a senior incident response recommendation agent.

DIAGNOSTIC ANALYSIS:
- Root Cause: {diagnostic['root_cause_hypothesis']}
- Confidence: {diagnostic['confidence']:.2f}
- Evidence: {'; '.join(diagnostic['evidence'])}

CONTEXT ANALYSIS:
- Adjustment: {context_analysis['diagnosis_adjustment']}
- Business Impact: {context_analysis.get('business_impact', 'unknown')}
- Context Factors: {'; '.join(context_analysis['context_factors'])}
- Adjusted Confidence: {context_analysis['adjusted_confidence']:.2f}

TASK: Synthesize the diagnostic and context analysis to propose specific actions.

Provide a comprehensive incident response plan:

1. IMMEDIATE ACTIONS (next 15 minutes):
   - Specific technical steps
   - Monitoring adjustments
   - Team notifications

2. MITIGATION STEPS (next 1-2 hours):
   - Infrastructure changes
   - Traffic management
   - Service adjustments

3. PREVENTION MEASURES (long-term):
   - Process improvements
   - Monitoring enhancements
   - Documentation updates

4. ESTIMATES:
   - Resolution time
   - Approximate cost
   - Resource requirements

5. FINAL ASSESSMENT:
   - Business impact level
   - Incident priority
   - Confidence in recommendations

Format your response as:
IMMEDIATE_ACTIONS: [specific action 1], [specific action 2], ...
MITIGATION_STEPS: [step 1], [step 2], ...
PREVENTION_MEASURES: [measure 1], [measure 2], ...
RESOLUTION_TIME: [time estimate]
COST_ESTIMATE: [cost range]
BUSINESS_IMPACT: [low/medium/high with reasoning]
PRIORITY: [low/medium/high/critical]
FINAL_CONFIDENCE: [0.0-1.0]
"""

            try:
                messages = [
                    SystemMessage(content="You are an expert incident response recommendation agent with 10+ years experience."),
                    HumanMessage(content=recommendation_prompt)
                ]
                response = await self.llm.ainvoke(messages)

                recommendations = self._parse_recommendation_response(
                    response.content, diagnostic, context_analysis
                )

            except Exception as e:
                print(f"Error in recommendation agent: {e}")
                # Enhanced fallback
                final_confidence = min(
                    diagnostic["confidence"] * 0.7 + context_analysis["adjusted_confidence"] * 0.3,
                    1.0
                )
                recommendations = {
                    "root_cause": diagnostic["root_cause_hypothesis"],
                    "confidence": final_confidence,
                    "immediate_actions": ["Investigate the incident", "Monitor systems", "Alert team"],
                    "mitigation_steps": ["Scale services", "Enable monitoring", "Reroute traffic"],
                    "prevention_measures": ["Update alerting", "Review procedures", "Document lessons"],
                    "estimated_resolution_time": "60-120 minutes",
                    "estimated_cost": "$200-1000",
                    "business_impact": context_analysis.get("business_impact", "medium"),
                    "priority": "medium"
                }

        state["recommendation"] = recommendations
        state["root_cause"] = recommendations["root_cause"]
        state["confidence"] = recommendations["confidence"]
        state["recommended_actions"] = (
            recommendations["immediate_actions"] +
            recommendations["mitigation_steps"]
        )
        state["reasoning_chain"].append(
            f"Recommendation: Generated {len(state['recommended_actions'])} actions, "
            f"priority: {recommendations.get('priority', 'unknown')}, "
            f"est. time: {recommendations.get('estimated_resolution_time', 'unknown')}"
        )

        return state

    async def governance_check(self, state: IncidentState) -> IncidentState:
        """Governance check: Enforces hard rules and validates analysis quality."""
        print("[GOVERNANCE] Validating analysis...")

        confidence = state["confidence"]
        root_cause = state["root_cause"]
        recommendations = state["recommendation"]

        # Initialize governance state
        is_complete = True
        requires_human_review = False
        governance_violations = []

        # HARD RULE 1: Security incidents always escalate (cannot be overridden)
        security_keywords = ['security', 'breach', 'attack', 'malware', 'unauthorized', 'vulnerability']
        if any(keyword in root_cause.lower() for keyword in security_keywords):
            requires_human_review = True
            governance_violations.append("SECURITY_ESCALATION: Security incident detected - mandatory human review")
            state["reasoning_chain"].append("Governance: Security incident - escalating to security team")
            print("[CRITICAL] Security incident detected - mandatory escalation")

        # HARD RULE 2: High business impact incidents require senior approval
        business_impact = recommendations.get("business_impact", "unknown").lower()
        if business_impact in ["high", "critical"]:
            requires_human_review = True
            governance_violations.append(f"HIGH_IMPACT_ESCALATION: {business_impact} business impact requires approval")
            state["reasoning_chain"].append(f"Governance: {business_impact} impact - requiring senior approval")
            print(f"[CRITICAL] {business_impact.title()} business impact - requiring approval")

        # HARD RULE 3: Low confidence threshold (0.70) triggers review
        if confidence < 0.70:
            requires_human_review = True
            governance_violations.append(f"LOW_CONFIDENCE: {confidence:.2f} < 0.70 threshold")
            state["reasoning_chain"].append(f"Governance: Low confidence ({confidence:.2f}) - human review required")
            print(f"[WARNING] Low confidence ({confidence:.2f}) - triggering human review")

        # HARD RULE 4: Unknown root causes require expert investigation
        if "unknown" in root_cause.lower():
            requires_human_review = True
            governance_violations.append("UNKNOWN_ROOT_CAUSE: Expert investigation required")
            state["reasoning_chain"].append("Governance: Unknown root cause - expert investigation needed")
            print("[WARNING] Unknown root cause - expert investigation required")

        # VALIDATION RULE 5: Cost estimates must be within bounds
        cost_estimate = recommendations.get("estimated_cost", "")
        if cost_estimate and ">" in cost_estimate:  # High cost indicator
            requires_human_review = True
            governance_violations.append("HIGH_COST_ESCALATION: Cost estimate requires approval")
            state["reasoning_chain"].append("Governance: High cost estimate - requiring financial approval")
            print("[WARNING] High cost estimate - requiring approval")

        # VALIDATION RULE 6: Sufficient recommendations check
        if len(state["recommended_actions"]) < 3:
            is_complete = False
            governance_violations.append("INSUFFICIENT_RECOMMENDATIONS: Less than 3 actions generated")
            state["reasoning_chain"].append("Governance: Insufficient recommendations - requesting more detail")
            print("[WARNING] Insufficient recommendations - requesting more detail")

        # VALIDATION RULE 7: Resolution time reasonableness check
        resolution_time = recommendations.get("estimated_resolution_time", "")
        if "hours" in resolution_time.lower() or "days" in resolution_time.lower():
            requires_human_review = True
            governance_violations.append("LONG_RESOLUTION_TIME: Extended resolution time requires oversight")
            state["reasoning_chain"].append("Governance: Long resolution time - oversight required")
            print("[WARNING] Extended resolution time - requiring oversight")

        # HARD RULE 8: Critical priority incidents get immediate escalation
        priority = recommendations.get("priority", "medium").lower()
        if priority == "critical":
            requires_human_review = True
            governance_violations.append("CRITICAL_PRIORITY: Immediate escalation required")
            state["reasoning_chain"].append("Governance: Critical priority - immediate escalation")
            print("[CRITICAL] Critical priority incident - immediate escalation")

        # Set final state - be more aggressive about completion to prevent loops
        # Complete the workflow if we have sufficient information, even with violations
        loop_count = state.get("loop_count", 0)
        force_completion = (loop_count >= 2) or len(state["recommended_actions"]) >= 3

        state["is_complete"] = is_complete or force_completion
        state["needs_governance_check"] = not state["is_complete"]
        state["requires_human_review"] = requires_human_review
        state["governance_violations"] = governance_violations

        if force_completion and not is_complete:
            state["reasoning_chain"].append(f"Governance: Forced completion after {loop_count} iterations to prevent loops")

        # Log final governance decision
        if requires_human_review:
            print(f"[ESCALATION] Human review required: {len(governance_violations)} governance rules triggered")
            state["reasoning_chain"].append(f"Governance: Escalating to human review ({len(governance_violations)} rules triggered)")
        elif not is_complete:
            print("[INFO] Analysis needs refinement - workflow will continue")
        else:
            print("[SUCCESS] Analysis approved - no governance violations")
            state["reasoning_chain"].append("Governance: Analysis approved for automated execution")

        # Add governance summary to state
        state["governance_summary"] = {
            "approved_for_automation": state["is_complete"],
            "requires_human_review": requires_human_review,
            "violations": governance_violations,
            "confidence_threshold_met": confidence >= 0.70,
            "security_incident": any("SECURITY" in v for v in governance_violations),
            "high_impact": any("HIGH_IMPACT" in v for v in governance_violations)
        }

        return state

    def _parse_diagnostic_response(self, response: str, model_pred: Dict) -> Dict[str, Any]:
        """Parse diagnostic agent response with robust extraction."""
        lines = response.split('\n')
        result = {
            "root_cause_hypothesis": model_pred["predicted_class"],
            "confidence": float(model_pred["confidence"]),
            "evidence": ["Model prediction"],
            "reasoning": "Parsed from LLM response",
            "conflicting_signals": []
        }

        evidence_list = []
        conflicting_list = []

        for line in lines:
            line = line.strip()
            if line.startswith("ROOT_CAUSE:"):
                result["root_cause_hypothesis"] = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    # Handle percentage format
                    if "%" in conf_str:
                        conf_str = conf_str.replace("%", "")
                        result["confidence"] = float(conf_str) / 100.0
                    else:
                        result["confidence"] = float(conf_str)
                    # Clamp to valid range
                    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                except:
                    pass
            elif line.startswith("EVIDENCE:"):
                evidence_text = line.split(":", 1)[1].strip()
                # Parse list format
                if evidence_text.startswith("[") and evidence_text.endswith("]"):
                    evidence_text = evidence_text[1:-1]
                evidence_list = [e.strip() for e in evidence_text.split(",") if e.strip()]
                if evidence_list:
                    result["evidence"] = evidence_list
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("CONFLICTING_SIGNALS:"):
                conflicting_text = line.split(":", 1)[1].strip()
                if conflicting_text.startswith("[") and conflicting_text.endswith("]"):
                    conflicting_text = conflicting_text[1:-1]
                conflicting_list = [c.strip() for c in conflicting_text.split(",") if c.strip()]
                if conflicting_list:
                    result["conflicting_signals"] = conflicting_list

        return result

    def _parse_context_response(self, response: str, diagnostic: Dict) -> Dict[str, Any]:
        """Parse context agent response with structured extraction."""
        lines = response.split('\n')
        result = {
            "business_impact": "moderate",
            "expected_behavior": False,
            "context_factors": ["Parsed from LLM"],
            "diagnosis_adjustment": "confirmed",
            "adjusted_confidence": diagnostic["confidence"],
            "reasoning": "Parsed from LLM response"
        }

        for line in lines:
            line = line.strip()
            if line.startswith("EXPECTED_BEHAVIOR:"):
                behavior_text = line.split(":", 1)[1].strip().lower()
                result["expected_behavior"] = behavior_text.startswith("yes")
            elif line.startswith("CONTEXT_FACTORS:"):
                factors_text = line.split(":", 1)[1].strip()
                if factors_text.startswith("[") and factors_text.endswith("]"):
                    factors_text = factors_text[1:-1]
                factors = [f.strip() for f in factors_text.split(",") if f.strip()]
                if factors:
                    result["context_factors"] = factors
            elif line.startswith("DIAGNOSIS_ADJUSTMENT:"):
                adjustment = line.split(":", 1)[1].strip().lower()
                result["diagnosis_adjustment"] = adjustment
            elif line.startswith("ADJUSTED_CONFIDENCE:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    if "%" in conf_str:
                        conf_str = conf_str.replace("%", "")
                        result["adjusted_confidence"] = float(conf_str) / 100.0
                    else:
                        result["adjusted_confidence"] = float(conf_str)
                    result["adjusted_confidence"] = max(0.0, min(1.0, result["adjusted_confidence"]))
                except:
                    pass
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        return result

    def _parse_recommendation_response(self, response: str, diagnostic: Dict, context: Dict) -> Dict[str, Any]:
        """Parse recommendation agent response with comprehensive extraction."""
        lines = response.split('\n')
        result = {
            "root_cause": diagnostic["root_cause_hypothesis"],
            "confidence": context["adjusted_confidence"],
            "immediate_actions": [],
            "mitigation_steps": [],
            "prevention_measures": [],
            "estimated_resolution_time": "60-120 minutes",
            "estimated_cost": "$200-1000",
            "business_impact": context.get("business_impact", "medium"),
            "priority": "medium"
        }

        def parse_list_field(text):
            """Helper to parse list fields from LLM response."""
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            return [item.strip() for item in text.split(",") if item.strip()]

        for line in lines:
            line = line.strip()
            if line.startswith("IMMEDIATE_ACTIONS:"):
                actions_text = line.split(":", 1)[1].strip()
                actions = parse_list_field(actions_text)
                if actions:
                    result["immediate_actions"] = actions
            elif line.startswith("MITIGATION_STEPS:"):
                steps_text = line.split(":", 1)[1].strip()
                steps = parse_list_field(steps_text)
                if steps:
                    result["mitigation_steps"] = steps
            elif line.startswith("PREVENTION_MEASURES:"):
                measures_text = line.split(":", 1)[1].strip()
                measures = parse_list_field(measures_text)
                if measures:
                    result["prevention_measures"] = measures
            elif line.startswith("RESOLUTION_TIME:"):
                result["estimated_resolution_time"] = line.split(":", 1)[1].strip()
            elif line.startswith("COST_ESTIMATE:"):
                result["estimated_cost"] = line.split(":", 1)[1].strip()
            elif line.startswith("BUSINESS_IMPACT:"):
                impact_text = line.split(":", 1)[1].strip().lower()
                # Extract just the impact level
                if any(level in impact_text for level in ["low", "medium", "high", "critical"]):
                    for level in ["critical", "high", "medium", "low"]:
                        if level in impact_text:
                            result["business_impact"] = level
                            break
            elif line.startswith("PRIORITY:"):
                priority_text = line.split(":", 1)[1].strip().lower()
                if any(level in priority_text for level in ["low", "medium", "high", "critical"]):
                    for level in ["critical", "high", "medium", "low"]:
                        if level in priority_text:
                            result["priority"] = level
                            break
            elif line.startswith("FINAL_CONFIDENCE:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    if "%" in conf_str:
                        conf_str = conf_str.replace("%", "")
                        result["confidence"] = float(conf_str) / 100.0
                    else:
                        result["confidence"] = float(conf_str)
                    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                except:
                    pass

        return result

    async def investigate(self, incident_data: Dict, context: Dict, model_prediction: Dict) -> Dict[str, Any]:
        """Run full agent workflow for incident investigation.

        Args:
            incident_data: Raw incident data and metrics
            context: Business and operational context
            model_prediction: ML model prediction results

        Returns:
            Investigation results with reasoning chain
        """
        print(f"[INVESTIGATE] Starting agent investigation workflow")

        # Initialize state for workflow
        initial_state = IncidentState(
            incident_id=incident_data.get("incident_id", "UNKNOWN"),
            incident_data=incident_data,
            context=context,
            model_prediction=model_prediction,
            diagnostic_result=None,
            context_analysis=None,
            recommendation=None,
            root_cause=None,
            confidence=None,
            recommended_actions=None,
            reasoning_chain=[],
            needs_governance_check=True,
            is_complete=False,
            loop_count=0
        )

        # Run workflow
        try:
            final_state = await self.workflow.ainvoke(initial_state)
        except Exception as e:
            print(f"Error in investigation workflow: {e}")
            # Return fallback result
            return {
                "incident_id": initial_state["incident_id"],
                "root_cause": model_prediction.get("predicted_class", "unknown"),
                "confidence": model_prediction.get("confidence", 0.5),
                "recommended_actions": ["Manual investigation required"],
                "reasoning_chain": [f"Workflow error: {e}"],
                "investigation_complete": False,
                "requires_human_review": True,
                "governance_violations": ["WORKFLOW_ERROR"],
                "error": str(e)
            }

        # Format investigation results
        result = {
            "incident_id": final_state["incident_id"],
            "root_cause": final_state["root_cause"],
            "confidence": final_state["confidence"],
            "recommended_actions": final_state["recommended_actions"],
            "reasoning_chain": final_state["reasoning_chain"],
            "investigation_complete": final_state["is_complete"],
            "requires_human_review": final_state.get("requires_human_review", False),
            "governance_violations": final_state.get("governance_violations", []),
            "governance_summary": final_state.get("governance_summary", {}),
            "model_prediction": model_prediction,
            "diagnostic_result": final_state["diagnostic_result"],
            "context_analysis": final_state["context_analysis"],
            "recommendation": final_state["recommendation"]
        }

        print(f"[INVESTIGATE] Investigation complete for {final_state['incident_id']}")
        return result

    async def analyze_incident(self, incident: SyntheticIncident) -> Dict[str, Any]:
        """Analyze an incident using the multi-agent workflow.

        Args:
            incident: The incident to analyze

        Returns:
            Analysis results with root cause, confidence, and recommendations
        """
        print(f"[ALERT] Starting multi-agent analysis for incident {incident.incident_id}")

        # Extract model features and get prediction
        if self.classifier.is_trained:
            features = self.feature_extractor.extract_model_features(incident)
            pred_class, confidence, is_edge = self.classifier.predict(features)
            model_prediction = {
                "predicted_class": pred_class,
                "confidence": confidence,
                "is_edge_case": is_edge
            }
        else:
            # Fallback if model not trained
            model_prediction = {
                "predicted_class": "unknown",
                "confidence": 0.5,
                "is_edge_case": True
            }

        # Initialize state
        initial_state = IncidentState(
            incident_id=incident.incident_id,
            incident_data=asdict(incident),
            context=asdict(incident.context),
            model_prediction=model_prediction,
            diagnostic_result=None,
            context_analysis=None,
            recommendation=None,
            root_cause=None,
            confidence=None,
            recommended_actions=None,
            reasoning_chain=[],
            needs_governance_check=True,
            is_complete=False,
            loop_count=0
        )

        # Run workflow
        try:
            final_state = await self.workflow.ainvoke(initial_state)
        except Exception as e:
            print(f"Error in workflow execution: {e}")
            # Return fallback result
            return {
                "incident_id": incident.incident_id,
                "root_cause": model_prediction["predicted_class"],
                "confidence": model_prediction["confidence"],
                "recommended_actions": ["Manual investigation required"],
                "reasoning_chain": [f"Workflow error: {e}"],
                "analysis_complete": False,
                "error": str(e)
            }

        # Format final results
        result = {
            "incident_id": final_state["incident_id"],
            "root_cause": final_state["root_cause"],
            "confidence": final_state["confidence"],
            "recommended_actions": final_state["recommended_actions"],
            "reasoning_chain": final_state["reasoning_chain"],
            "model_prediction": model_prediction,
            "diagnostic_result": final_state["diagnostic_result"],
            "context_analysis": final_state["context_analysis"],
            "recommendation": final_state["recommendation"],
            "analysis_complete": final_state["is_complete"]
        }

        print(f"[SUCCESS] Analysis complete for {incident.incident_id}")
        return result


# Example usage and testing
if __name__ == "__main__":
    async def test_agent_system():
        """Test the agent system with a sample incident."""
        print("Testing IncidentAgentSystem...")
        print("=" * 50)

        # Create agent system (will run in mock mode without API key)
        agent_system = IncidentAgentSystem()

        # Generate a test incident
        from src.synthetic_data import SyntheticIncidentGenerator
        generator = SyntheticIncidentGenerator(seed=42)
        incidents = generator.generate_training_dataset(n_samples=1)
        test_incident = incidents[0]

        print(f"Test incident: {test_incident.incident_id}")
        print(f"Ground truth: {test_incident.ground_truth.get('actual_root_cause', 'unknown')}")

        # Analyze the incident
        result = await agent_system.analyze_incident(test_incident)

        print("\n[RESULTS] Analysis Results:")
        print(f"Root Cause: {result['root_cause']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Actions: {len(result['recommended_actions'])}")
        print(f"Complete: {result['analysis_complete']}")

        print("\n[REASONING] Reasoning Chain:")
        for i, step in enumerate(result['reasoning_chain'], 1):
            print(f"{i}. {step}")

        print("\n[ACTIONS] Recommended Actions:")
        for i, action in enumerate(result['recommended_actions'], 1):
            print(f"{i}. {action}")

        return result

    async def test_investigate_method():
        """Test the investigate method directly."""
        print("\n" + "=" * 50)
        print("Testing investigate() method...")
        print("=" * 50)

        # Create agent system
        agent_system = IncidentAgentSystem()

        # Sample incident data
        incident_data = {
            "incident_id": "TEST_001",
            "metrics": {
                "cpu_usage": 85.0,
                "memory_usage": 90.0,
                "response_time_ms": 2500.0,
                "error_rate": 0.15
            }
        }

        context = {
            "business_event": "black_friday",
            "recent_deployments": ["auth-service-v2.1.0"],
            "traffic_multiplier": 3.5
        }

        model_prediction = {
            "predicted_class": "memory_leak",
            "confidence": 0.85,
            "is_edge_case": False
        }

        # Run investigation
        result = await agent_system.investigate(incident_data, context, model_prediction)

        print(f"\n[INVESTIGATION] Results for {result['incident_id']}:")
        print(f"Root Cause: {result['root_cause']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Human Review Required: {result['requires_human_review']}")
        print(f"Governance Violations: {len(result['governance_violations'])}")
        print(f"Actions Generated: {len(result['recommended_actions'])}")

        if result['governance_violations']:
            print("\n[GOVERNANCE] Violations:")
            for violation in result['governance_violations']:
                print(f"  - {violation}")

        return result

    # Run the tests (works in mock mode too)
    try:
        import asyncio
        print("Running agent system tests...")
        asyncio.run(test_agent_system())
        asyncio.run(test_investigate_method())
        print("\n[SUCCESS] All tests completed successfully!")
    except Exception as e:
        print(f"Error running test: {e}")
        print("Agent system implementation complete. Install LangGraph for full functionality.")