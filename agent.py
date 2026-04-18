import json
import os
from pathlib import Path
import pandas as pd
import joblib
from typing import TypedDict
from dotenv import load_dotenv

# Load environment variables right after standard library/basic imports
load_dotenv()

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from rag_setup import get_retriever

# 1. Threshold Standardization
CHURN_THRESHOLD = 0.3

# 2. State Definition
class AgentState(TypedDict):
    customer_data: dict
    churn_probability: float
    key_factors: list[str]
    retrieved_strategies: str
    final_report: dict

# 3. Structured Output Schema
class RetentionReport(BaseModel):
    risk_summary: str = Field(description="A 1-2 sentence summary of the customer's flight risk.")
    contributing_factors: list[str] = Field(description="Bullet points of why they might leave.")
    recommended_actions: list[str] = Field(description="Actionable, specific strategies based on the knowledge base.")
    disclaimer: str = Field(
        default="AI-generated recommendation. Verify before applying account credits or policy changes.", 
        description="A required legal/business disclaimer."
    )

# 4. LLM Initialization (lazy to avoid import-time crash if key is missing)
_llm_client = None


def get_llm_client():
    """Creates Gemini client on first use. Returns None when API key is unavailable."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    _llm_client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    return _llm_client

# 5. Graph Nodes
def analyze_factors(state: AgentState):
    """Loads feature importance to identify which of the customer's attributes are most critical."""
    factors = []
    try:
        # Load artifacts/feature_importance.csv using pandas
        df = pd.read_csv("artifacts/feature_importance.csv")
        
        # Identify the top 3 most important features globally
        if 'Importance' in df.columns:
            df = df.sort_values(by='Importance', ascending=False)
            
        # Assuming the first column contains the feature names
        top_features = df.iloc[:3, 0].dropna().tolist()
    except Exception as e:
        print(f"[Warning] Could not extract features from CSV: {e}")
        top_features = []

    # Look up specific feature keys in state["customer_data"]
    # Signaling Fix: explicitly denote missing data rather than skipping
    # so the LLM is aware it lacks context for that key factor.
    customer_data = state.get("customer_data", {})
    for feature in top_features:
        value = customer_data.get(feature, "data unavailable")
        factors.append(f"{feature}: {value}")

    return {"key_factors": factors}

def retrieve_context(state: AgentState):
    """Joins key customer risk factors and queries the local ChromaDB for best practices."""
    key_factors = state.get("key_factors", [])
    if not key_factors:
        return {"retrieved_strategies": "No specific risk factors identified."}
        
    query = ", ".join(key_factors)
    
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        # Extract page_content from retrieved documents
        strategies = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"[Warning] Context retrieval failed: {e}")
        strategies = "Could not retrieve specific best practices."
        
    return {"retrieved_strategies": strategies}

def generate_strategy(state: AgentState):
    """Puts together risk factors and Chroma context into the Gemini structured prompt."""
    llm_client = get_llm_client()

    if llm_client is None:
        factors = state.get("key_factors", [])
        return {
            "final_report": {
                "risk_summary": "High churn risk detected by the ML model, but AI narrative generation is unavailable because no Gemini API key is configured.",
                "contributing_factors": factors,
                "recommended_actions": [
                    "Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment or .env file.",
                    "Use high-risk threshold filtering to prioritize outreach until AI recommendations are enabled.",
                    "Review top churn drivers in the dashboard for immediate manual action."
                ],
                "disclaimer": "AI-generated recommendation is currently disabled due to missing API credentials."
            }
        }

    structured_llm = llm_client.with_structured_output(RetentionReport)
    
    customer_data = state.get("customer_data", {})
    key_factors = state.get("key_factors", [])
    strategies = state.get("retrieved_strategies", "")
    
    # Message Role Separation
    messages = [
        SystemMessage(content="You are a senior telecom customer retention expert. Your goal is to analyze customer churn risk and formulate actionable retention strategies based strictly on the provided best practices. Return your response matching the required JSON schema."),
        HumanMessage(content=f"Customer Profile Data: {customer_data}\n\nTop Risk Factors: {key_factors}\n\nRetention Best Practices: {strategies}")
    ]
    
    # Generate the strategy and safely serialize the output version-agnostic
    result = structured_llm.invoke(messages)
    report = result.model_dump() if hasattr(result, "model_dump") else result.dict()
    
    # Overwrite LLM's generated disclaimer with the Pydantic default (V1/V2 safe)
    if hasattr(RetentionReport, "model_fields"):
        report["disclaimer"] = RetentionReport.model_fields["disclaimer"].default
    else:
        report["disclaimer"] = RetentionReport.__fields__["disclaimer"].default
        
    return {"final_report": report}

# 6. Routing Logic
def route_risk(state: AgentState):
    """Ensures we don't spin up heavy generation workflows for safe background customers."""
    if state.get("churn_probability", 0.0) < CHURN_THRESHOLD:
        return END
    return "retrieve_context"

# 7. Compilation
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze_factors", analyze_factors)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("generate_strategy", generate_strategy)

# Set Graph Entry
workflow.set_entry_point("analyze_factors")

# Add edges 
workflow.add_conditional_edges(
    "analyze_factors", 
    route_risk, 
    {
        "retrieve_context": "retrieve_context", 
        END: END
    }
)
workflow.add_edge("retrieve_context", "generate_strategy")
workflow.add_edge("generate_strategy", END)

# Compile the final graph
retention_graph = workflow.compile()

# 8. Entry Function
def get_retention_plan(customer_row: dict, ml_model_path: str) -> dict:
    """Wrapper function to format input payload, compute probability using ML model, and trigger execution."""
    
    # 1. Load ML model using joblib
    model = joblib.load(ml_model_path)
    
    input_df = pd.DataFrame([customer_row])
    
    # 2. Data Alignment Logic (One-Hot Encoding and Shape Matching)
    input_df = pd.get_dummies(input_df)
    
    artifacts_dir = Path(ml_model_path).parent
    feature_names_path = artifacts_dir / "feature_names.json"
    
    try:
        with open(feature_names_path, "r") as f:
            expected_features = json.load(f)
            
        # Reconcile missing features dynamically
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
                
        # Reorder to match model expectations exactly
        input_df = input_df[expected_features]
    except FileNotFoundError:
        print(f"[Warning] Expected feature list not found at {feature_names_path}. Prediction may fail.")
    except Exception as e:
        print(f"[Warning] Failed to align features: {e}")

    # 3. Compute probability using the loaded model and aligned input
    probability = float(model.predict_proba(input_df)[0][1])

    # 4. Apply CHURN_THRESHOLD directly from the globally defined constant
    if probability < CHURN_THRESHOLD:
        return {
            "risk_summary": "Low Risk, no action needed.",
            "contributing_factors": [],
            "recommended_actions": [],
            "disclaimer": "This is an automated low-risk assessment."
        }
        
    initial_state = {
        "customer_data": customer_row, # Pass human-readable raw schema to the LLM
        "churn_probability": probability,
        "key_factors": [],
        "retrieved_strategies": "",
        "final_report": {}
    }
    
    # 5. Invoke graph execution for high risk customers
    final_state = retention_graph.invoke(initial_state)
    return final_state.get("final_report", {})

if __name__ == "__main__":
    # Internal test execution block
    # Provide a realistic, raw customer dictionary
    dummy_row = {
        "MonthlyCharges": 95.50, 
        "tenure": 2, 
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "Dependents": "No",
        "DeviceProtection": "No",
        "PaperlessBilling": "Yes"
    }
    print("Testing get_retention_plan()...")
    
    try:
        result_report = get_retention_plan(dummy_row, "artifacts/model.pkl")
        print("\n--- Final Generated Report ---")
        print(result_report)
    except FileNotFoundError:
        print("Model or feature tracking files not found at artifacts/. Build the ML model first.")
    except Exception as e:
        print(f"Error during execution: {e}")
