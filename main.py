import json
from typing import List, Optional, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------
# 1. API CONTRACTS (Pydantic Schemas)
# ---------------------------------------------------------
class EvaluationRequest(BaseModel):
    job_description: str = Field(..., description="Raw text of the target job description.")
    resume_text: str = Field(..., description="Raw text of the candidate's resume.")

class GapAnalysis(BaseModel):
    category: str
    missing_skill: str
    severity: str
    reasoning: str

class StrengthAnalysis(BaseModel):
    category: str
    matched_skill: str
    reasoning: str

class EvaluationResponse(BaseModel):
    candidate_name: str
    overall_match_score: int = Field(ge=0, le=100)
    is_viable_candidate: bool
    strengths: List[StrengthAnalysis] = []
    gaps: List[GapAnalysis] = []
    agentic_summary: str

# ---------------------------------------------------------
# 2. LANGGRAPH STATE DEFINITION
# ---------------------------------------------------------
class AgenticATSState(TypedDict):
    job_description: str
    resume_text: str
    extracted_candidate_data: Optional[dict]
    draft_evaluation: Optional[dict]
    final_evaluation: Optional[EvaluationResponse]
    critique_feedback: Optional[str]
    revision_count: int

# ---------------------------------------------------------
# 3. LOCAL LLM INITIALIZATION
# ---------------------------------------------------------
# Using Qwen 2.5 32b via local Ollama. format="json" forces valid JSON output.
llm = ChatOllama(
    model="qwen2.5:32b", 
    base_url="http://localhost:11434", 
    temperature=0.0, 
    format="json"
)

# ---------------------------------------------------------
# 4. LANGGRAPH NODES (The Agents)
# ---------------------------------------------------------
async def extractor_node(state: AgenticATSState):
    """Parses the unstructured resume into structured JSON."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert ATS data extractor. Extract the candidate's name, core skills, and work experience into strict JSON format. Return ONLY JSON."),
        ("human", "{resume_text}")
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({"resume_text": state["resume_text"]})
    
    try:
        extracted_data = json.loads(response.content)
    except json.JSONDecodeError:
        extracted_data = {"candidate_name": "Unknown", "error": "Parsing failed"}
        
    return {"extracted_candidate_data": extracted_data}

async def evaluator_node(state: AgenticATSState):
    """Compares the structured candidate data against the JD."""
    feedback_context = f"\nCRITIQUE FEEDBACK TO FIX: {state['critique_feedback']}" if state.get("critique_feedback") else ""
    
    # 1. Plain string (no f-string injection)
    schema_example = """
    {
        "candidate_name": "string",
        "overall_match_score": 0,
        "is_viable_candidate": true,
        "strengths": [{"category": "string", "matched_skill": "string", "reasoning": "string"}],
        "gaps": [{"category": "string", "missing_skill": "string", "severity": "Critical/Moderate/Low", "reasoning": "string"}],
        "agentic_summary": "string"
    }
    """
    
    # 2. Removed the f"" prefix. Using LangChain's native variable injection.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an elite ATS Evaluator. Output ONLY a flat JSON object matching this exact schema:\n{schema}\nDo NOT wrap the output in an 'evaluation' root key. Do not include markdown.{feedback}"),
        ("human", "CANDIDATE DATA: {candidate_data}\n\nJOB DESCRIPTION: {job_description}")
    ])
    
    chain = prompt | llm
    
    # 3. Pass all variables safely through ainvoke
    response = await chain.ainvoke({
        "schema": schema_example,
        "feedback": feedback_context,
        "candidate_data": json.dumps(state["extracted_candidate_data"]),
        "job_description": state["job_description"]
    })
    
    try:
        draft_eval = json.loads(response.content)
        
        # 4. Defensive Unwrapping
        if "evaluation" in draft_eval and isinstance(draft_eval["evaluation"], dict):
            draft_eval = draft_eval["evaluation"]
            
        if "name" in draft_eval and "candidate_name" not in draft_eval:
            draft_eval["candidate_name"] = draft_eval.pop("name")
            
    except json.JSONDecodeError:
        draft_eval = {}
        
    return {"draft_evaluation": draft_eval, "revision_count": state.get("revision_count", 0) + 1}

async def critique_node(state: AgenticATSState):
    """Audits the evaluation for hallucinations against the original extracted data."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Quality Governance Agent. Audit the draft evaluation. Did it hallucinate skills not present in the candidate data? Return strictly JSON with 'passed': boolean and 'feedback': string."),
        ("human", "CANDIDATE DATA: {candidate_data}\n\nDRAFT EVALUATION: {draft_evaluation}")
    ])
    chain = prompt | llm
    response = await chain.ainvoke({
        "candidate_data": json.dumps(state["extracted_candidate_data"]),
        "draft_evaluation": json.dumps(state["draft_evaluation"])
    })
    
    try:
        critique = json.loads(response.content)
    except json.JSONDecodeError:
        critique = {"passed": True, "feedback": ""}
        
    # Route logic based on critique
    if critique.get("passed") or state["revision_count"] >= 3:
        # Pass validation, or hit retry cap (prevent infinite compute loop)
        return {
            "final_evaluation": EvaluationResponse(**state["draft_evaluation"]), 
            "critique_feedback": None
        }
    else:
        # Fail validation, generate feedback to send back to evaluator
        return {"critique_feedback": critique.get("feedback")}

def route_evaluation(state: AgenticATSState):
    """Determines whether to end the graph or loop back to the evaluator."""
    if state.get("final_evaluation") is not None:
        return END
    return "evaluator_node"

# ---------------------------------------------------------
# 5. ASSEMBLE THE GRAPH
# ---------------------------------------------------------
workflow = StateGraph(AgenticATSState)

workflow.add_node("extractor_node", extractor_node)
workflow.add_node("evaluator_node", evaluator_node)
workflow.add_node("critique_node", critique_node)

workflow.set_entry_point("extractor_node")
workflow.add_edge("extractor_node", "evaluator_node")
workflow.add_edge("evaluator_node", "critique_node")
# The critique node decides: return to evaluator or end
workflow.add_conditional_edges("critique_node", route_evaluation)

app_graph = workflow.compile()

# ---------------------------------------------------------
# 6. FASTAPI APPLICATION
# ---------------------------------------------------------
app = FastAPI(
    title="Agentic ATS Matcher",
    description="Asynchronous LangGraph API for Resume vs JD Evaluation"
)

@app.post("/evaluate-candidate", response_model=EvaluationResponse)
async def evaluate_candidate(request: EvaluationRequest):
    initial_state = {
        "job_description": request.job_description,
        "resume_text": request.resume_text,
        "revision_count": 0
    }
    
    try:
        # Trigger the LangGraph workflow asynchronously
        final_state = await app_graph.ainvoke(initial_state)
        
        if not final_state.get("final_evaluation"):
            raise HTTPException(status_code=500, detail="State machine failed to produce a final evaluation.")
            
        return final_state["final_evaluation"]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
