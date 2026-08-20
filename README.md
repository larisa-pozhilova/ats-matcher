# Agentic ATS Matcher

An enterprise-grade, asynchronous AI evaluation engine designed to autonomously parse, evaluate, and critique candidate resumes against complex Job Descriptions. 

Built with **Python, FastAPI, and LangGraph**, this microservice utilizes a multi-step agentic state machine to process non-deterministic LLM outputs into strictly typed JSON contracts, preventing hallucinations and ensuring evaluation reliability.

## 🏗 System Architecture

The system utilizes a directed acyclic graph (DAG) to manage the evaluation lifecycle, incorporating a dedicated Human-in-the-Loop/Critique node to enforce output quality before API response.

```mermaid
graph TD
    START((API Request)) --> Extractor[Extractor Node]
    Extractor --> Evaluator[Evaluator Node]
    Evaluator --> Critique[Critique Node]
    Critique --> Condition{Hallucination Detected?}
    Condition -- Yes (Inject Feedback) --> Evaluator
    Condition -- No (Passed) --> END((API Response))
