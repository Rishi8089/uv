"""HealthBot LangGraph workflow and command-line learning session."""
from __future__ import annotations
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

load_dotenv("config.env")

class HealthBotState(TypedDict, total=False):
    patient_topic: str
    raw_search_results: list[dict]
    patient_summary: str
    quiz_question: str
    patient_answer: str
    quiz_grade: str
    feedback_explanation: str
    continue_session: bool

def _require_environment() -> None:
    missing = []
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")):
        missing.append("OPENAI_API_KEY or GROQ_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        raise RuntimeError("Missing " + ", ".join(missing) + ". Add it to config.env before running HealthBot.")

def get_llm() -> ChatOpenAI | ChatGroq:
    _require_environment()
    # Groq offers an OpenAI-compatible endpoint. Supporting it here lets this
    # project use either a standard OpenAI key or the existing Groq setup.
    if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        model = os.getenv("GROQ_MODEL")
        if not model:
            # OPENAI_MODEL may have been left at the OpenAI example value.
            # That model is not available on Groq.
            configured = os.getenv("OPENAI_MODEL", "")
            model = configured if configured and not configured.startswith("gpt-") else "openai/gpt-oss-20b"
        return ChatGroq(
            model=model,
            temperature=0,
        )
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

def get_tavily() -> TavilySearch:
    _require_environment()
    return TavilySearch(max_results=4, search_depth="advanced")

def ask_topic(_: HealthBotState) -> HealthBotState:
    return {"patient_topic": input("What health topic would you like to learn about? ").strip()}

def search_medical_information(state: HealthBotState) -> HealthBotState:
    query = f"{state['patient_topic']} patient education site:cdc.gov OR site:nih.gov OR site:medlineplus.gov OR site:who.int"
    response = get_tavily().invoke(query)
    results = response.get("results", []) if isinstance(response, dict) else []
    if not results:
        raise RuntimeError("No reputable health sources were returned. Please try a more specific topic.")
    return {"raw_search_results": results}

def _source_text(results: list[dict]) -> str:
    return "\n\n".join(f"SOURCE: {item.get('title', 'Untitled')}\nURL: {item.get('url', '')}\nCONTENT: {item.get('content', '')}" for item in results)

def summarize_content(state: HealthBotState) -> HealthBotState:
    prompt = f"""You are a patient education assistant. Write exactly 3 or 4 short, patient-friendly paragraphs about {state['patient_topic']} using ONLY the search material below. Do not add facts from your own knowledge, diagnose, or give personalized medical advice. Add inline source citations using the source titles in square brackets whenever you state a medical fact. Return only the summary.

SEARCH MATERIAL:
{_source_text(state['raw_search_results'])}"""
    return {"patient_summary": get_llm().invoke(prompt).content}

def present_summary(state: HealthBotState) -> HealthBotState:
    print("\nHEALTH INFORMATION\n" + state["patient_summary"])
    input("\nPress Enter when you are ready for a comprehension check.")
    return {}

def generate_quiz_question(state: HealthBotState) -> HealthBotState:
    prompt = f"""Create exactly one short, open-ended comprehension question that can be answered using ONLY the health summary below. Do not include the answer, options, commentary, or outside facts. Return only the question.

HEALTH SUMMARY:
{state['patient_summary']}"""
    return {"quiz_question": get_llm().invoke(prompt).content.strip()}

def ask_quiz_answer(state: HealthBotState) -> HealthBotState:
    print("\nCOMPREHENSION CHECK\n" + state["quiz_question"])
    return {"patient_answer": input("Your answer: ").strip()}

def grade_response(state: HealthBotState) -> HealthBotState:
    prompt = f"""Grade the student's response using ONLY the summary below. Give one letter grade (A, B, C, or D) and a brief explanation. The explanation must quote or refer to the relevant inline citation(s) already in the summary; do not invent citations or use outside information. Use this exact format:
Grade: <letter>
Feedback: <explanation>

HEALTH SUMMARY:
{state['patient_summary']}

QUESTION:
{state['quiz_question']}

STUDENT ANSWER:
{state['patient_answer']}"""
    feedback = get_llm().invoke(prompt).content.strip()
    grade = feedback.splitlines()[0].replace("Grade:", "").strip()
    return {"quiz_grade": grade, "feedback_explanation": feedback}

def present_results(state: HealthBotState) -> HealthBotState:
    print("\nQUIZ RESULT\n" + state["feedback_explanation"])
    return {}

def ask_another_topic(_: HealthBotState) -> HealthBotState:
    answer = input("\nWould you like to learn about another topic? (yes/no): ").strip().lower()
    return {"continue_session": answer in {"y", "yes"}}

def reset_topic(_: HealthBotState) -> HealthBotState:
    return {"patient_topic": "", "raw_search_results": [], "patient_summary": "", "quiz_question": "", "patient_answer": "", "quiz_grade": "", "feedback_explanation": "", "continue_session": True}

def route_session(state: HealthBotState) -> str:
    return "new_topic" if state.get("continue_session") else "end"

def build_healthbot():
    graph = StateGraph(HealthBotState)
    nodes = {"ask_topic": ask_topic, "search_medical_information": search_medical_information, "summarize_content": summarize_content, "present_summary": present_summary, "generate_quiz_question": generate_quiz_question, "ask_quiz_answer": ask_quiz_answer, "grade_response": grade_response, "present_results": present_results, "ask_another_topic": ask_another_topic, "reset_topic": reset_topic}
    for name, node in nodes.items(): graph.add_node(name, node)
    graph.add_edge(START, "ask_topic")
    for source, target in [("ask_topic", "search_medical_information"), ("search_medical_information", "summarize_content"), ("summarize_content", "present_summary"), ("present_summary", "generate_quiz_question"), ("generate_quiz_question", "ask_quiz_answer"), ("ask_quiz_answer", "grade_response"), ("grade_response", "present_results"), ("present_results", "ask_another_topic"), ("reset_topic", "ask_topic")]: graph.add_edge(source, target)
    graph.add_conditional_edges("ask_another_topic", route_session, {"new_topic": "reset_topic", "end": END})
    return graph.compile()

healthbot = build_healthbot()

def main() -> None:
    print("HealthBot — educational information only; not medical advice.")
    # A session may contain several topics, so allow more than LangGraph's
    # short default traversal limit while retaining a finite safety cap.
    healthbot.invoke({}, {"recursion_limit": 1_000})

if __name__ == "__main__": main()
