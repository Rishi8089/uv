"""Streamlit interface for the HealthBot LangGraph services."""
import streamlit as st
from healthbot import generate_quiz_question, grade_response, search_medical_information, summarize_content

st.set_page_config(page_title="HealthBot", page_icon="🩺", layout="centered")
st.title("🩺 HealthBot")
st.caption("AI-powered patient education")
st.warning("For education only — not a diagnosis or personalized medical advice.")

for key, default in {"topic": "", "results": [], "summary": "", "quiz": "", "answer": "", "feedback": "", "finished": False}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def clear_topic() -> None:
    # Clears prior search and answer content before another topic starts.
    for key, default in {"topic": "", "results": [], "summary": "", "quiz": "", "answer": "", "feedback": "", "finished": False}.items():
        st.session_state[key] = default

topic = st.text_input("What health topic would you like to learn about?", value=st.session_state.topic, placeholder="For example: type 2 diabetes")
if st.button("Search and explain", type="primary"):
    if not topic.strip():
        st.info("Please enter a health topic.")
    else:
        try:
            with st.spinner("Finding reputable medical information…"):
                state = search_medical_information({"patient_topic": topic.strip()})
        except Exception as exc:
            st.error(f"Tavily search could not connect: {exc}")
        else:
            try:
                with st.spinner("Writing a patient-friendly explanation…"):
                    state.update(summarize_content({"patient_topic": topic.strip(), **state}))
            except Exception as exc:
                st.error(f"The AI model could not connect: {exc}")
            else:
                st.session_state.topic = topic.strip()
                st.session_state.results = state["raw_search_results"]
                st.session_state.summary = state["patient_summary"]
                st.session_state.quiz = st.session_state.answer = st.session_state.feedback = ""
                st.rerun()

if st.session_state.summary:
    st.subheader("Health information")
    st.write(st.session_state.summary)
    st.caption("Sources used")
    for source in st.session_state.results:
        title, url = source.get("title", "Source"), source.get("url", "")
        if url:
            st.markdown(f"- [{title}]({url})")
    if not st.session_state.quiz and st.button("I'm ready for a comprehension check", type="primary"):
        try:
            with st.spinner("Creating your question…"):
                st.session_state.quiz = generate_quiz_question({"patient_summary": st.session_state.summary})["quiz_question"]
            st.rerun()
        except Exception as exc:
            st.error(f"The question could not be created: {exc}")

if st.session_state.quiz:
    st.subheader("Comprehension check")
    st.write(st.session_state.quiz)
    answer = st.text_area("Your answer", value=st.session_state.answer, placeholder="Answer in your own words")
    if st.button("Submit answer", type="primary"):
        if not answer.strip():
            st.info("Please enter an answer before submitting.")
        else:
            try:
                with st.spinner("Checking your answer…"):
                    result = grade_response({"patient_summary": st.session_state.summary, "quiz_question": st.session_state.quiz, "patient_answer": answer.strip()})
                st.session_state.answer = answer.strip()
                st.session_state.feedback = result["feedback_explanation"]
                st.rerun()
            except Exception as exc:
                st.error(f"The answer could not be graded: {exc}")

if st.session_state.feedback:
    st.subheader("Quiz result")
    st.write(st.session_state.feedback)
    st.write("Would you like to learn about another topic?")
    left, right = st.columns(2)
    if left.button("Yes, another topic"):
        clear_topic(); st.rerun()
    if right.button("No, finish"):
        st.session_state.finished = True
if st.session_state.finished:
    st.success("Thank you for using HealthBot.")
