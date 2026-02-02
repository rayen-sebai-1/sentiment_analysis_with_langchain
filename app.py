import os
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 0) PROMPT TEMPLATES
# =========================================================

sentiment_template = ChatPromptTemplate.from_template(
    """In a single word, either 'positive' or 'negative',
provide the overall sentiment of the following piece of text: {text}"""
)

main_topic_template = ChatPromptTemplate.from_template(
    """In a short phrase, provide the main topic of the following piece of text: {text}"""
)

followup_template = ChatPromptTemplate.from_template(
    """Provide an interesting followup question about the following piece of text: {text}"""
)


# =========================================================
# 1) LLM SETUP
# =========================================================

@st.cache_resource
def get_llm(
    model: str,
    temperature: float,
    max_retries: int,
    reasoning_format: str | None,
) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "Missing GROQ_API_KEY environment variable. "
            "Set it in your terminal before running Streamlit."
        )
        st.stop()


    llm = ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=None,
        reasoning_format=reasoning_format,
        timeout=None,
        max_retries=max_retries,
    )
    return llm


# =========================================================
# 2) CORE LOGIC
# =========================================================

def analyze_statements(llm: ChatGroq, statements: List[str]) -> List[Dict[str, Any]]:
    # Build prompts
    sentiment_prompts = [sentiment_template.format_messages(text=s) for s in statements]
    main_topic_prompts = [main_topic_template.format_messages(text=s) for s in statements]
    followup_prompts = [followup_template.format_messages(text=s) for s in statements]

    # Batch calls
    sentiments = llm.batch(sentiment_prompts)
    main_topics = llm.batch(main_topic_prompts)
    followups = llm.batch(followup_prompts)

    # Return structured results
    results: List[Dict[str, Any]] = []
    for statement, sentiment, main_topic, followup in zip(statements, sentiments, main_topics, followups):
        results.append(
            {
                "statement": statement,
                "sentiment": (sentiment.content or "").strip(),
                "main_topic": (main_topic.content or "").strip(),
                "followup_question": (followup.content or "").strip(),
            }
        )
    return results


# =========================================================
# 3) STREAMLIT UI
# =========================================================

def main():
    st.set_page_config(page_title="Statement Analyzer (Groq)", layout="wide")
    st.title("Statement Analyzer – Groq + LangChain ")
    st.caption("Enter one statement per line.")

    # Model configuration
    model = "qwen/qwen3-32b"
    temperature = 0.0
    max_retries = 2
    reasoning_format = "parsed"

    st.markdown(
        """
Enter one statement per line.  
For each statement, the app will:
- classify **sentiment** (positive/negative)
- extract **main topic**
- generate a **follow-up question**
"""
    )

    sample_text = (
        "I loved the new update, it made everything faster!\n"
        "The app keeps crashing and I'm frustrated.\n"
        "Customer support was quick and helpful.\n"
        "This feature is confusing and poorly documented."
    )

    statements_input = st.text_area(
        "Statements (one per line):",
        value=sample_text,
        height=200,
    )

    col_left, col_right = st.columns([1, 3])
    with col_left:
        run_button = st.button("Analyze", type="primary")

    if not run_button:
        return

    raw_lines = [line.strip() for line in statements_input.splitlines()]
    statements: List[str] = [line for line in raw_lines if line]

    if not statements:
        st.warning("No non-empty lines found. Add at least one statement.")
        return

    llm = get_llm(
        model=model,
        temperature=temperature,
        max_retries=int(max_retries),
        reasoning_format=reasoning_format,
    )

    with st.spinner("Analyzing statements with Groq..."):
        results = analyze_statements(llm, statements)

    df = pd.DataFrame(
        results,
        columns=["statement", "sentiment", "main_topic", "followup_question"],
    )

    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    st.subheader("Raw JSON (for debugging / integration)")
    st.json(results)


if __name__ == "__main__":
    main()
