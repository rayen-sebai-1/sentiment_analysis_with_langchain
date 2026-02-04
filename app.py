import os
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda


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
# 2) CORE LOGIC (RUNNABLES)
# =========================================================

def build_analyzer(llm: ChatGroq):
    """
    Returns a Runnable that:
    - takes {"text": "..."}
    - runs sentiment/topic/followup in parallel
    - returns {"sentiment": "...", "main_topic": "...", "followup_question": "..."}
    """
    parser = StrOutputParser()

    sentiment_chain = sentiment_template | llm | parser
    main_topic_chain = main_topic_template | llm | parser
    followup_chain = followup_template | llm | parser

    parallel = RunnableParallel(
        sentiment=sentiment_chain,
        main_topic=main_topic_chain,
        followup_question=followup_chain,
    )

    # Small post-processing step to normalize whitespace
    def _clean(d: Dict[str, str]) -> Dict[str, str]:
        return {k: (v or "").strip() for k, v in d.items()}

    return parallel | RunnableLambda(_clean)


def analyze_statements(llm: ChatGroq, statements: List[str]) -> List[Dict[str, Any]]:
    analyzer = build_analyzer(llm)

    # batch inputs -> list[dict]
    inputs = [{"text": s} for s in statements]
    outputs = analyzer.batch(inputs)

    # Attach original statement to each row
    results: List[Dict[str, Any]] = []
    for statement, out in zip(statements, outputs):
        results.append(
            {
                "statement": statement,
                "sentiment": out.get("sentiment", ""),
                "main_topic": out.get("main_topic", ""),
                "followup_question": out.get("followup_question", ""),
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
