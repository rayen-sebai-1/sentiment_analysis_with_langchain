# Statement Analyzer (Groq + LangChain)

A Streamlit app that analyzes multiple text statements with Groq via LangChain. For each statement (one per line), it:

- classifies **sentiment** (positive/negative)
- extracts the **main topic**
- generates a **follow‑up question**

## Features

- Batch analysis for multiple statements
- Clean, interactive Streamlit UI
- Results shown as a table and raw JSON

## Requirements

- Python 3.10+
- A Groq API key

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Groq API key:

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

**macOS/Linux (bash/zsh):**
```bash
export GROQ_API_KEY="your_api_key_here"
```

## Run

```bash
streamlit run app.py
```

Open the local URL printed in your terminal to use the app.

## Usage

- Enter one statement per line.
- Click **Analyze** to get sentiment, main topic, and a follow‑up question.

## Configuration

The app currently uses:

- Model: `qwen/qwen3-32b`
- Temperature: `0.0`
- Max retries: `2`
- Reasoning format: `parsed`

You can adjust these values in the `main()` function inside [app.py](app.py).

## Project Structure

```
.
├── app.py
├── requirements.txt
└── README.md
```

## Notes

If `GROQ_API_KEY` is not set, the app will stop and show an error in the UI.
