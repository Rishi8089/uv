# HealthBot

HealthBot is an AI-powered patient education prototype. It searches reputable medical sources with Tavily, writes a patient-friendly 3–4 paragraph summary, creates one comprehension question, grades the answer, and can start a privacy-cleared new topic session.

## Setup

Create a `config.env` file next to this README (it is ignored by Git):

```env
OPENAI_API_KEY="your OpenAI API key"
TAVILY_API_KEY="your Tavily API key"
OPENAI_MODEL="gpt-4o-mini"
```

Alternatively, HealthBot supports Groq's OpenAI-compatible API:

```env
GROQ_API_KEY="your Groq API key"
TAVILY_API_KEY="your Tavily API key"
GROQ_MODEL="openai/gpt-oss-20b"
```

Install dependencies with `uv sync`, then choose either interface:

```powershell
uv run streamlit run app.py
uv run python healthbot.py
```

The command-line workflow uses `input()` prompts and `print()` output. The Streamlit interface provides the same learning flow in a browser.

HealthBot is educational only. It does not diagnose conditions or provide personalized medical advice.
