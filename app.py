#!/usr/bin/env python3
"""
Synthetic Persona Reactor - Streamlit Web App
Test ideas, run surveys and do A/B testing against investor and advisor personas.
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import re
import random
import os
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from difflib import SequenceMatcher
from datetime import datetime
import io
import base64
from io import BytesIO

# Optional: wordcloud for visualization (graceful fallback if not installed)
try:
    from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False
    WC_STOPWORDS = set()

from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================

# API keys loaded from .streamlit/secrets.toml (local) or Streamlit Cloud secrets
API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_API_KEY = API_KEY  # alias for clarity
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

MODEL_NAME = "gemini-3.1-flash-lite-preview"  # default for utility calls (news summarization, etc.)

# Model catalog: 2 models per provider (low cost + higher cost) for persona testing.
# Costs are USD per 1M tokens (input / output) and are best estimates - verify with provider pricing pages.
MODEL_CONFIGS = {
    "gemini-3.1-flash-lite-preview": {
        "provider": "google",
        "label": "Gemini 3.1 Flash-Lite",
        "tier": "Low cost",
        "description": "Google's fastest model. Cheap and quick. Good baseline for high-volume persona testing.",
        "input_cost_per_1m": 0.075,
        "output_cost_per_1m": 0.30,
    },
    "gemini-3-pro-preview": {
        "provider": "google",
        "label": "Gemini 3 Pro",
        "tier": "Higher cost",
        "description": "Google's flagship reasoning model. More nuanced persona responses, better at complex reasoning.",
        "input_cost_per_1m": 1.25,
        "output_cost_per_1m": 5.00,
    },
    "gpt-5-mini": {
        "provider": "openai",
        "label": "GPT-5 Mini",
        "tier": "Low cost",
        "description": "OpenAI's compact model. Fast and capable for persona role-play.",
        "input_cost_per_1m": 0.25,
        "output_cost_per_1m": 2.00,
    },
    "gpt-5": {
        "provider": "openai",
        "label": "GPT-5",
        "tier": "Higher cost",
        "description": "OpenAI's flagship model. Best for complex persona reasoning and detailed responses.",
        "input_cost_per_1m": 1.25,
        "output_cost_per_1m": 10.00,
    },
    "claude-haiku-4-6": {
        "provider": "anthropic",
        "label": "Claude Haiku 4.6",
        "tier": "Low cost",
        "description": "Anthropic's fast model. Strong at nuanced persona role-play with tight, on-character responses.",
        "input_cost_per_1m": 1.00,
        "output_cost_per_1m": 5.00,
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "label": "Claude Sonnet 4.6",
        "tier": "Higher cost",
        "description": "Anthropic's flagship model. Most thoughtful, in-depth persona responses.",
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
    },
}

DEFAULT_MODEL_ID = "gemini-3.1-flash-lite-preview"


def estimate_run_cost(model_id, n_personas, prompt_type="reaction"):
    """Rough cost estimate for a run. Returns (cost_usd, total_tokens_est)."""
    cfg = MODEL_CONFIGS.get(model_id, MODEL_CONFIGS[DEFAULT_MODEL_ID])
    # Token estimates per persona call
    if prompt_type == "survey":
        in_tokens = 1500
        out_tokens = 800
    else:
        in_tokens = 1100
        out_tokens = 500
    total_in = in_tokens * n_personas
    total_out = out_tokens * n_personas
    cost = (total_in * cfg["input_cost_per_1m"] / 1_000_000
            + total_out * cfg["output_cost_per_1m"] / 1_000_000)
    return cost, total_in + total_out

MAX_WORKERS = 10
PERSONAS_FILE = os.path.join(os.path.dirname(__file__), "personas.json")
ADVISOR_PERSONAS_FILE = os.path.join(os.path.dirname(__file__), "advisor_personas.json")
RBC_DS_PERSONAS_FILE = os.path.join(os.path.dirname(__file__), "rbc_ds_personas.json")
RBC_PIM_PERSONAS_FILE = os.path.join(os.path.dirname(__file__), "rbc_pim_personas.json")
USAGE_LOG_FILE = os.path.join(os.path.dirname(__file__), "usage_log.json")
ADMIN_SECRET = st.secrets.get("ADMIN_KEY", "persona-reactor-admin-2024")
COST_PER_API_CALL = 0.0001  # Estimated cost per Gemini Flash-Lite call (USD)

REACTION_PROMPT = """You are roleplaying as this Canadian investor persona. Stay fully in character. Think and respond as this specific person would, given their background, financial situation, knowledge level, and life circumstances.

PERSONA:
{persona}
{context}
React to the following investment idea, product, or concept. Consider your financial situation, investment goals, risk tolerance, knowledge level, and life stage.

IDEA:
{idea}

Respond ONLY with valid JSON in this exact format (no markdown, no code fences):
{{
  "interest_score": <integer 1-10>,
  "sentiment": "<positive|neutral|negative|mixed>",
  "gut_reaction": "<1-2 sentence first-person initial reaction>",
  "key_concerns": ["<concern1>", "<concern2>"],
  "appeal_factors": ["<what appeals to you about this>"],
  "would_invest": <true or false>,
  "investment_amount": "<none|small ($100-$1K)|moderate ($1K-$10K)|significant ($10K-$50K)|major ($50K+)>",
  "what_would_help": "<what would make you more likely to invest>",
  "verbatim_quote": "<2-3 sentences as if talking to a friend about this>"
}}"""

SURVEY_PROMPT = """You are roleplaying as this Canadian persona. Stay fully in character. Think and respond as this specific person would, given their background, demographics, financial situation, values, and life circumstances.

PERSONA:
{persona}
{context}
You are being surveyed. Answer each question honestly and naturally from this persona's perspective. Consider your age, income, education, family situation, values, location, and life experiences.

For multiple-choice questions:
- Read ALL options carefully before deciding.
- Consider how each option relates to your specific background, values, and circumstances.
- Different people with different backgrounds would genuinely choose different options.
- You MUST select exactly ONE of the provided options. Do NOT create your own answer, say "other", "none of the above", or refuse to choose.
- Even if no option is a perfect fit, pick the CLOSEST one to your perspective.
- Set "selected_option" to the EXACT text of your chosen option, copied verbatim from the list.
- In your "answer" field, briefly explain why this option fits you and why another option did not.

For open-ended questions, set "selected_option" to null and provide your full answer in the "answer" field.

QUESTIONS:
{questions_formatted}

Respond ONLY with valid JSON in this exact format (no markdown, no code fences):
{{
  "answers": [
    {{
      "question_number": 1,
      "question_type": "<open_ended|multiple_choice>",
      "selected_option": "<chosen option text for MC, or null for open-ended>",
      "answer": "<your natural, in-character answer/explanation in 1-3 sentences>",
      "sentiment": "<positive|neutral|negative|mixed>",
      "confidence": "<low|medium|high>",
      "key_themes": ["<theme1>", "<theme2>"]
    }}
  ]
}}

Include one entry in the "answers" array for each question, in order."""


# ---------- ADVISOR-SPECIFIC PROMPTS ----------

ADVISOR_REACTION_PROMPT = """You are roleplaying as this Canadian financial advisor persona. Stay fully in character. Think and respond as this specific advisor would, given their professional background, designations, client base, practice focus, and years of experience.

PERSONA:
{persona}
{context}
React to the following investment idea, product, or concept FROM A PROFESSIONAL PERSPECTIVE. Consider whether you would recommend this to your clients, how it fits various client profiles, regulatory considerations, and practical implementation in your practice.

IDEA:
{idea}

Respond ONLY with valid JSON in this exact format (no markdown, no code fences):
{{
  "interest_score": <integer 1-10>,
  "sentiment": "<positive|neutral|negative|mixed>",
  "gut_reaction": "<1-2 sentence first-person professional reaction>",
  "key_concerns": ["<concern1>", "<concern2>"],
  "appeal_factors": ["<what appeals to you professionally about this>"],
  "would_recommend": <true or false>,
  "client_suitability": "<unsuitable|niche|moderate|broad>",
  "what_would_help": "<what would make you more likely to recommend this to clients>",
  "verbatim_quote": "<2-3 sentences as if talking to a colleague about this>"
}}"""

# ---------- GENERAL IDEA PROMPTS ----------

GENERAL_IDEA_PROMPT = """You are roleplaying as this Canadian persona. Stay fully in character. Think and respond as this specific person would, given their background, financial situation, knowledge level, and life circumstances.

PERSONA:
{persona}
{context}
React to the following idea, concept, or proposal. Consider your personal situation, needs, values, preferences, and daily life. This is NOT specifically about investing — evaluate whether this idea appeals to you personally.

IDEA:
{idea}

Respond ONLY with valid JSON in this exact format (no markdown, no code fences):
{{
  "interest_score": <integer 1-10>,
  "sentiment": "<positive|neutral|negative|mixed>",
  "gut_reaction": "<1-2 sentence first-person initial reaction>",
  "key_concerns": ["<concern1>", "<concern2>"],
  "appeal_factors": ["<what appeals to you about this>"],
  "would_invest": <true or false>,
  "investment_amount": "<none|small ($100-$1K)|moderate ($1K-$10K)|significant ($10K-$50K)|major ($50K+)>",
  "what_would_help": "<what would make you more interested>",
  "verbatim_quote": "<2-3 sentences as if talking to a friend about this>"
}}"""

ADVISOR_GENERAL_IDEA_PROMPT = """You are roleplaying as this Canadian financial advisor persona. Stay fully in character. Think and respond as this specific advisor would, given their professional background, designations, client base, practice focus, and years of experience.

PERSONA:
{persona}
{context}
React to the following idea, concept, or proposal FROM A PROFESSIONAL PERSPECTIVE. Consider how this relates to your practice, your clients, and the broader industry. This is NOT specifically about a financial product — evaluate this idea on its merits.

IDEA:
{idea}

Respond ONLY with valid JSON in this exact format (no markdown, no code fences):
{{
  "interest_score": <integer 1-10>,
  "sentiment": "<positive|neutral|negative|mixed>",
  "gut_reaction": "<1-2 sentence first-person professional reaction>",
  "key_concerns": ["<concern1>", "<concern2>"],
  "appeal_factors": ["<what appeals to you professionally about this>"],
  "would_recommend": <true or false>,
  "client_suitability": "<unsuitable|niche|moderate|broad>",
  "what_would_help": "<what would make you more likely to support this>",
  "verbatim_quote": "<2-3 sentences as if talking to a colleague about this>"
}}"""


ADVISOR_SURVEY_PROMPT = """You are roleplaying as this Canadian financial advisor persona. Stay fully in character. Think and respond as this specific advisor would, given their professional background, designations, client base, practice focus, and experience.

PERSONA:
{persona}
{context}
You are being surveyed about your professional opinions and practices. Answer each question honestly and naturally from this advisor's professional perspective. Consider your experience, client base, firm type, designations, and practice focus.

For multiple-choice questions:
- Read ALL options carefully before deciding.
- Consider how each option relates to your specific background, values, and circumstances.
- Different people with different backgrounds would genuinely choose different options.
- You MUST select exactly ONE of the provided options. Do NOT create your own answer, say "other", "none of the above", or refuse to choose.
- Even if no option is a perfect fit, pick the CLOSEST one to your perspective.
- Set "selected_option" to the EXACT text of your chosen option, copied verbatim from the list.
- In your "answer" field, briefly explain why this option fits you and why another option did not.

For open-ended questions, set "selected_option" to null and provide your full answer in the "answer" field.

QUESTIONS:
{questions_formatted}

Respond ONLY with valid JSON in this exact format (no markdown, no code fences):
{{
  "answers": [
    {{
      "question_number": 1,
      "question_type": "<open_ended|multiple_choice>",
      "selected_option": "<chosen option text for MC, or null for open-ended>",
      "answer": "<your natural, in-character answer/explanation in 1-3 sentences>",
      "sentiment": "<positive|neutral|negative|mixed>",
      "confidence": "<low|medium|high>",
      "key_themes": ["<theme1>", "<theme2>"]
    }}
  ]
}}

Include one entry in the "answers" array for each question, in order."""

# Regex: line ending with [opt1, opt2, ...] (at least 2 comma-separated items)
MC_PATTERN = re.compile(r'^(.+?)\s*\[([^,\]]+(?:,\s*[^,\]]+)+)\]\s*$')


def parse_questions(raw_text):
    """Parse text area input into structured question dicts.

    Lines ending with [option1, option2, ...] (2+ comma-separated items)
    are multiple-choice. All other non-empty lines are open-ended.
    """
    questions = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = MC_PATTERN.match(line)
        if match:
            text = match.group(1).strip()
            options = [opt.strip() for opt in match.group(2).split(",")]
            questions.append({"type": "mc", "text": text, "options": options})
        else:
            questions.append({"type": "open", "text": line})
    return questions


def _fuzzy_match_option(selected, options):
    """Match a selected_option string to the closest valid option.
    Returns the best matching option text, or the first option as fallback."""
    if not selected or not options:
        return options[0] if options else None
    sel = str(selected).lower().strip()
    # Try exact match first
    for opt in options:
        if opt.lower().strip() == sel:
            return opt
    # Try substring / contains match
    for opt in options:
        if sel in opt.lower() or opt.lower() in sel:
            return opt
    # Fuzzy match using SequenceMatcher
    best_score = 0
    best_opt = options[0]
    for opt in options:
        score = SequenceMatcher(None, sel, opt.lower().strip()).ratio()
        if score > best_score:
            best_score = score
            best_opt = opt
    return best_opt


def format_questions_for_prompt(questions, seed=None):
    """Format structured questions for the LLM prompt.

    If seed is provided, MC option order is shuffled per-persona to reduce
    position bias in responses.
    """
    rng = random.Random(seed) if seed is not None else None
    lines = []
    for i, q in enumerate(questions, 1):
        if q["type"] == "mc":
            options = list(q["options"])
            if rng:
                rng.shuffle(options)
            option_lines = "\n".join(
                f"  {chr(97+j)}) {opt}" for j, opt in enumerate(options)
            )
            lines.append(
                f"Q{i}. [MULTIPLE CHOICE - select exactly one]\n"
                f"  {q['text']}\n"
                f"  Options:\n{option_lines}"
            )
        else:
            lines.append(f"Q{i}. [OPEN-ENDED]\n  {q['text']}")
    return "\n\n".join(lines)



def render_question_builder(panel_key="consumer"):
    """Render a structured question builder UI. Returns list of question dicts."""
    builder_key = f"{panel_key}_question_builder"
    counter_key = f"{panel_key}_q_counter"

    if builder_key not in st.session_state:
        st.session_state[builder_key] = []
    if counter_key not in st.session_state:
        st.session_state[counter_key] = 0

    questions = st.session_state[builder_key]

    # --- Add Question Buttons ---
    col_add1, col_add2 = st.columns(2)
    with col_add1:
        if st.button("\u2795 Open-Ended Question", key=f"{panel_key}_add_open",
                      use_container_width=True):
            st.session_state[counter_key] += 1
            questions.append({"id": st.session_state[counter_key],
                              "type": "open", "text": ""})
            st.rerun()
    with col_add2:
        if st.button("\u2795 Multiple-Choice Question", key=f"{panel_key}_add_mc",
                      use_container_width=True):
            st.session_state[counter_key] += 1
            questions.append({"id": st.session_state[counter_key],
                              "type": "mc", "text": "", "options": ["", ""]})
            st.rerun()

    if not questions:
        st.info("Add questions using the buttons above.")

    # --- Render Each Question ---
    to_remove = None
    for idx, q in enumerate(questions):
        qid = q["id"]
        with st.container(border=True):
            header_col, remove_col = st.columns([6, 1])
            with header_col:
                q_type_label = "Multiple Choice" if q["type"] == "mc" else "Open-Ended"
                st.markdown(f"**Q{idx + 1}** \u2014 {q_type_label}")
            with remove_col:
                if st.button("\u2716", key=f"{panel_key}_rm_{qid}",
                             help="Remove this question"):
                    to_remove = idx

            q["text"] = st.text_input(
                "Question",
                value=q.get("text", ""),
                key=f"{panel_key}_qt_{qid}",
                label_visibility="collapsed",
                placeholder="Enter your question here...",
            )

            if q["type"] == "mc":
                opt_to_remove = None
                for opt_idx, opt in enumerate(q["options"]):
                    opt_col, rm_col = st.columns([8, 1])
                    with opt_col:
                        q["options"][opt_idx] = st.text_input(
                            f"Option {chr(65 + opt_idx)}",
                            value=opt,
                            key=f"{panel_key}_qo_{qid}_{opt_idx}",
                            placeholder=f"Option {chr(65 + opt_idx)}",
                        )
                    with rm_col:
                        if len(q["options"]) > 2:
                            if st.button("\u2716", key=f"{panel_key}_ro_{qid}_{opt_idx}",
                                         help="Remove option"):
                                opt_to_remove = opt_idx

                if opt_to_remove is not None:
                    q["options"].pop(opt_to_remove)
                    st.rerun()

                if len(q["options"]) < 8:
                    if st.button("+ Add Option", key=f"{panel_key}_ao_{qid}",
                                 type="secondary"):
                        q["options"].append("")
                        st.rerun()

    if to_remove is not None:
        questions.pop(to_remove)
        st.rerun()

    # --- Bottom Add Buttons (visible after scrolling through questions) ---
    if questions:
        st.divider()
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("➕ Open-Ended Question", key=f"{panel_key}_add_open_btm",
                          use_container_width=True):
                st.session_state[counter_key] += 1
                questions.append({"id": st.session_state[counter_key],
                                  "type": "open", "text": ""})
                st.rerun()
        with bcol2:
            if st.button("➕ Multiple-Choice Question", key=f"{panel_key}_add_mc_btm",
                          use_container_width=True):
                st.session_state[counter_key] += 1
                questions.append({"id": st.session_state[counter_key],
                                  "type": "mc", "text": "", "options": ["", ""]})
                st.rerun()

    # --- Build Validated Output ---
    valid_questions = []
    for q in questions:
        if not q.get("text", "").strip():
            continue
        if q["type"] == "mc":
            valid_opts = [o.strip() for o in q.get("options", []) if o.strip()]
            if len(valid_opts) >= 2:
                valid_questions.append({
                    "type": "mc",
                    "text": q["text"].strip(),
                    "options": valid_opts,
                })
        else:
            valid_questions.append({"type": "open", "text": q["text"].strip()})

    if valid_questions:
        n_open = sum(1 for q in valid_questions if q["type"] == "open")
        n_mc = sum(1 for q in valid_questions if q["type"] == "mc")
        st.caption(f"\u2705 {len(valid_questions)} valid question(s) ready ({n_open} open-ended, {n_mc} multiple-choice)")
    elif questions:
        st.warning("Fill in question text and at least 2 options for MC questions.")

    return valid_questions



def fetch_news_context(topic):
    """Fetch recent headlines from Google News RSS and summarize with one Gemini call."""
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(topic)}&hl=en-CA&gl=CA&ceid=CA:en"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")[:10]
        headlines = []
        for item in items:
            title = item.findtext("title", "")
            if title:
                headlines.append(title)
        if not headlines:
            return f"No recent news found for: {topic}"
        headline_text = "\n".join(f"- {h}" for h in headlines)
        client = genai.Client(api_key=API_KEY)
        summary_prompt = (
            f"Below are recent news headlines about \"{topic}\". "
            f"Summarize the key themes and current situation in 2-3 concise sentences. "
            f"Focus on facts and developments that would shape how a Canadian would think "
            f"about this topic today. Be neutral and factual.\n\n"
            f"HEADLINES:\n{headline_text}\n\n"
            f"SUMMARY:"
        )
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=summary_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=300,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"Could not fetch news: {str(e)[:200]}"




def render_context_panel(panel_key="consumer"):
    """Render an optional current events context panel. Returns context string."""
    area_key = f"{panel_key}_context_area"
    topic_key = f"{panel_key}_news_topic"

    if area_key not in st.session_state:
        st.session_state[area_key] = ""
    if topic_key not in st.session_state:
        st.session_state[topic_key] = ""

    with st.expander("\U0001f4f0 Current Events Context (Optional)", expanded=bool(st.session_state[area_key])):
        st.caption(
            "Ground personas in recent events so their responses reflect current reality. "
            "Enter a topic to auto-fetch headlines, or type your own context below."
        )
        tcol1, tcol2 = st.columns([3, 1])
        with tcol1:
            topic = st.text_input(
                "News Topic",
                value=st.session_state[topic_key],
                key=f"{panel_key}_topic_input",
                placeholder="e.g. Canada US trade relations, Canadian housing market...",
                label_visibility="collapsed",
            )
        with tcol2:
            fetch_btn = st.button(
                "\U0001f50d Fetch Headlines",
                key=f"{panel_key}_fetch_news",
                use_container_width=True,
                disabled=not topic.strip(),
            )

        if fetch_btn and topic.strip():
            st.session_state[topic_key] = topic.strip()
            with st.spinner("Fetching and summarizing recent headlines..."):
                summary = fetch_news_context(topic.strip())
            st.session_state[area_key] = summary
            st.rerun()

        context = st.text_area(
            "Context (edit or type your own)",
            key=area_key,
            height=100,
            placeholder="Recent events, market conditions, or other context that personas should consider when responding...",
        )

    return context.strip()



# ============================================================
# EXCEL EXPORT
# ============================================================

def _counter_to_df(counter_list, col_name="Item", count_name="Count"):
    """Convert Counter.most_common() list to a DataFrame."""
    if not counter_list:
        return pd.DataFrame({col_name: [], count_name: []})
    return pd.DataFrame(counter_list, columns=[col_name, count_name])


def _write_sheet(writer, sheet_name, df, index=False):
    """Write a DataFrame to an Excel sheet, truncating sheet name if needed."""
    # Excel sheet names limited to 31 chars
    safe_name = sheet_name[:31]
    df.to_excel(writer, sheet_name=safe_name, index=index)


def build_excel_export_reactor(df, analysis, idea_text, is_advisor=False, idea_type="investment", context_text=""):
    """Build a multi-sheet Excel workbook for Idea Reactor / General Idea results."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # --- Sheet 1: Summary ---
        summary_rows = []
        summary_rows.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        summary_rows.append(["Mode", "Idea Reactor (Advisors)" if is_advisor else "Idea Reactor"])
        summary_rows.append(["Idea Type", "General Idea" if idea_type == "general" else "Investment Idea"])
        summary_rows.append(["Idea", idea_text])
        if context_text:
            summary_rows.append(["Current Events Context", context_text])
        summary_rows.append(["Sample Size", len(df)])
        summary_rows.append(["", ""])
        summary_rows.append(["Key Metrics", ""])
        summary_rows.append(["Avg Interest Score", round(df["interest_score"].mean(), 2)])
        summary_rows.append(["Median Interest Score", round(df["interest_score"].median(), 2)])
        if is_advisor:
            summary_rows.append(["% Would Recommend", round(df["would_recommend"].mean() * 100, 1)])
        else:
            summary_rows.append(["% Would Invest / Try", round(df["would_invest"].mean() * 100, 1)])
        top_sent = df["sentiment"].mode().iloc[0] if not df["sentiment"].mode().empty else "N/A"
        summary_rows.append(["Top Sentiment", top_sent])
        summary_rows.append(["", ""])
        summary_rows.append(["Sentiment Breakdown", ""])
        for s, c in df["sentiment"].value_counts().items():
            summary_rows.append([f"  {s}", int(c)])
        summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
        _write_sheet(writer, "Summary", summary_df)

        # --- Sheet 2: Raw Responses ---
        if is_advisor:
            resp_cols = [
                "persona_name", "age", "gender", "province", "firm_type", "years_in_business",
                "designations", "book_size_aum", "practice_focus", "personal_income",
                "interest_score", "sentiment", "would_recommend", "client_suitability",
                "gut_reaction", "verbatim_quote", "what_would_help",
            ]
        else:
            resp_cols = [
                "persona_name", "age", "gender", "province", "city", "income", "net_worth",
                "risk_tolerance_profile", "life_stage", "investment_knowledge", "education",
                "interest_score", "sentiment", "would_invest", "investment_amount",
                "gut_reaction", "verbatim_quote", "what_would_help",
            ]
        available = [c for c in resp_cols if c in df.columns]
        resp_df = df[available].copy().sort_values("interest_score", ascending=False)
        _write_sheet(writer, "Raw Responses", resp_df)

        # --- Sheet 3: Top Concerns ---
        concerns_df = _counter_to_df(analysis.get("top_concerns", []), "Concern", "Mentions")
        _write_sheet(writer, "Top Concerns", concerns_df)

        # --- Sheet 4: Top Appeals ---
        appeals_df = _counter_to_df(analysis.get("top_appeals", []), "Appeal Factor", "Mentions")
        _write_sheet(writer, "Top Appeals", appeals_df)

        # --- Sheet 5: What Would Help ---
        helps = analysis.get("what_would_help", [])
        helps_df = pd.DataFrame({"Suggestion": helps})
        _write_sheet(writer, "What Would Help", helps_df)

        # --- Sheet 6: By Age Group ---
        if "age_group" in df.columns:
            if is_advisor:
                age_agg = df.groupby("age_group", observed=True).agg(
                    avg_score=("interest_score", "mean"),
                    would_recommend_pct=("would_recommend", lambda x: x.mean() * 100),
                    count=("interest_score", "count"),
                ).reset_index()
            else:
                age_agg = df.groupby("age_group", observed=True).agg(
                    avg_score=("interest_score", "mean"),
                    would_invest_pct=("would_invest", lambda x: x.mean() * 100),
                    count=("interest_score", "count"),
                ).reset_index()
            _write_sheet(writer, "By Age Group", age_agg)

        # --- Sheet 7: By Income ---
        if "income_bracket" in df.columns:
            if is_advisor:
                inc_agg = df.groupby("income_bracket", observed=True).agg(
                    avg_score=("interest_score", "mean"),
                    would_recommend_pct=("would_recommend", lambda x: x.mean() * 100),
                    count=("interest_score", "count"),
                ).reset_index()
            else:
                inc_agg = df.groupby("income_bracket", observed=True).agg(
                    avg_score=("interest_score", "mean"),
                    would_invest_pct=("would_invest", lambda x: x.mean() * 100),
                    count=("interest_score", "count"),
                ).reset_index()
            _write_sheet(writer, "By Income", inc_agg)

        # --- Sheet 8: Advisor-specific or Risk-specific ---
        if is_advisor and "years_group" in df.columns:
            yr_agg = df.groupby("years_group", observed=True).agg(
                avg_score=("interest_score", "mean"),
                would_recommend_pct=("would_recommend", lambda x: x.mean() * 100),
                count=("interest_score", "count"),
            ).reset_index()
            _write_sheet(writer, "By Experience", yr_agg)

            if "book_size_group" in df.columns:
                bs_agg = df.groupby("book_size_group", observed=True).agg(
                    avg_score=("interest_score", "mean"),
                    would_recommend_pct=("would_recommend", lambda x: x.mean() * 100),
                    count=("interest_score", "count"),
                ).reset_index()
                _write_sheet(writer, "By Book Size", bs_agg)

        elif not is_advisor and "risk_tolerance_profile" in df.columns:
            risk_agg = df.groupby("risk_tolerance_profile", observed=True).agg(
                avg_score=("interest_score", "mean"),
                would_invest_pct=("would_invest", lambda x: x.mean() * 100),
                count=("interest_score", "count"),
            ).reset_index()
            _write_sheet(writer, "By Risk Tolerance", risk_agg)

        # --- Sheet 9: By Province ---
        if "province" in df.columns:
            prov_agg = df.groupby("province", observed=True).agg(
                avg_score=("interest_score", "mean"),
                count=("interest_score", "count"),
            ).reset_index().sort_values("count", ascending=False)
            _write_sheet(writer, "By Province", prov_agg)

    return output.getvalue()


def build_excel_export_ab(df_a, df_b, analysis_a, analysis_b, idea_a, idea_b, is_advisor=False, context_text=""):
    """Build a multi-sheet Excel workbook for A/B Test results."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # --- Sheet 1: Comparison Summary ---
        rows = []
        rows.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        rows.append(["Mode", "A/B Test (Advisors)" if is_advisor else "A/B Test"])
        rows.append(["Variant A", idea_a])
        rows.append(["Variant B", idea_b])
        if context_text:
            rows.append(["Current Events Context", context_text])
        rows.append(["", ""])
        rows.append(["Metric", "Variant A | Variant B"])

        def _get_metric(name, val_a, val_b, fmt="{:.2f}"):
            a_str = fmt.format(val_a) if val_a is not None else "N/A"
            b_str = fmt.format(val_b) if val_b is not None else "N/A"
            rows.append([name, f"{a_str}  |  {b_str}"])

        if df_a is not None and not df_a.empty:
            avg_a = df_a["interest_score"].mean()
            med_a = df_a["interest_score"].median()
        else:
            avg_a = med_a = None
        if df_b is not None and not df_b.empty:
            avg_b = df_b["interest_score"].mean()
            med_b = df_b["interest_score"].median()
        else:
            avg_b = med_b = None

        _get_metric("Avg Interest Score", avg_a, avg_b)
        _get_metric("Median Interest Score", med_a, med_b)

        if is_advisor:
            rec_a = df_a["would_recommend"].mean() * 100 if df_a is not None and not df_a.empty else None
            rec_b = df_b["would_recommend"].mean() * 100 if df_b is not None and not df_b.empty else None
            _get_metric("% Would Recommend", rec_a, rec_b, "{:.1f}%")
        else:
            inv_a = df_a["would_invest"].mean() * 100 if df_a is not None and not df_a.empty else None
            inv_b = df_b["would_invest"].mean() * 100 if df_b is not None and not df_b.empty else None
            _get_metric("% Would Invest / Try", inv_a, inv_b, "{:.1f}%")

        summary_df = pd.DataFrame(rows, columns=["Item", "Value"])
        _write_sheet(writer, "Comparison Summary", summary_df)

        # --- Variant A & B Raw Data ---
        if is_advisor:
            resp_cols = [
                "persona_name", "age", "gender", "province", "firm_type", "years_in_business",
                "designations", "book_size_aum", "practice_focus",
                "interest_score", "sentiment", "would_recommend", "client_suitability",
                "gut_reaction", "verbatim_quote",
            ]
        else:
            resp_cols = [
                "persona_name", "age", "gender", "province", "income", "net_worth",
                "risk_tolerance_profile", "life_stage", "investment_knowledge",
                "interest_score", "sentiment", "would_invest", "investment_amount",
                "gut_reaction", "verbatim_quote",
            ]

        if df_a is not None and not df_a.empty:
            a_cols = [c for c in resp_cols if c in df_a.columns]
            _write_sheet(writer, "Variant A Responses", df_a[a_cols].sort_values("interest_score", ascending=False))
        if df_b is not None and not df_b.empty:
            b_cols = [c for c in resp_cols if c in df_b.columns]
            _write_sheet(writer, "Variant B Responses", df_b[b_cols].sort_values("interest_score", ascending=False))

        # --- Concerns & Appeals for each variant ---
        if analysis_a:
            _write_sheet(writer, "A - Top Concerns", _counter_to_df(analysis_a.get("top_concerns", []), "Concern", "Mentions"))
            _write_sheet(writer, "A - Top Appeals", _counter_to_df(analysis_a.get("top_appeals", []), "Appeal", "Mentions"))
        if analysis_b:
            _write_sheet(writer, "B - Top Concerns", _counter_to_df(analysis_b.get("top_concerns", []), "Concern", "Mentions"))
            _write_sheet(writer, "B - Top Appeals", _counter_to_df(analysis_b.get("top_appeals", []), "Appeal", "Mentions"))

    return output.getvalue()


def build_excel_export_survey(df, per_question, questions, is_advisor=False, context_text=""):
    """Build a multi-sheet Excel workbook for Survey results."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # --- Sheet 1: Summary ---
        rows = []
        rows.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        rows.append(["Mode", "Survey (Advisors)" if is_advisor else "Survey"])
        rows.append(["Respondents", df["persona_id"].nunique()])
        rows.append(["Questions", len(questions)])
        if context_text:
            rows.append(["Current Events Context", context_text])
        rows.append(["", ""])
        rows.append(["Questions Summary", ""])
        for i, q in enumerate(questions, 1):
            q_type_label = "MC" if q["type"] == "mc" else "Open-Ended"
            rows.append([f"Q{i} ({q_type_label})", q["text"]])
            if q["type"] == "mc":
                for opt in q.get("options", []):
                    rows.append(["  Option", opt])
        summary_df = pd.DataFrame(rows, columns=["Item", "Value"])
        _write_sheet(writer, "Summary", summary_df)

        # --- Sheet 2: All Responses (long format) ---
        if is_advisor:
            resp_cols = [
                "persona_name", "age", "gender", "province", "firm_type", "years_in_business",
                "designations", "book_size_aum", "practice_focus",
                "question_number", "question_text", "question_type", "selected_option",
                "answer", "sentiment", "confidence",
            ]
        else:
            resp_cols = [
                "persona_name", "age", "gender", "province", "income", "net_worth",
                "risk_tolerance_profile", "life_stage", "investment_knowledge",
                "question_number", "question_text", "question_type", "selected_option",
                "answer", "sentiment", "confidence",
            ]
        available = [c for c in resp_cols if c in df.columns]
        _write_sheet(writer, "All Responses", df[available].sort_values(["question_number", "persona_name"]))

        # --- Per-question sheets ---
        for qnum, pq in per_question.items():
            q_def = questions[qnum - 1]
            sheet_name = f"Q{qnum}"
            qrows = []
            qrows.append(["Question", q_def["text"]])
            qrows.append(["Type", q_def["type"]])
            qrows.append(["Responses", pq.get("response_count", 0)])
            qrows.append(["", ""])
            qrows.append(["Sentiment", "Count"])
            for s, c in pq.get("sentiment_counts", {}).items():
                qrows.append([s, int(c)])
            qrows.append(["", ""])
            qrows.append(["Confidence", "Count"])
            for s, c in pq.get("confidence_counts", {}).items():
                qrows.append([s, int(c)])

            if q_def["type"] == "mc" and "option_counts" in pq:
                qrows.append(["", ""])
                qrows.append(["Option", "Selections"])
                for opt, count in pq["option_counts"].items():
                    qrows.append([opt, int(count)])

            themes = pq.get("top_themes", [])
            if themes:
                qrows.append(["", ""])
                qrows.append(["Top Theme", "Mentions"])
                for t, c in themes:
                    qrows.append([t, int(c)])

            q_df = pd.DataFrame(qrows, columns=["Item", "Value"])
            _write_sheet(writer, sheet_name, q_df)

    return output.getvalue()


def render_excel_export_button(label, excel_bytes, filename_prefix, key):
    """Render a styled download button for Excel export."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label=f"📊  {label}",
        data=excel_bytes,
        file_name=f"{filename_prefix}_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
        type="primary",
        use_container_width=False,
    )



# ============================================================
# OPEN-ENDED RESPONSE ANALYSIS HELPERS
# ============================================================

EXTRA_STOPWORDS = {
    "would", "could", "should", "really", "actually", "thing", "things",
    "people", "person", "make", "makes", "made", "get", "got", "getting",
    "going", "go", "way", "ways", "lot", "lots", "much", "many", "good",
    "bad", "great", "well", "even", "still", "never", "always", "sometimes",
    "feel", "feels", "felt", "think", "thought", "thinking", "see", "look",
    "know", "knew", "want", "wants", "wanted", "need", "needs", "needed",
    "use", "used", "using", "say", "said", "saying", "tell", "told", "talk",
    "one", "two", "three", "first", "second", "every", "another", "also",
    "however", "therefore", "moreover", "though", "since", "because",
    "instead", "rather", "quite", "very", "pretty", "kind", "sort", "bit",
    "around", "back", "another", "yes", "no", "maybe", "perhaps",
    "rbc", "advisor", "client", "clients", "advisors", "im", "ive", "id",
    "im", "thats", "theres", "youre", "dont", "doesnt", "wont", "cant",
    "isnt", "arent", "wasnt", "werent", "wouldnt", "couldnt", "shouldnt",
}


def _extract_words(texts, min_length=3):
    """Tokenize a list of texts into clean lowercase words."""
    word_re = re.compile(r"[a-zA-Z']+")
    all_words = []
    for t in texts:
        if not isinstance(t, str):
            continue
        for w in word_re.findall(t.lower()):
            w = w.strip("'")
            if len(w) >= min_length and w not in EXTRA_STOPWORDS:
                all_words.append(w)
    return all_words


def _extract_ngrams(texts, n=2, min_count=2, top_k=20):
    """Extract top-k n-grams (bigrams or trigrams) from texts, excluding pure stopword combinations."""
    word_re = re.compile(r"[a-zA-Z']+")
    sw = EXTRA_STOPWORDS | (set(WC_STOPWORDS) if WC_STOPWORDS else set())
    counts = Counter()
    for t in texts:
        if not isinstance(t, str):
            continue
        words = [w.strip("'") for w in word_re.findall(t.lower())]
        words = [w for w in words if len(w) >= 2]
        for i in range(len(words) - n + 1):
            gram = words[i:i+n]
            # Skip if all words are stopwords
            non_sw = [w for w in gram if w not in sw and len(w) >= 3]
            if len(non_sw) >= max(1, n - 1):
                counts[" ".join(gram)] += 1
    return [(g, c) for g, c in counts.most_common(top_k) if c >= min_count]


def render_word_cloud(texts, key=None):
    """Render a word cloud image inside Streamlit. Falls back gracefully if wordcloud unavailable."""
    if not WORDCLOUD_AVAILABLE:
        return False
    words = _extract_words(texts)
    if len(words) < 5:
        return False
    text_blob = " ".join(words)
    sw = set(WC_STOPWORDS) | EXTRA_STOPWORDS
    wc = WordCloud(
        width=900, height=350,
        background_color="white",
        colormap="viridis",
        stopwords=sw,
        max_words=80,
        relative_scaling=0.5,
        collocations=True,
    ).generate(text_blob)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px;" />',
        unsafe_allow_html=True
    )
    return True


@st.cache_data(ttl=3600, show_spinner=False)
def summarize_open_ended(question_text, answers_text_blob, audience_label="respondents"):
    """One LLM call to summarize all open-ended responses to a question.
    Returns a dict with: headline (str), key_themes (list of str), surprising (str)."""
    if not answers_text_blob.strip():
        return None
    # Trim if too long (prevent runaway token cost)
    snippet = answers_text_blob[:12000]
    prompt = (
        f"Below are open-ended survey responses from {audience_label} answering the question:\n"
        f"\"{question_text}\"\n\n"
        f"Respond ONLY with valid JSON in this exact format (no markdown, no code fences):\n"
        f"{{\n"
        f'  "headline": "<one-sentence summary of the dominant pattern in the responses>",\n'
        f'  "key_themes": ["<theme 1 with 1-line description>", "<theme 2>", "<theme 3>", "<theme 4>"],\n'
        f'  "surprising": "<one notable or surprising insight from the responses, or empty string>",\n'
        f'  "tactical_takeaway": "<one actionable next step suggested by the responses>"\n'
        f"}}\n\n"
        f"RESPONSES:\n{snippet}"
    )
    try:
        text = call_llm(prompt, model_id=DEFAULT_MODEL_ID, response_json=True,
                        max_tokens=600, temperature=0.3)
        return parse_json_response(text)
    except Exception as e:
        return {"headline": f"Could not summarize: {str(e)[:120]}", "key_themes": [], "surprising": "", "tactical_takeaway": ""}


def render_open_ended_panel(question_text, q_df, audience_label="respondents", key_prefix=""):
    """Render the rich open-ended analysis panel: AI summary, word cloud,
    key phrases, and representative quotes by sentiment."""
    answers = [str(a) for a in q_df.get("answer", []) if isinstance(a, str) and a.strip()]
    if not answers:
        st.info("No open-ended responses to analyze.")
        return

    # AI summary - cached so re-renders don't re-call
    blob = "\n".join(f"- {a[:500]}" for a in answers)
    with st.spinner("Summarizing responses..."):
        summary = summarize_open_ended(question_text, blob, audience_label=audience_label)

    if summary:
        st.markdown(f"### 💡 {summary.get('headline', '')}")
        themes = summary.get("key_themes", [])
        if themes:
            st.markdown("**Key themes:**")
            for t in themes:
                st.markdown(f"- {t}")
        surprising = summary.get("surprising", "")
        if surprising:
            st.info(f"**Notable:** {surprising}")
        tact = summary.get("tactical_takeaway", "")
        if tact:
            st.success(f"**Suggested action:** {tact}")

    st.divider()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**☁️ Word Cloud**")
        rendered = render_word_cloud(answers, key=f"{key_prefix}_wc")
        if not rendered:
            st.caption("Word cloud unavailable (wordcloud package not installed or too few words).")
    with c2:
        st.markdown("**🔍 Top Phrases (2-3 word)**")
        bigrams = _extract_ngrams(answers, n=2, min_count=2, top_k=10)
        trigrams = _extract_ngrams(answers, n=3, min_count=2, top_k=5)
        all_phrases = bigrams + trigrams
        if all_phrases:
            ph_df = pd.DataFrame(all_phrases, columns=["Phrase", "Mentions"])
            ph_df = ph_df.sort_values("Mentions", ascending=True).tail(15)
            fig = px.bar(ph_df, x="Mentions", y="Phrase", orientation="h",
                         color_discrete_sequence=["#2d5a87"], height=350)
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_phrases")
        else:
            st.caption("Not enough repeated phrases to chart.")

    # Representative quotes per sentiment
    st.markdown("**🗣️ Representative Quotes by Sentiment**")
    sent_cols = st.columns(3)
    for i, sentiment in enumerate(["positive", "neutral", "negative"]):
        with sent_cols[i]:
            sent_emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}[sentiment]
            sent_df = q_df[q_df.get("sentiment", "") == sentiment].head(3)
            st.markdown(f"{sent_emoji} **{sentiment.title()}** ({(q_df.get('sentiment','') == sentiment).sum()})")
            if sent_df.empty:
                st.caption("_No responses_")
            else:
                for _, r in sent_df.iterrows():
                    name = r.get("persona_name", "")
                    ans = str(r.get("answer", ""))[:300]
                    st.markdown(f"> *{ans}*  \n— _{name}_")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Synthetic Persona Reactor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 16px 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .stMetric label { color: rgba(255,255,255,0.85) !important; }
    .stMetric [data-testid="stMetricValue"] { color: white !important; }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_personas():
    with open(PERSONAS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_advisor_personas():
    with open(ADVISOR_PERSONAS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rbc_ds_personas():
    """Load all RBC Dominion Securities advisor personas."""
    try:
        with open(RBC_DS_PERSONAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def load_rbc_pim_personas():
    """Load RBC DS advisors who are PIM / Discretionary licensed."""
    try:
        with open(RBC_PIM_PERSONAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# ============================================================
# USAGE LOGGING
# ============================================================

def log_usage(entry):
    """Append a usage entry to the persistent log file."""
    entry["timestamp"] = datetime.now().isoformat()
    entry["estimated_cost_usd"] = round(entry.get("api_calls", 0) * COST_PER_API_CALL, 4)
    try:
        if os.path.exists(USAGE_LOG_FILE):
            with open(USAGE_LOG_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = []
        log.append(entry)
        with open(USAGE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception:
        pass  # Don't break the app if logging fails


def stratified_sample(personas, n):
    rng = random.Random(42)
    if n >= len(personas):
        return personas
    strata = {}
    for p in personas:
        age_group = "18-34" if p["age"] < 35 else ("35-54" if p["age"] < 55 else "55+")
        income_group = "low" if p["household_income"] < 60000 else ("mid" if p["household_income"] < 120000 else "high")
        key = (age_group, income_group, p["risk_tolerance"])
        strata.setdefault(key, []).append(p)
    sampled = []
    total = len(personas)
    for members in strata.values():
        stratum_n = max(1, round(len(members) / total * n))
        sampled.extend(rng.sample(members, min(stratum_n, len(members))))
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)
    elif len(sampled) < n:
        remaining = [p for p in personas if p not in sampled]
        sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))
    return sampled[:n]


def stratified_sample_advisors(personas, n):
    """Stratified sample for advisor personas by years_group x book_size_group x firm_type."""
    rng = random.Random(42)
    if n >= len(personas):
        return personas
    strata = {}
    for p in personas:
        yrs = p["years_in_business"]
        years_group = "junior" if yrs <= 5 else ("mid" if yrs <= 15 else "senior")
        aum = p["book_size_aum"]
        book_group = "small" if aum < 50_000_000 else ("medium" if aum < 150_000_000 else "large")
        key = (years_group, book_group, p["firm_type"])
        strata.setdefault(key, []).append(p)
    sampled = []
    total = len(personas)
    for members in strata.values():
        stratum_n = max(1, round(len(members) / total * n))
        sampled.extend(rng.sample(members, min(stratum_n, len(members))))
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)
    elif len(sampled) < n:
        remaining = [p for p in personas if p not in sampled]
        sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))
    return sampled[:n]


def stratified_sample_rbc(personas, n):
    """Stratified sample for RBC advisors by province x years_group x pim_licensed."""
    rng = random.Random(42)
    if n >= len(personas):
        return personas
    strata = {}
    for p in personas:
        yrs = p.get("years_in_business", 15)
        years_group = "junior" if yrs <= 5 else ("mid" if yrs <= 15 else "senior")
        prov = p.get("province", "Unknown")
        # Collapse small provinces to keep strata balanced
        major_provs = {"Ontario", "British Columbia", "Quebec", "Alberta"}
        prov_key = prov if prov in major_provs else "Other"
        pim = p.get("pim_licensed", "Unknown")
        key = (prov_key, years_group, pim)
        strata.setdefault(key, []).append(p)
    sampled = []
    total = len(personas)
    for members in strata.values():
        stratum_n = max(1, round(len(members) / total * n))
        sampled.extend(rng.sample(members, min(stratum_n, len(members))))
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)
    elif len(sampled) < n:
        remaining = [p for p in personas if p not in sampled]
        sampled.extend(rng.sample(remaining, min(n - len(sampled), len(remaining))))
    return sampled[:n]


def attach_persona_metadata(result, persona):
    """Attach persona demographic metadata to an API result."""
    result["persona_id"] = persona["id"]
    result["persona_name"] = f"{persona['first_name']} {persona['last_name']}"
    result["age"] = persona["age"]
    result["gender"] = persona["gender"]
    result["ethnicity"] = persona["ethnicity"]
    result["province"] = persona["province"]
    result["city"] = persona["city"]
    result["income"] = persona["household_income"]
    result["net_worth"] = persona["net_worth"]
    result["risk_tolerance_profile"] = persona["risk_tolerance"]
    result["life_stage"] = persona["life_stage"]
    result["investment_knowledge"] = persona["investment_knowledge"]
    result["education"] = persona["education"]
    result["family_status"] = persona.get("family_status", "")
    return result


def attach_advisor_metadata(result, persona):
    """Attach advisor persona metadata to an API result. Handles real advisor
    records (e.g. RBC DS) that may not include every synthetic field."""
    result["persona_id"] = persona.get("id", 0)
    fn = persona.get("first_name", "")
    ln = persona.get("last_name", "")
    result["persona_name"] = f"{fn} {ln}".strip() or "Unknown"
    result["age"] = persona.get("age", 0)
    result["gender"] = persona.get("gender", "Unknown")
    result["province"] = persona.get("province", "Unknown")
    result["city"] = persona.get("city", "Unknown")
    result["firm_type"] = persona.get("firm_type", "Unknown")
    result["years_in_business"] = persona.get("years_in_business", 0)
    designations = persona.get("designations", [])
    result["designations"] = ", ".join(designations) if isinstance(designations, list) else (designations or "")
    result["book_size_aum"] = persona.get("book_size_aum", 0)
    result["num_clients"] = persona.get("num_clients", 0)
    result["practice_focus"] = persona.get("practice_focus", "")
    result["compensation_model"] = persona.get("compensation_model", "")
    result["personal_income"] = persona.get("personal_income", 0)
    result["business_maturity"] = persona.get("business_maturity", "")
    result["client_demographics"] = persona.get("client_demographics", "")
    result["education"] = persona.get("education", "")
    result["family_status"] = persona.get("family_status", "")
    # RBC-specific
    if "pim_licensed" in persona:
        result["pim_licensed"] = persona["pim_licensed"]
    if "title" in persona:
        result["title"] = persona["title"]
    return result


def make_error_result(persona, error_msg):
    """Create an error result dict for a persona."""
    return {
        "persona_id": persona["id"],
        "persona_name": f"{persona['first_name']} {persona['last_name']}",
        "error": error_msg,
    }


# ============================================================
# LLM PROVIDER ABSTRACTION
# ============================================================

def call_llm(prompt, model_id=None, response_json=True, max_tokens=1000, temperature=0.8):
    """Provider-agnostic LLM call. Returns the text response (string).
    Routes to OpenAI / Anthropic / Google based on MODEL_CONFIGS[model_id]['provider']."""
    if not model_id:
        model_id = DEFAULT_MODEL_ID
    cfg = MODEL_CONFIGS.get(model_id)
    if not cfg:
        # Unknown model — fall back to default
        cfg = MODEL_CONFIGS[DEFAULT_MODEL_ID]
        model_id = DEFAULT_MODEL_ID
    provider = cfg["provider"]

    if provider == "google":
        client = genai.Client(api_key=GEMINI_API_KEY)
        gen_cfg_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }
        if response_json:
            gen_cfg_kwargs["response_mime_type"] = "application/json"
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(**gen_cfg_kwargs),
        )
        return response.text.strip()

    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set in secrets.toml")
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        # GPT-5 family uses Responses API and may not accept temperature; use generic chat.completions
        kwargs = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        # GPT-5 family doesn't support custom temperature for some endpoints; use default
        if not model_id.startswith("gpt-5"):
            kwargs["temperature"] = temperature
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    elif provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set in secrets.toml")
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # Anthropic doesn't have a JSON mode; rely on the prompt's JSON instruction.
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns content blocks; concatenate text blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "".join(text_parts).strip()

    else:
        raise RuntimeError(f"Unknown provider: {provider}")


def parse_json_response(text):
    """Strip markdown fences and parse JSON. Raises json.JSONDecodeError on failure."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Some models wrap JSON in extra text; try to find the JSON object
    if not text.startswith("{") and not text.startswith("["):
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
    return json.loads(text)


# ============================================================
# API CALLS - REACTOR
# ============================================================

def get_reaction(persona, idea, is_advisor=False, context="", idea_type="investment", model_id=None):
    if idea_type == "general":
        prompt_template = ADVISOR_GENERAL_IDEA_PROMPT if is_advisor else GENERAL_IDEA_PROMPT
    else:
        prompt_template = ADVISOR_REACTION_PROMPT if is_advisor else REACTION_PROMPT
    attach_fn = attach_advisor_metadata if is_advisor else attach_persona_metadata
    ctx_block = f"\nCURRENT EVENTS CONTEXT (recent developments this persona would be aware of):\n{context}\n" if context else "\n"
    prompt = prompt_template.format(persona=persona["persona_summary"], idea=idea, context=ctx_block)
    for attempt in range(3):
        try:
            text = call_llm(prompt, model_id=model_id, response_json=True,
                            max_tokens=1000, temperature=0.8)
            reaction = parse_json_response(text)
            return attach_fn(reaction, persona)
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(1)
                continue
            return make_error_result(persona, "JSON parse error")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                time.sleep((attempt + 1) * 10)
                continue
            if attempt < 2:
                time.sleep(2)
                continue
            return make_error_result(persona, str(e)[:100])


def collect_reactions(personas, idea, is_advisor=False, context="", idea_type="investment", model_id=None):
    total = len(personas)
    reactions = []
    completed = 0
    errors = 0
    # Adjust pacing for non-Google providers (lower default rate limits)
    cfg = MODEL_CONFIGS.get(model_id or DEFAULT_MODEL_ID, {})
    if cfg.get("provider") in ("openai", "anthropic"):
        delay = 60.0 / 60   # ~1 req/s
        max_workers = 5
    else:
        delay = 60.0 / 450
        max_workers = MAX_WORKERS

    label = MODEL_CONFIGS.get(model_id or DEFAULT_MODEL_ID, {}).get("label", "model")
    progress_bar = st.progress(0, text=f"Starting ({label})... 0/{total}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_persona = {}
        for persona in personas:
            future = executor.submit(get_reaction, persona, idea, is_advisor, context, idea_type, model_id)
            future_to_persona[future] = persona
            time.sleep(delay)

        for future in as_completed(future_to_persona):
            completed += 1
            result = future.result()
            reactions.append(result)
            if "error" in result:
                errors += 1
            progress_bar.progress(
                completed / total,
                text=f"Collected {completed}/{total} reactions ({errors} errors) — {label}"
            )

    progress_bar.progress(1.0, text=f"Done! {total} reactions collected ({errors} errors) — {label}")
    return reactions


# ============================================================
# API CALLS - SURVEY
# ============================================================

def get_survey_response(persona, questions, is_advisor=False, context="", model_id=None):
    seed = hash(persona.get("id", persona.get("persona_summary", "")))
    questions_formatted = format_questions_for_prompt(questions, seed=seed)
    prompt_template = ADVISOR_SURVEY_PROMPT if is_advisor else SURVEY_PROMPT
    attach_fn = attach_advisor_metadata if is_advisor else attach_persona_metadata
    ctx_block = f"\nCURRENT EVENTS CONTEXT (recent developments this persona would be aware of):\n{context}\n" if context else "\n"
    prompt = prompt_template.format(
        persona=persona["persona_summary"],
        questions_formatted=questions_formatted,
        context=ctx_block,
    )
    max_tok = min(300 * len(questions), 8000)
    for attempt in range(3):
        try:
            text = call_llm(prompt, model_id=model_id, response_json=True,
                            max_tokens=max_tok, temperature=0.8)
            result = parse_json_response(text)
            return attach_fn(result, persona)
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(1)
                continue
            return make_error_result(persona, "JSON parse error")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                time.sleep((attempt + 1) * 10)
                continue
            if attempt < 2:
                time.sleep(2)
                continue
            return make_error_result(persona, str(e)[:100])


def collect_survey_responses(personas, questions, is_advisor=False, context="", model_id=None):
    total = len(personas)
    responses = []
    completed = 0
    errors = 0
    cfg = MODEL_CONFIGS.get(model_id or DEFAULT_MODEL_ID, {})
    if cfg.get("provider") in ("openai", "anthropic"):
        delay = 60.0 / 60
        max_workers = 5
    else:
        delay = 60.0 / 450
        max_workers = MAX_WORKERS

    label = cfg.get("label", "model")
    progress_bar = st.progress(0, text=f"Starting survey ({label})... 0/{total}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_persona = {}
        for persona in personas:
            future = executor.submit(get_survey_response, persona, questions, is_advisor, context, model_id)
            future_to_persona[future] = persona
            time.sleep(delay)

        for future in as_completed(future_to_persona):
            completed += 1
            result = future.result()
            responses.append(result)
            if "error" in result:
                errors += 1
            progress_bar.progress(
                completed / total,
                text=f"Surveyed {completed}/{total} personas ({errors} errors) — {label}"
            )

    progress_bar.progress(1.0, text=f"Done! {total} personas surveyed ({errors} errors) — {label}")
    return responses



# ============================================================
# ANALYSIS - REACTOR
# ============================================================

def build_analysis(reactions):
    valid = [r for r in reactions if "error" not in r]
    if not valid:
        return None, None

    df = pd.DataFrame(valid)

    df["interest_score"] = pd.to_numeric(df["interest_score"], errors="coerce")
    df["would_invest"] = df["would_invest"].astype(bool)
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df["net_worth"] = pd.to_numeric(df["net_worth"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["age_group"] = pd.cut(df["age"], bins=[0, 34, 54, 100], labels=["18-34", "35-54", "55+"])
    df["income_bracket"] = pd.cut(
        df["income"], bins=[0, 60000, 120000, float("inf")],
        labels=["Under $60K", "$60K-$120K", "$120K+"]
    )

    all_concerns = []
    all_appeals = []
    all_helps = []
    for _, row in df.iterrows():
        concerns = row.get("key_concerns", [])
        if isinstance(concerns, list):
            all_concerns.extend(concerns)
        appeals = row.get("appeal_factors", [])
        if isinstance(appeals, list):
            all_appeals.extend(appeals)
        wh = row.get("what_would_help", "")
        if isinstance(wh, str) and wh:
            all_helps.append(wh)

    analysis = {
        "top_concerns": Counter(all_concerns).most_common(12),
        "top_appeals": Counter(all_appeals).most_common(12),
        "what_would_help": all_helps,
    }

    return df, analysis


# ============================================================
# ANALYSIS - SURVEY
# ============================================================

def build_survey_analysis(responses, questions):
    valid = [r for r in responses if "error" not in r]
    if not valid:
        return None, None

    # Build long-format DataFrame: one row per persona per question
    rows = []
    for resp in valid:
        answers = resp.get("answers", [])
        for ans in answers:
            qnum = ans.get("question_number", 0)
            if 1 <= qnum <= len(questions):
                q_def = questions[qnum - 1]
                selected = ans.get("selected_option")

                # Validate MC selected_option against allowed options (force to valid option)
                if q_def["type"] == "mc":
                    selected = _fuzzy_match_option(selected, q_def["options"])

                rows.append({
                    "persona_id": resp["persona_id"],
                    "persona_name": resp["persona_name"],
                    "age": resp["age"],
                    "gender": resp["gender"],
                    "ethnicity": resp["ethnicity"],
                    "province": resp["province"],
                    "city": resp.get("city", ""),
                    "income": resp["income"],
                    "net_worth": resp["net_worth"],
                    "risk_tolerance_profile": resp["risk_tolerance_profile"],
                    "life_stage": resp["life_stage"],
                    "investment_knowledge": resp["investment_knowledge"],
                    "education": resp["education"],
                    "family_status": resp.get("family_status", ""),
                    "question_number": qnum,
                    "question_text": q_def["text"],
                    "question_type": q_def["type"],
                    "selected_option": selected if q_def["type"] == "mc" else None,
                    "answer": ans.get("answer", ""),
                    "sentiment": ans.get("sentiment", "neutral"),
                    "confidence": ans.get("confidence", "medium"),
                    "key_themes": ans.get("key_themes", []),
                })

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df["age_group"] = pd.cut(df["age"], bins=[0, 34, 54, 100], labels=["18-34", "35-54", "55+"])
    df["income_bracket"] = pd.cut(
        df["income"], bins=[0, 60000, 120000, float("inf")],
        labels=["Under $60K", "$60K-$120K", "$120K+"]
    )

    # Per-question aggregation
    per_question = {}
    for qnum in range(1, len(questions) + 1):
        q_def = questions[qnum - 1]
        q_df = df[df["question_number"] == qnum]
        all_themes = []
        for themes in q_df["key_themes"]:
            if isinstance(themes, list):
                all_themes.extend(themes)

        pq = {
            "question": q_def["text"],
            "question_type": q_def["type"],
            "response_count": len(q_df),
            "sentiment_counts": q_df["sentiment"].value_counts().to_dict(),
            "confidence_counts": q_df["confidence"].value_counts().to_dict(),
            "top_themes": Counter(all_themes).most_common(15),
        }

        if q_def["type"] == "mc":
            option_counts = {}
            for opt in q_def["options"]:
                option_counts[opt] = int((q_df["selected_option"] == opt).sum())

            pq["option_counts"] = option_counts
            pq["options"] = q_def["options"]

        per_question[qnum] = pq

    return df, per_question


# ============================================================
# ANALYSIS - ADVISOR REACTOR
# ============================================================

def build_advisor_analysis(reactions):
    """Build analysis DataFrame and aggregation for advisor reactions."""
    valid = [r for r in reactions if "error" not in r]
    if not valid:
        return None, None

    df = pd.DataFrame(valid)

    df["interest_score"] = pd.to_numeric(df["interest_score"], errors="coerce")
    df["would_recommend"] = df.get("would_recommend", pd.Series([False]*len(df))).astype(bool)
    df["personal_income"] = pd.to_numeric(df.get("personal_income", 0), errors="coerce")
    df["years_in_business"] = pd.to_numeric(df.get("years_in_business", 0), errors="coerce")
    df["book_size_aum"] = pd.to_numeric(df.get("book_size_aum", 0), errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["age_group"] = pd.cut(df["age"], bins=[0, 34, 54, 100], labels=["Under 35", "35-54", "55+"])
    df["years_group"] = pd.cut(df["years_in_business"], bins=[-1, 5, 15, 100], labels=["Junior (0-5yr)", "Mid (6-15yr)", "Senior (16+yr)"])
    df["book_size_group"] = pd.cut(
        df["book_size_aum"], bins=[0, 50e6, 150e6, float("inf")],
        labels=["Under $50M", "$50M-$150M", "$150M+"]
    )
    df["income_bracket"] = pd.cut(
        df["personal_income"], bins=[0, 100000, 250000, float("inf")],
        labels=["Under $100K", "$100K-$250K", "$250K+"]
    )

    all_concerns = []
    all_appeals = []
    all_helps = []
    for _, row in df.iterrows():
        concerns = row.get("key_concerns", [])
        if isinstance(concerns, list):
            all_concerns.extend(concerns)
        appeals = row.get("appeal_factors", [])
        if isinstance(appeals, list):
            all_appeals.extend(appeals)
        wh = row.get("what_would_help", "")
        if isinstance(wh, str) and wh:
            all_helps.append(wh)

    analysis = {
        "top_concerns": Counter(all_concerns).most_common(12),
        "top_appeals": Counter(all_appeals).most_common(12),
        "what_would_help": all_helps,
    }

    return df, analysis


def build_advisor_survey_analysis(responses, questions):
    """Build survey analysis DataFrame for advisor personas."""
    valid = [r for r in responses if "error" not in r]
    if not valid:
        return None, None

    rows = []
    for resp in valid:
        answers = resp.get("answers", [])
        for ans in answers:
            qnum = ans.get("question_number", 0)
            if 1 <= qnum <= len(questions):
                q_def = questions[qnum - 1]
                selected = ans.get("selected_option")
                if q_def["type"] == "mc":
                    selected = _fuzzy_match_option(selected, q_def["options"])

                rows.append({
                    "persona_id": resp["persona_id"],
                    "persona_name": resp["persona_name"],
                    "age": resp.get("age", 0),
                    "gender": resp.get("gender", ""),
                    "province": resp.get("province", ""),
                    "city": resp.get("city", ""),
                    "firm_type": resp.get("firm_type", ""),
                    "years_in_business": resp.get("years_in_business", 0),
                    "designations": resp.get("designations", ""),
                    "book_size_aum": resp.get("book_size_aum", 0),
                    "practice_focus": resp.get("practice_focus", ""),
                    "compensation_model": resp.get("compensation_model", ""),
                    "personal_income": resp.get("personal_income", 0),
                    "business_maturity": resp.get("business_maturity", ""),
                    "client_demographics": resp.get("client_demographics", ""),
                    "education": resp.get("education", ""),
                    "family_status": resp.get("family_status", ""),
                    "question_number": qnum,
                    "question_text": q_def["text"],
                    "question_type": q_def["type"],
                    "selected_option": selected if q_def["type"] == "mc" else None,
                    "answer": ans.get("answer", ""),
                    "sentiment": ans.get("sentiment", "neutral"),
                    "confidence": ans.get("confidence", "medium"),
                    "key_themes": ans.get("key_themes", []),
                })

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["personal_income"] = pd.to_numeric(df["personal_income"], errors="coerce")
    df["years_in_business"] = pd.to_numeric(df["years_in_business"], errors="coerce")
    df["book_size_aum"] = pd.to_numeric(df["book_size_aum"], errors="coerce")
    df["years_group"] = pd.cut(df["years_in_business"], bins=[-1, 5, 15, 100], labels=["Junior (0-5yr)", "Mid (6-15yr)", "Senior (16+yr)"])
    df["book_size_group"] = pd.cut(
        df["book_size_aum"], bins=[0, 50e6, 150e6, float("inf")],
        labels=["Under $50M", "$50M-$150M", "$150M+"]
    )

    per_question = {}
    for qnum in range(1, len(questions) + 1):
        q_def = questions[qnum - 1]
        q_df = df[df["question_number"] == qnum]
        all_themes = []
        for themes in q_df["key_themes"]:
            if isinstance(themes, list):
                all_themes.extend(themes)

        pq = {
            "question": q_def["text"],
            "question_type": q_def["type"],
            "response_count": len(q_df),
            "sentiment_counts": q_df["sentiment"].value_counts().to_dict(),
            "confidence_counts": q_df["confidence"].value_counts().to_dict(),
            "top_themes": Counter(all_themes).most_common(15),
        }

        if q_def["type"] == "mc":
            option_counts = {}
            for opt in q_def["options"]:
                option_counts[opt] = int((q_df["selected_option"] == opt).sum())

            pq["option_counts"] = option_counts
            pq["options"] = q_def["options"]

        per_question[qnum] = pq

    return df, per_question



# ============================================================
# DISPLAY - REACTOR
# ============================================================

SENTIMENT_COLORS = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c", "mixed": "#f39c12"}


def show_overview(df):
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    avg_score = df["interest_score"].mean()
    invest_pct = df["would_invest"].mean() * 100
    top_sentiment = df["sentiment"].mode().iloc[0] if not df["sentiment"].mode().empty else "N/A"
    median_score = df["interest_score"].median()

    col1.metric("Avg Interest Score", f"{avg_score:.1f} / 10")
    col2.metric("Would Invest", f"{invest_pct:.0f}%")
    col3.metric("Top Sentiment", top_sentiment.title())
    col4.metric("Median Score", f"{median_score:.0f} / 10")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df, x="interest_score", nbins=10,
            title="Interest Score Distribution",
            labels={"interest_score": "Interest Score", "count": "Count"},
            color_discrete_sequence=["#2d5a87"],
        )
        fig.update_layout(bargap=0.1, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sent_counts = df["sentiment"].value_counts()
        fig = px.pie(
            values=sent_counts.values, names=sent_counts.index,
            title="Sentiment Breakdown",
            color=sent_counts.index,
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    if "investment_amount" in df.columns:
        amount_order = ["none", "small ($100-$1K)", "moderate ($1K-$10K)", "significant ($10K-$50K)", "major ($50K+)"]
        amt_counts = df["investment_amount"].value_counts()
        amt_df = pd.DataFrame({"Amount": amt_counts.index, "Count": amt_counts.values})
        amt_df["sort_key"] = amt_df["Amount"].apply(lambda x: amount_order.index(x) if x in amount_order else 99)
        amt_df = amt_df.sort_values("sort_key")
        fig = px.bar(
            amt_df, x="Amount", y="Count",
            title="Investment Amount Willingness",
            color_discrete_sequence=["#2d5a87"],
        )
        st.plotly_chart(fig, use_container_width=True)


def show_demographics(df):
    st.subheader("Breakdown by Demographics")

    c1, c2 = st.columns(2)
    with c1:
        age_stats = df.groupby("age_group", observed=True).agg(
            avg_score=("interest_score", "mean"),
            would_invest_pct=("would_invest", lambda x: x.mean() * 100),
            count=("interest_score", "count"),
        ).reset_index()
        fig = px.bar(
            age_stats, x="age_group", y="avg_score",
            title="Avg Interest Score by Age Group",
            text="avg_score", color_discrete_sequence=["#2d5a87"],
            labels={"age_group": "Age Group", "avg_score": "Avg Score"},
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            age_stats, x="age_group", y="would_invest_pct",
            title="% Would Invest by Age Group",
            text="would_invest_pct", color_discrete_sequence=["#27ae60"],
            labels={"age_group": "Age Group", "would_invest_pct": "% Would Invest"},
        )
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    risk_order = ["Very Conservative", "Conservative", "Moderate", "Growth", "Aggressive"]
    c1, c2 = st.columns(2)
    with c1:
        risk_stats = df.groupby("risk_tolerance_profile", observed=True).agg(
            avg_score=("interest_score", "mean"),
            count=("interest_score", "count"),
        ).reset_index()
        risk_stats["risk_tolerance_profile"] = pd.Categorical(risk_stats["risk_tolerance_profile"], categories=risk_order, ordered=True)
        risk_stats = risk_stats.sort_values("risk_tolerance_profile")
        fig = px.bar(
            risk_stats, x="risk_tolerance_profile", y="avg_score",
            title="Avg Interest Score by Risk Tolerance",
            text="avg_score", color_discrete_sequence=["#8e44ad"],
            labels={"risk_tolerance_profile": "Risk Tolerance", "avg_score": "Avg Score"},
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        income_stats = df.groupby("income_bracket", observed=True).agg(
            avg_score=("interest_score", "mean"),
            would_invest_pct=("would_invest", lambda x: x.mean() * 100),
            count=("interest_score", "count"),
        ).reset_index()
        fig = px.bar(
            income_stats, x="income_bracket", y="avg_score",
            title="Avg Interest Score by Income",
            text="avg_score", color_discrete_sequence=["#e67e22"],
            labels={"income_bracket": "Income Bracket", "avg_score": "Avg Score"},
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)

    prov_stats = df.groupby("province", observed=True).agg(
        avg_score=("interest_score", "mean"),
        would_invest_pct=("would_invest", lambda x: x.mean() * 100),
        count=("interest_score", "count"),
    ).reset_index().sort_values("count", ascending=False).head(6)
    fig = px.bar(
        prov_stats, x="province", y="avg_score",
        title="Avg Interest Score by Province (top 6 by sample size)",
        text="avg_score", color="would_invest_pct",
        color_continuous_scale="RdYlGn",
        labels={"province": "Province", "avg_score": "Avg Score", "would_invest_pct": "% Would Invest"},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        know_order = ["Beginner", "Some knowledge", "Knowledgeable", "Advanced", "Expert"]
        know_stats = df.groupby("investment_knowledge", observed=True).agg(
            avg_score=("interest_score", "mean"),
            count=("interest_score", "count"),
        ).reset_index()
        know_stats["investment_knowledge"] = pd.Categorical(know_stats["investment_knowledge"], categories=know_order, ordered=True)
        know_stats = know_stats.sort_values("investment_knowledge")
        fig = px.bar(
            know_stats, x="investment_knowledge", y="avg_score",
            title="Avg Score by Investment Knowledge",
            text="avg_score", color_discrete_sequence=["#16a085"],
            labels={"investment_knowledge": "Knowledge Level", "avg_score": "Avg Score"},
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        life_stats = df.groupby("life_stage", observed=True).agg(
            avg_score=("interest_score", "mean"),
            count=("interest_score", "count"),
        ).reset_index().sort_values("avg_score", ascending=True)
        fig = px.bar(
            life_stats, x="avg_score", y="life_stage",
            title="Avg Score by Life Stage",
            text="avg_score", color_discrete_sequence=["#2980b9"],
            labels={"life_stage": "Life Stage", "avg_score": "Avg Score"},
            orientation="h",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(xaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)


def show_insights(df, analysis):
    st.subheader("Key Insights")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Top Concerns**")
        if analysis["top_concerns"]:
            concern_df = pd.DataFrame(analysis["top_concerns"], columns=["Concern", "Mentions"])
            fig = px.bar(
                concern_df.iloc[::-1], x="Mentions", y="Concern",
                color_discrete_sequence=["#e74c3c"],
                title="Most Common Concerns",
                orientation="h",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Top Appeal Factors**")
        if analysis["top_appeals"]:
            appeal_df = pd.DataFrame(analysis["top_appeals"], columns=["Appeal", "Mentions"])
            fig = px.bar(
                appeal_df.iloc[::-1], x="Mentions", y="Appeal",
                color_discrete_sequence=["#2ecc71"],
                title="Most Common Appeal Factors",
                orientation="h",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**What Would Make Them More Likely to Invest**")
    if analysis["what_would_help"]:
        helps = Counter(analysis["what_would_help"]).most_common(15)
        for h, count in helps:
            st.markdown(f"- {h} *({count})*")


def show_quotes(df, key_suffix=""):
    st.subheader("Verbatim Reactions")

    c1, c2, c3 = st.columns(3)
    with c1:
        filter_sentiment = st.selectbox("Filter by sentiment", ["All", "positive", "negative", "neutral", "mixed"], key=f"quote_sent{key_suffix}")
    with c2:
        filter_invest = st.selectbox("Filter by would invest", ["All", "Yes", "No"], key=f"quote_invest{key_suffix}")
    with c3:
        sort_by = st.selectbox("Sort by", ["Interest Score (High to Low)", "Interest Score (Low to High)", "Random"], key=f"quote_sort{key_suffix}")

    filtered = df.copy()
    if filter_sentiment != "All":
        filtered = filtered[filtered["sentiment"] == filter_sentiment]
    if filter_invest == "Yes":
        filtered = filtered[filtered["would_invest"] == True]
    elif filter_invest == "No":
        filtered = filtered[filtered["would_invest"] == False]

    if sort_by == "Interest Score (High to Low)":
        filtered = filtered.sort_values("interest_score", ascending=False)
    elif sort_by == "Interest Score (Low to High)":
        filtered = filtered.sort_values("interest_score", ascending=True)
    else:
        filtered = filtered.sample(frac=1, random_state=42)

    st.caption(f"Showing {len(filtered)} reactions")

    for _, row in filtered.head(20).iterrows():
        score = row.get("interest_score", "?")
        sentiment = row.get("sentiment", "?")
        emoji = {"positive": ":green_circle:", "negative": ":red_circle:", "neutral": ":white_circle:", "mixed": ":orange_circle:"}.get(sentiment, ":white_circle:")

        with st.expander(f"{emoji} **{row['persona_name']}** | Age {row['age']}, {row['province']} | Score: {score}/10 | {sentiment}"):
            st.markdown(f"**Gut reaction:** {row.get('gut_reaction', 'N/A')}")
            st.markdown(f"**Would invest:** {'Yes' if row.get('would_invest') else 'No'} | **Amount:** {row.get('investment_amount', 'N/A')}")

            quote = row.get("verbatim_quote", "N/A")
            st.markdown(f'> *"{quote}"*')

            concerns = row.get("key_concerns", [])
            if isinstance(concerns, list) and concerns:
                st.markdown(f"**Concerns:** {', '.join(concerns)}")

            appeals = row.get("appeal_factors", [])
            if isinstance(appeals, list) and appeals:
                st.markdown(f"**Appeals:** {', '.join(appeals)}")

            st.markdown(f"**What would help:** {row.get('what_would_help', 'N/A')}")
            st.caption(f"{row.get('life_stage', '')} | {row.get('education', '')} | Income: ${row.get('income', 0):,} | {row.get('risk_tolerance_profile', '')} risk | {row.get('investment_knowledge', '')} investor")


def show_data(df, key_suffix=""):
    st.subheader("Raw Data")

    display_cols = [
        "persona_name", "age", "gender", "province", "income", "net_worth",
        "risk_tolerance_profile", "interest_score", "sentiment", "would_invest",
        "investment_amount", "gut_reaction",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].sort_values("interest_score", ascending=False), use_container_width=True, height=500, key=f"data_table{key_suffix}")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Results as CSV",
        data=csv,
        file_name=f"reactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key=f"download_csv{key_suffix}",
    )



# ============================================================
# DISPLAY - SURVEY
# ============================================================

def show_survey_overview(df, per_question, questions):
    st.subheader("Survey Overview")

    n_open = sum(1 for q in questions if q["type"] == "open")
    n_mc = sum(1 for q in questions if q["type"] == "mc")

    col1, col2, col3 = st.columns(3)
    n_respondents = df["persona_id"].nunique()
    col1.metric("Respondents", n_respondents)
    col2.metric("Questions", f"{len(questions)} ({n_open} open, {n_mc} MC)")
    top_sentiment = df["sentiment"].mode().iloc[0] if not df["sentiment"].mode().empty else "N/A"
    col3.metric("Overall Sentiment", top_sentiment.title())

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        sent_counts = df["sentiment"].value_counts()
        fig = px.pie(
            values=sent_counts.values, names=sent_counts.index,
            title="Overall Sentiment (all answers)",
            color=sent_counts.index,
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        all_themes = []
        for themes_list in df["key_themes"]:
            if isinstance(themes_list, list):
                all_themes.extend(themes_list)
        theme_counts = Counter(all_themes).most_common(12)
        if theme_counts:
            theme_df = pd.DataFrame(theme_counts, columns=["Theme", "Mentions"])
            fig = px.bar(
                theme_df.iloc[::-1], x="Mentions", y="Theme",
                orientation="h", title="Top Themes Across All Questions",
                color_discrete_sequence=["#2d5a87"],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Sentiment by question heatmap
    st.subheader("Sentiment by Question")
    heatmap_data = []
    q_labels = []
    for qnum in range(1, len(questions) + 1):
        qa = per_question.get(qnum, {})
        sc = qa.get("sentiment_counts", {})
        q_type_tag = "(MC)" if qa.get("question_type") == "mc" else "(Open)"
        label = f"Q{qnum} {q_type_tag}: {questions[qnum-1]['text'][:55]}"
        q_labels.append(label)
        heatmap_data.append([
            sc.get("positive", 0),
            sc.get("neutral", 0),
            sc.get("mixed", 0),
            sc.get("negative", 0),
        ])
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=["Positive", "Neutral", "Mixed", "Negative"],
        y=q_labels,
        colorscale="RdYlGn",
        texttemplate="%{z}",
        hovertemplate="Question: %{y}<br>%{x}: %{z}<extra></extra>",
    ))
    fig.update_layout(height=max(250, len(questions) * 50), yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    # MC option distribution summary
    mc_questions = [(qnum, qa) for qnum, qa in per_question.items() if qa.get("question_type") == "mc"]
    if mc_questions:
        st.subheader("Multiple-Choice Summary")
        for qnum, qa in mc_questions:
            option_counts = qa.get("option_counts", {})
            if option_counts:
                opt_df = pd.DataFrame(
                    list(option_counts.items()), columns=["Option", "Count"]
                )
                fig = px.bar(
                    opt_df, x="Count", y="Option", orientation="h",
                    title=f"Q{qnum}: {qa['question'][:70]}",
                    color_discrete_sequence=["#2d5a87"],
                )
                fig.update_layout(height=max(200, len(opt_df) * 40))
                st.plotly_chart(fig, use_container_width=True, key=f"overview_mc_{qnum}")


def show_per_question_analysis(df, per_question, questions):
    st.subheader("Per-Question Breakdown")

    for qnum in range(1, len(questions) + 1):
        qa = per_question.get(qnum, {})
        q_def = questions[qnum - 1]
        q_df = df[df["question_number"] == qnum]
        q_type = qa.get("question_type", "open")
        type_badge = "Multiple Choice" if q_type == "mc" else "Open-Ended"

        with st.expander(f"Q{qnum} ({type_badge}): {q_def['text']}", expanded=(qnum == 1)):

            # === MC PRIMARY: Option Distribution ===
            if q_type == "mc":
                option_counts = qa.get("option_counts", {})
                if option_counts:
                    c1, c2 = st.columns(2)
                    with c1:
                        opt_df = pd.DataFrame(
                            list(option_counts.items()), columns=["Option", "Count"]
                        )
                        fig = px.pie(
                            opt_df, values="Count", names="Option",
                            title="Option Distribution",
                        )
                        fig.update_traces(textposition="inside", textinfo="percent+label")
                        st.plotly_chart(fig, use_container_width=True, key=f"sq_opt_pie_{qnum}")

                    with c2:
                        fig = px.bar(
                            opt_df, x="Option", y="Count",
                            title="Option Counts",
                            color_discrete_sequence=["#2d5a87"],
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"sq_opt_bar_{qnum}")

                # Option by age group
                st.markdown("**Option Selection by Age Group**")
                age_opt = q_df.groupby(["age_group", "selected_option"], observed=True).size().reset_index(name="count")
                if not age_opt.empty:
                    fig = px.bar(
                        age_opt, x="age_group", y="count", color="selected_option",
                        barmode="group",
                        labels={"age_group": "Age Group", "count": "Count", "selected_option": "Option"},
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"sq_opt_age_{qnum}")

                # Option by income bracket
                st.markdown("**Option Selection by Income Bracket**")
                inc_opt = q_df.groupby(["income_bracket", "selected_option"], observed=True).size().reset_index(name="count")
                if not inc_opt.empty:
                    fig = px.bar(
                        inc_opt, x="income_bracket", y="count", color="selected_option",
                        barmode="group",
                        labels={"income_bracket": "Income Bracket", "count": "Count", "selected_option": "Option"},
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"sq_opt_inc_{qnum}")

                st.markdown("---")
                st.markdown("**Sentiment & Confidence (from explanations)**")

            # === Sentiment & Confidence (shown for ALL question types) ===
            c1, c2 = st.columns(2)
            with c1:
                sent_counts = q_df["sentiment"].value_counts()
                fig = px.pie(
                    values=sent_counts.values, names=sent_counts.index,
                    title="Sentiment",
                    color=sent_counts.index,
                    color_discrete_map=SENTIMENT_COLORS,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True, key=f"sq_sent_{qnum}")

            with c2:
                conf_order = ["low", "medium", "high"]
                conf_counts = q_df["confidence"].value_counts()
                conf_df = pd.DataFrame({"Confidence": conf_counts.index, "Count": conf_counts.values})
                conf_df["sort_key"] = conf_df["Confidence"].apply(lambda x: conf_order.index(x) if x in conf_order else 99)
                conf_df = conf_df.sort_values("sort_key")
                fig = px.bar(
                    conf_df, x="Confidence", y="Count",
                    title="Confidence Level",
                    color_discrete_sequence=["#8e44ad"],
                )
                st.plotly_chart(fig, use_container_width=True, key=f"sq_conf_{qnum}")

            # Themes
            if qa.get("top_themes"):
                themes_df = pd.DataFrame(qa["top_themes"], columns=["Theme", "Count"])
                fig = px.bar(
                    themes_df.iloc[::-1], x="Count", y="Theme",
                    orientation="h", title="Top Themes",
                    color_discrete_sequence=["#2d5a87"],
                )
                fig.update_layout(height=max(300, len(themes_df) * 28))
                st.plotly_chart(fig, use_container_width=True, key=f"sq_themes_{qnum}")

            # Rich open-ended analysis (AI summary, word cloud, key phrases, sentiment quotes)
            if q_type == "open":
                st.markdown("---")
                render_open_ended_panel(q_def["text"], q_df,
                                         audience_label="Canadians",
                                         key_prefix=f"sq_consumer_{qnum}")
                st.markdown("---")

            # Sentiment by demographics (for open-ended questions)
            if q_type == "open":
                st.markdown("**Sentiment by Age Group**")
                age_sent = q_df.groupby(["age_group", "sentiment"], observed=True).size().reset_index(name="count")
                if not age_sent.empty:
                    fig = px.bar(
                        age_sent, x="age_group", y="count", color="sentiment",
                        barmode="group", color_discrete_map=SENTIMENT_COLORS,
                        labels={"age_group": "Age Group", "count": "Count"},
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"sq_age_{qnum}")

                st.markdown("**Sentiment by Income Bracket**")
                inc_sent = q_df.groupby(["income_bracket", "sentiment"], observed=True).size().reset_index(name="count")
                if not inc_sent.empty:
                    fig = px.bar(
                        inc_sent, x="income_bracket", y="count", color="sentiment",
                        barmode="group", color_discrete_map=SENTIMENT_COLORS,
                        labels={"income_bracket": "Income Bracket", "count": "Count"},
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"sq_inc_{qnum}")


def show_survey_responses(df, questions):
    st.subheader("Individual Responses")

    q_labels = []
    for i, q in enumerate(questions, 1):
        type_tag = "(MC)" if q["type"] == "mc" else "(Open)"
        q_labels.append(f"Q{i} {type_tag}: {q['text']}")

    selected_q = st.selectbox(
        "Select question",
        q_labels,
        key="survey_q_select",
    )
    qnum = int(selected_q.split(":")[0].strip().split()[0][1:])
    q_def = questions[qnum - 1]
    q_df = df[df["question_number"] == qnum].copy()

    # Filters - add option filter for MC questions
    if q_def["type"] == "mc":
        c1, c2, c3 = st.columns(3)
        with c1:
            filter_sentiment = st.selectbox("Filter by sentiment", ["All", "positive", "negative", "neutral", "mixed"], key="survey_sent_filter")
        with c2:
            filter_confidence = st.selectbox("Filter by confidence", ["All", "high", "medium", "low"], key="survey_conf_filter")
        with c3:
            options_list = ["All"] + q_def["options"]
            filter_option = st.selectbox("Filter by selection", options_list, key="survey_opt_filter")
        if filter_option != "All":
            q_df = q_df[q_df["selected_option"] == filter_option]
    else:
        c1, c2 = st.columns(2)
        with c1:
            filter_sentiment = st.selectbox("Filter by sentiment", ["All", "positive", "negative", "neutral", "mixed"], key="survey_sent_filter")
        with c2:
            filter_confidence = st.selectbox("Filter by confidence", ["All", "high", "medium", "low"], key="survey_conf_filter")

    if filter_sentiment != "All":
        q_df = q_df[q_df["sentiment"] == filter_sentiment]
    if filter_confidence != "All":
        q_df = q_df[q_df["confidence"] == filter_confidence]

    st.caption(f"Showing {len(q_df)} responses")

    for _, row in q_df.head(30).iterrows():
        sentiment = row.get("sentiment", "neutral")
        confidence = row.get("confidence", "medium")
        emoji = {"positive": ":green_circle:", "negative": ":red_circle:", "neutral": ":white_circle:", "mixed": ":orange_circle:"}.get(sentiment, ":white_circle:")

        with st.expander(f"{emoji} **{row['persona_name']}** | Age {row['age']}, {row['province']} | {sentiment} | {confidence} confidence"):
            if q_def["type"] == "mc" and pd.notna(row.get("selected_option")):
                st.markdown(f"**Selected:** :ballot_box_with_check: **{row['selected_option']}**")
                st.markdown(f"**Explanation:** {row['answer']}")
            else:
                st.markdown(f"**Answer:** {row['answer']}")

            themes = row.get("key_themes", [])
            if isinstance(themes, list) and themes:
                st.markdown(f"**Themes:** {', '.join(themes)}")

            st.caption(f"{row.get('life_stage', '')} | {row.get('education', '')} | Income: ${row.get('income', 0):,} | {row.get('family_status', '')}")


def show_survey_data(df):
    st.subheader("Raw Data")

    display_df = df.copy()
    display_df["key_themes"] = display_df["key_themes"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )

    display_cols = [
        "question_number", "question_type", "question_text", "persona_name", "age", "gender",
        "province", "income", "life_stage", "selected_option", "answer", "sentiment",
        "confidence", "key_themes",
    ]
    available = [c for c in display_cols if c in display_df.columns]
    st.dataframe(display_df[available], use_container_width=True, height=500)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Survey Results as CSV",
        data=csv,
        file_name=f"survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ============================================================
# DISPLAY - ADVISOR REACTOR
# ============================================================

SUITABILITY_ORDER = ["unsuitable", "niche", "moderate", "broad"]
SUITABILITY_COLORS = {"unsuitable": "#e74c3c", "niche": "#f39c12", "moderate": "#3498db", "broad": "#2ecc71"}


def show_advisor_overview(df, key_suffix=""):
    """Overview metrics for advisor reactions."""
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    avg_score = df["interest_score"].mean()
    recommend_pct = df["would_recommend"].mean() * 100 if "would_recommend" in df.columns else 0
    top_sentiment = df["sentiment"].mode().iloc[0] if len(df) > 0 else "N/A"
    median_score = df["interest_score"].median()

    col1.metric("Avg Interest Score", f"{avg_score:.1f}/10")
    col2.metric("Would Recommend", f"{recommend_pct:.0f}%")
    col3.metric("Top Sentiment", top_sentiment.capitalize())
    col4.metric("Median Score", f"{median_score:.1f}/10")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="interest_score", nbins=10, title="Interest Score Distribution",
                           color_discrete_sequence=["#2d5a87"])
        fig.update_layout(xaxis_title="Score", yaxis_title="Count", bargap=0.1)
        st.plotly_chart(fig, use_container_width=True, key=f"adv_score_hist{key_suffix}")

    with c2:
        sent_counts = df["sentiment"].value_counts()
        colors = [SENTIMENT_COLORS.get(s, "#95a5a6") for s in sent_counts.index]
        fig = go.Figure(data=[go.Pie(labels=sent_counts.index, values=sent_counts.values,
                                      marker=dict(colors=colors), hole=0.4)])
        fig.update_layout(title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True, key=f"adv_sent_pie{key_suffix}")

    # Client suitability bar chart
    if "client_suitability" in df.columns:
        suit_counts = df["client_suitability"].str.lower().value_counts()
        ordered = [s for s in SUITABILITY_ORDER if s in suit_counts.index]
        colors = [SUITABILITY_COLORS.get(s, "#95a5a6") for s in ordered]
        fig = go.Figure(data=[go.Bar(
            x=ordered, y=[suit_counts[s] for s in ordered],
            marker_color=colors,
        )])
        fig.update_layout(title="Client Suitability Assessment", xaxis_title="Suitability", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True, key=f"adv_suit_bar{key_suffix}")


def show_advisor_demographics(df, key_suffix=""):
    """Demographic breakdowns for advisor reactions."""
    st.subheader("Professional Breakdowns")

    # By Years in Business
    c1, c2 = st.columns(2)
    with c1:
        if "years_group" in df.columns:
            group_data = df.groupby("years_group", observed=True).agg(
                avg_score=("interest_score", "mean"),
                recommend_pct=("would_recommend", lambda x: x.mean() * 100),
                count=("interest_score", "size"),
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=group_data["years_group"], y=group_data["avg_score"],
                                  name="Avg Score", marker_color="#2d5a87"))
            fig.update_layout(title="Interest by Experience Level", yaxis_title="Avg Score")
            st.plotly_chart(fig, use_container_width=True, key=f"adv_yrs_score{key_suffix}")

    with c2:
        if "years_group" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=group_data["years_group"], y=group_data["recommend_pct"],
                                  name="% Recommend", marker_color="#27ae60"))
            fig.update_layout(title="% Would Recommend by Experience", yaxis_title="% Recommend")
            st.plotly_chart(fig, use_container_width=True, key=f"adv_yrs_rec{key_suffix}")

    # By Firm Type
    c1, c2 = st.columns(2)
    with c1:
        if "firm_type" in df.columns:
            firm_data = df.groupby("firm_type").agg(
                avg_score=("interest_score", "mean"),
                count=("interest_score", "size"),
            ).sort_values("avg_score", ascending=True).reset_index()
            fig = px.bar(firm_data, y="firm_type", x="avg_score", orientation="h",
                         title="Interest by Firm Type", color_discrete_sequence=["#2d5a87"])
            fig.update_layout(xaxis_title="Avg Score", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True, key=f"adv_firm_score{key_suffix}")

    with c2:
        if "book_size_group" in df.columns:
            book_data = df.groupby("book_size_group", observed=True).agg(
                avg_score=("interest_score", "mean"),
                recommend_pct=("would_recommend", lambda x: x.mean() * 100),
                count=("interest_score", "size"),
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=book_data["book_size_group"], y=book_data["avg_score"],
                                  name="Avg Score", marker_color="#8e44ad"))
            fig.update_layout(title="Interest by Book Size", yaxis_title="Avg Score")
            st.plotly_chart(fig, use_container_width=True, key=f"adv_book_score{key_suffix}")

    # By Province and Practice Focus
    c1, c2 = st.columns(2)
    with c1:
        if "province" in df.columns:
            prov_data = df.groupby("province").agg(
                avg_score=("interest_score", "mean"),
                count=("interest_score", "size"),
            ).sort_values("count", ascending=False).head(6).sort_values("avg_score", ascending=True).reset_index()
            fig = px.bar(prov_data, y="province", x="avg_score", orientation="h",
                         title="Interest by Province (Top 6)", color_discrete_sequence=["#16a085"])
            fig.update_layout(xaxis_title="Avg Score", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True, key=f"adv_prov_score{key_suffix}")

    with c2:
        if "practice_focus" in df.columns:
            focus_data = df.groupby("practice_focus").agg(
                avg_score=("interest_score", "mean"),
                count=("interest_score", "size"),
            ).sort_values("avg_score", ascending=True).reset_index()
            fig = px.bar(focus_data, y="practice_focus", x="avg_score", orientation="h",
                         title="Interest by Practice Focus", color_discrete_sequence=["#e67e22"])
            fig.update_layout(xaxis_title="Avg Score", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True, key=f"adv_focus_score{key_suffix}")


def show_advisor_quotes(df, key_suffix=""):
    """Show advisor verbatim quotes with filters."""
    st.subheader("Advisor Verbatim Reactions")
    c1, c2, c3 = st.columns(3)
    with c1:
        filter_sentiment = st.selectbox("Filter by Sentiment", ["All"] + list(SENTIMENT_COLORS.keys()), key=f"adv_quote_sent{key_suffix}")
    with c2:
        filter_recommend = st.selectbox("Filter by Recommendation", ["All", "Would Recommend", "Would Not Recommend"], key=f"adv_quote_rec{key_suffix}")
    with c3:
        sort_by = st.selectbox("Sort by", ["Interest Score (High-Low)", "Interest Score (Low-High)", "Years in Business"], key=f"adv_quote_sort{key_suffix}")

    filtered = df.copy()
    if filter_sentiment != "All":
        filtered = filtered[filtered["sentiment"] == filter_sentiment]
    if filter_recommend == "Would Recommend" and "would_recommend" in filtered.columns:
        filtered = filtered[filtered["would_recommend"] == True]
    elif filter_recommend == "Would Not Recommend" and "would_recommend" in filtered.columns:
        filtered = filtered[filtered["would_recommend"] == False]

    if sort_by == "Interest Score (High-Low)":
        filtered = filtered.sort_values("interest_score", ascending=False)
    elif sort_by == "Interest Score (Low-High)":
        filtered = filtered.sort_values("interest_score", ascending=True)
    elif "years_in_business" in filtered.columns:
        filtered = filtered.sort_values("years_in_business", ascending=False)

    for _, row in filtered.head(20).iterrows():
        score = row.get("interest_score", "?")
        sentiment = row.get("sentiment", "neutral")
        emoji = {"positive": "thumbsup", "negative": "thumbsdown", "mixed": "thinking_face", "neutral": "neutral_face"}.get(sentiment, "neutral_face")
        recommend = "Yes" if row.get("would_recommend", False) else "No"
        suitability = row.get("client_suitability", "N/A")

        with st.expander(f":{emoji}: **{row.get('persona_name', 'Unknown')}** — Score: {score}/10 | Recommend: {recommend} | Suitability: {suitability}"):
            firm = row.get("firm_type", "")
            desig = row.get("designations", "")
            yrs = row.get("years_in_business", "")
            aum = row.get("book_size_aum", 0)
            aum_str = f"${aum/1e6:.0f}M" if aum and aum > 0 else "N/A"
            focus = row.get("practice_focus", "")
            st.caption(f"{firm} | {desig} | {yrs} years | Book: {aum_str} | {focus}")

            st.markdown(f"**Gut reaction:** {row.get('gut_reaction', 'N/A')}")
            st.markdown(f"**Verbatim:** \"{row.get('verbatim_quote', 'N/A')}\"")
            concerns = row.get("key_concerns", [])
            if isinstance(concerns, list) and concerns:
                st.markdown(f"**Concerns:** {', '.join(concerns)}")
            appeals = row.get("appeal_factors", [])
            if isinstance(appeals, list) and appeals:
                st.markdown(f"**Appeals:** {', '.join(appeals)}")
            st.markdown(f"**What would help:** {row.get('what_would_help', 'N/A')}")


def show_advisor_data(df, key_suffix=""):
    """Show raw advisor data table."""
    st.subheader("Raw Data")
    display_cols = [
        "persona_name", "age", "province", "firm_type", "designations",
        "years_in_business", "book_size_aum", "practice_focus",
        "interest_score", "sentiment", "would_recommend",
        "client_suitability", "gut_reaction",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True, height=500, key=f"adv_data_table{key_suffix}")

    csv = df[available].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"advisor_reactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key=f"adv_download_csv{key_suffix}",
    )


def show_advisor_ab_comparison(df_a, df_b, analysis_a, analysis_b, idea_a, idea_b, key_suffix=""):
    """Head-to-head comparison for advisor A/B test."""
    st.subheader("Head-to-Head Comparison")

    # Key metrics
    col_a, col_vs, col_b = st.columns([2, 1, 2])
    avg_a = df_a["interest_score"].mean()
    avg_b = df_b["interest_score"].mean()
    rec_a = df_a["would_recommend"].mean() * 100 if "would_recommend" in df_a.columns else 0
    rec_b = df_b["would_recommend"].mean() * 100 if "would_recommend" in df_b.columns else 0
    sent_a = df_a["sentiment"].mode().iloc[0] if len(df_a) > 0 else "N/A"
    sent_b = df_b["sentiment"].mode().iloc[0] if len(df_b) > 0 else "N/A"
    med_a = df_a["interest_score"].median()
    med_b = df_b["interest_score"].median()

    with col_a:
        st.markdown(f"### Variant A")
        st.metric("Avg Score", f"{avg_a:.1f}", delta=f"{avg_a - avg_b:+.1f}" if avg_a != avg_b else None)
        st.metric("% Recommend", f"{rec_a:.0f}%", delta=f"{rec_a - rec_b:+.0f}%" if rec_a != rec_b else None)
        st.metric("Top Sentiment", sent_a.capitalize())
        st.metric("Median Score", f"{med_a:.1f}")
    with col_vs:
        st.markdown("### vs")
    with col_b:
        st.markdown(f"### Variant B")
        st.metric("Avg Score", f"{avg_b:.1f}", delta=f"{avg_b - avg_a:+.1f}" if avg_a != avg_b else None)
        st.metric("% Recommend", f"{rec_b:.0f}%", delta=f"{rec_b - rec_a:+.0f}%" if rec_a != rec_b else None)
        st.metric("Top Sentiment", sent_b.capitalize())
        st.metric("Median Score", f"{med_b:.1f}")

    st.divider()

    # Score distribution overlay
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_a["interest_score"], name="Variant A", opacity=0.6, marker_color="#2d5a87", nbinsx=10))
    fig.add_trace(go.Histogram(x=df_b["interest_score"], name="Variant B", opacity=0.6, marker_color="#e74c3c", nbinsx=10))
    fig.update_layout(title="Interest Score Distribution", barmode="overlay", xaxis_title="Score", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True, key=f"adv_ab_hist{key_suffix}")

    # Side-by-side sentiment
    c1, c2 = st.columns(2)
    for col, dfl, label in [(c1, df_a, "Variant A"), (c2, df_b, "Variant B")]:
        with col:
            sent = dfl["sentiment"].value_counts()
            colors = [SENTIMENT_COLORS.get(s, "#95a5a6") for s in sent.index]
            fig = go.Figure(data=[go.Pie(labels=sent.index, values=sent.values,
                                          marker=dict(colors=colors), hole=0.4)])
            fig.update_layout(title=f"Sentiment - {label}")
            st.plotly_chart(fig, use_container_width=True, key=f"adv_ab_sent_{label}{key_suffix}")

    # By years group
    if "years_group" in df_a.columns and "years_group" in df_b.columns:
        grp_a = df_a.groupby("years_group", observed=True)["would_recommend"].mean() * 100
        grp_b = df_b.groupby("years_group", observed=True)["would_recommend"].mean() * 100
        cats = sorted(set(grp_a.index) | set(grp_b.index), key=str)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cats, y=[grp_a.get(c, 0) for c in cats], name="Variant A", marker_color="#2d5a87"))
        fig.add_trace(go.Bar(x=cats, y=[grp_b.get(c, 0) for c in cats], name="Variant B", marker_color="#e74c3c"))
        fig.update_layout(title="% Would Recommend by Experience Level", barmode="group", yaxis_title="% Recommend")
        st.plotly_chart(fig, use_container_width=True, key=f"adv_ab_yrs{key_suffix}")

    # By firm type
    if "firm_type" in df_a.columns and "firm_type" in df_b.columns:
        grp_a = df_a.groupby("firm_type")["would_recommend"].mean() * 100
        grp_b = df_b.groupby("firm_type")["would_recommend"].mean() * 100
        cats = sorted(set(grp_a.index) | set(grp_b.index))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cats, y=[grp_a.get(c, 0) for c in cats], name="Variant A", marker_color="#2d5a87"))
        fig.add_trace(go.Bar(x=cats, y=[grp_b.get(c, 0) for c in cats], name="Variant B", marker_color="#e74c3c"))
        fig.update_layout(title="% Would Recommend by Firm Type", barmode="group", yaxis_title="% Recommend", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True, key=f"adv_ab_firm{key_suffix}")

    # By book size
    if "book_size_group" in df_a.columns and "book_size_group" in df_b.columns:
        grp_a = df_a.groupby("book_size_group", observed=True)["would_recommend"].mean() * 100
        grp_b = df_b.groupby("book_size_group", observed=True)["would_recommend"].mean() * 100
        cats = sorted(set(grp_a.index) | set(grp_b.index), key=str)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cats, y=[grp_a.get(c, 0) for c in cats], name="Variant A", marker_color="#2d5a87"))
        fig.add_trace(go.Bar(x=cats, y=[grp_b.get(c, 0) for c in cats], name="Variant B", marker_color="#e74c3c"))
        fig.update_layout(title="% Would Recommend by Book Size", barmode="group", yaxis_title="% Recommend")
        st.plotly_chart(fig, use_container_width=True, key=f"adv_ab_book{key_suffix}")

    # Concerns & Appeals comparison
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Concerns**")
        for label_name, analysis, color in [("A", analysis_a, "#2d5a87"), ("B", analysis_b, "#e74c3c")]:
            concerns = analysis.get("top_concerns", [])[:6]
            if concerns:
                items, counts = zip(*concerns)
                fig = px.bar(x=list(counts), y=list(items), orientation="h",
                             title=f"Variant {label_name}", color_discrete_sequence=[color])
                fig.update_layout(xaxis_title="Count", yaxis_title="", height=250)
                st.plotly_chart(fig, use_container_width=True, key=f"adv_ab_concerns_{label_name}{key_suffix}")
    with c2:
        st.markdown("**Top Appeals**")
        for label_name, analysis, color in [("A", analysis_a, "#2d5a87"), ("B", analysis_b, "#e74c3c")]:
            appeals = analysis.get("top_appeals", [])[:6]
            if appeals:
                items, counts = zip(*appeals)
                fig = px.bar(x=list(counts), y=list(items), orientation="h",
                             title=f"Variant {label_name}", color_discrete_sequence=[color])
                fig.update_layout(xaxis_title="Count", yaxis_title="", height=250)
                st.plotly_chart(fig, use_container_width=True, key=f"adv_ab_appeals_{label_name}{key_suffix}")

    # Statistical significance
    st.divider()
    st.subheader("Statistical Summary")
    diff_pct = abs(avg_a - avg_b) / max(avg_a, avg_b) * 100 if max(avg_a, avg_b) > 0 else 0
    winner = "A" if avg_a > avg_b else ("B" if avg_b > avg_a else "Tie")
    st.markdown(f"Variant A scored **{avg_a:.2f}** vs Variant B's **{avg_b:.2f}** — a **{diff_pct:.1f}%** difference. "
                f"{'Variant ' + winner + ' leads.' if winner != 'Tie' else 'It is a tie.'}")
    try:
        from scipy.stats import ttest_rel
        merged = df_a[["persona_id", "interest_score"]].merge(
            df_b[["persona_id", "interest_score"]], on="persona_id", suffixes=("_a", "_b")
        )
        if len(merged) > 1:
            t_stat, p_value = ttest_rel(merged["interest_score_a"], merged["interest_score_b"])
            sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
            st.markdown(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f} ({sig} at p<0.05, n={len(merged)} paired observations)")
    except ImportError:
        st.caption(f"Sample size: {len(df_a)} advisors per variant. Install scipy for statistical significance testing.")


def show_advisor_survey_overview(df, per_question, key_suffix=""):
    """Survey overview for advisor responses."""
    st.subheader("Survey Overview")
    total_responses = df["persona_id"].nunique()
    num_questions = len(per_question)
    st.markdown(f"**{total_responses}** advisors responded to **{num_questions}** question{'s' if num_questions > 1 else ''}")

    # MC summary
    mc_questions = {k: v for k, v in per_question.items() if v.get("question_type") == "mc"}
    if mc_questions:
        st.markdown("#### Multiple Choice Summary")
        for qnum, pq in sorted(mc_questions.items()):
            oc = pq.get("option_counts", {})
            if oc:
                top_option = max(oc, key=oc.get)
                top_pct = oc[top_option] / sum(oc.values()) * 100 if sum(oc.values()) > 0 else 0
                st.markdown(f"**Q{qnum}.** {pq['question']}")
                st.markdown(f"  Top answer: **{top_option}** ({top_pct:.0f}%)")

    # Sentiment heatmap
    if num_questions > 1:
        heatmap_data = []
        q_labels = []
        sentiments = ["positive", "neutral", "negative", "mixed"]
        for qnum in sorted(per_question.keys()):
            pq = per_question[qnum]
            q_type = "MC" if pq.get("question_type") == "mc" else "Open"
            q_labels.append(f"Q{qnum} [{q_type}]: {pq['question'][:40]}...")
            counts = pq.get("sentiment_counts", {})
            total = sum(counts.values()) or 1
            heatmap_data.append([counts.get(s, 0) / total * 100 for s in sentiments])

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data, x=sentiments, y=q_labels,
            colorscale="RdYlGn", text=[[f"{v:.0f}%" for v in row] for row in heatmap_data],
            texttemplate="%{text}", hovertemplate="Q: %{y}<br>Sentiment: %{x}<br>%{text}<extra></extra>",
        ))
        fig.update_layout(title="Sentiment Heatmap by Question", height=max(300, 50 * num_questions))
        st.plotly_chart(fig, use_container_width=True, key=f"adv_survey_heatmap{key_suffix}")


def show_advisor_per_question(df, per_question, key_suffix=""):
    """Per-question analysis for advisor survey responses."""
    st.subheader("Per-Question Analysis")
    for qnum in sorted(per_question.keys()):
        pq = per_question[qnum]
        q_type = pq.get("question_type", "open")
        st.markdown(f"### Q{qnum}: {pq['question']}")
        q_df = df[df["question_number"] == qnum]

        if q_type == "mc" and "option_counts" in pq:
            c1, c2 = st.columns(2)
            with c1:
                oc = pq["option_counts"]
                fig = go.Figure(data=[go.Pie(labels=list(oc.keys()), values=list(oc.values()), hole=0.4)])
                fig.update_layout(title="Option Distribution")
                st.plotly_chart(fig, use_container_width=True, key=f"adv_sq_pie_{qnum}{key_suffix}")
            with c2:
                if "firm_type" in q_df.columns:
                    ct = pd.crosstab(q_df["firm_type"], q_df["selected_option"])
                    fig = px.bar(ct, barmode="group", title="Options by Firm Type")
                    fig.update_layout(xaxis_title="Firm Type", yaxis_title="Count", xaxis_tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True, key=f"adv_sq_firm_{qnum}{key_suffix}")

            # By years group
            if "years_group" in q_df.columns:
                ct = pd.crosstab(q_df["years_group"], q_df["selected_option"])
                fig = px.bar(ct, barmode="group", title="Options by Experience Level")
                fig.update_layout(xaxis_title="Experience", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True, key=f"adv_sq_yrs_{qnum}{key_suffix}")
        else:
            # Open-ended: sentiment and confidence
            c1, c2 = st.columns(2)
            with c1:
                sent = pq.get("sentiment_counts", {})
                if sent:
                    colors = [SENTIMENT_COLORS.get(s, "#95a5a6") for s in sent.keys()]
                    fig = go.Figure(data=[go.Pie(labels=list(sent.keys()), values=list(sent.values()),
                                                  marker=dict(colors=colors), hole=0.4)])
                    fig.update_layout(title="Sentiment")
                    st.plotly_chart(fig, use_container_width=True, key=f"adv_sq_sent_{qnum}{key_suffix}")
            with c2:
                conf = pq.get("confidence_counts", {})
                if conf:
                    fig = go.Figure(data=[go.Pie(labels=list(conf.keys()), values=list(conf.values()), hole=0.4)])
                    fig.update_layout(title="Confidence")
                    st.plotly_chart(fig, use_container_width=True, key=f"adv_sq_conf_{qnum}{key_suffix}")

        # Top themes
        themes = pq.get("top_themes", [])[:8]
        if themes:
            items, counts = zip(*themes)
            fig = px.bar(x=list(counts), y=list(items), orientation="h", title="Top Themes",
                         color_discrete_sequence=["#2d5a87"])
            fig.update_layout(xaxis_title="Mentions", yaxis_title="", height=300)
            st.plotly_chart(fig, use_container_width=True, key=f"adv_sq_themes_{qnum}{key_suffix}")

        # Rich open-ended analysis for open-ended advisor questions
        if q_type == "open":
            st.markdown("---")
            audience = "RBC DS advisors" if "rbc" in str(key_suffix).lower() else "advisors"
            render_open_ended_panel(pq["question"], q_df,
                                     audience_label=audience,
                                     key_prefix=f"adv_sq_{qnum}{key_suffix}")

        st.divider()


def show_advisor_survey_responses(df, per_question, key_suffix=""):
    """Individual advisor survey responses."""
    st.subheader("Individual Responses")
    q_options = [f"Q{qnum}: {pq['question'][:50]}" for qnum, pq in sorted(per_question.items())]
    selected_q = st.selectbox("Select question", q_options, key=f"adv_surv_q_sel{key_suffix}")
    qnum = int(selected_q.split(":")[0][1:])
    q_df = df[df["question_number"] == qnum].copy()
    pq = per_question[qnum]

    # Filter for MC
    if pq.get("question_type") == "mc" and "option_counts" in pq:
        opts = ["All"] + pq.get("options", [])
        sel_opt = st.selectbox("Filter by option", opts, key=f"adv_surv_opt_sel{key_suffix}")
        if sel_opt != "All":
            q_df = q_df[q_df["selected_option"] == sel_opt]

    for _, row in q_df.head(30).iterrows():
        sentiment = row.get("sentiment", "neutral")
        emoji = {"positive": "thumbsup", "negative": "thumbsdown", "mixed": "thinking_face", "neutral": "neutral_face"}.get(sentiment, "neutral_face")
        selected = row.get("selected_option", "")
        opt_display = f" | Selected: **{selected}**" if selected else ""
        with st.expander(f":{emoji}: **{row.get('persona_name', 'Unknown')}**{opt_display}"):
            firm = row.get("firm_type", "")
            desig = row.get("designations", "")
            yrs = row.get("years_in_business", "")
            focus = row.get("practice_focus", "")
            st.caption(f"{firm} | {desig} | {yrs} years | {focus}")
            st.markdown(row.get("answer", "No answer"))
            themes = row.get("key_themes", [])
            if isinstance(themes, list) and themes:
                st.caption(f"Themes: {', '.join(themes)}")


def show_advisor_survey_data(df, key_suffix=""):
    """Raw advisor survey data table."""
    st.subheader("Raw Survey Data")
    display_cols = [
        "persona_name", "firm_type", "designations", "years_in_business",
        "practice_focus", "question_number", "question_text", "question_type",
        "selected_option", "answer", "sentiment", "confidence",
    ]
    available = [c for c in display_cols if c in df.columns]
    display_df = df[available].copy()
    st.dataframe(display_df, use_container_width=True, height=500, key=f"adv_survey_data_tbl{key_suffix}")

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Advisor Survey Results as CSV",
        data=csv,
        file_name=f"advisor_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key=f"adv_survey_download{key_suffix}",
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(consumer_personas, advisor_personas, rbc_ds_personas, rbc_pim_personas):
    with st.sidebar:
        st.title("Synthetic Persona Reactor")
        st.caption("Test ideas, run surveys and do A/B testing against investor and advisor personas")
        st.divider()

        panel = st.radio(
            "Panel",
            ["Canadians as a Whole", "Financial Advisors", "RBC DS - All", "RBC DS - PIM Licensed"],
            horizontal=False,
            help=(
                "Canadians as a Whole: 1000 synthetic Canadian investors. "
                "Financial Advisors: 1000 synthetic Canadian advisors. "
                "RBC DS - All: real RBC Dominion Securities advisor panel (from public sources). "
                "RBC DS - PIM Licensed: subset of RBC DS advisors with discretionary licence."
            ),
        )

        st.divider()

        mode = st.radio(
            "Mode",
            ["Idea Reactor", "A/B Test", "Survey"],
            horizontal=True,
            help="Idea Reactor tests one idea. A/B Test compares two variants. Survey sends custom questions.",
        )

        st.divider()

        if panel == "Canadians as a Whole":
            personas = consumer_personas
        elif panel == "Financial Advisors":
            personas = advisor_personas
        elif panel == "RBC DS - All":
            personas = rbc_ds_personas
        elif panel == "RBC DS - PIM Licensed":
            personas = rbc_pim_personas
        else:
            personas = consumer_personas

        sample_size = st.slider(
            "Sample Size",
            min_value=10, max_value=len(personas), value=50, step=10,
            help="Number of personas to test. Use 50-100 for quick feedback, 500-1000 for thorough analysis."
        )

        st.divider()

        with st.expander("Panel Demographics"):
            if panel == "Canadians as a Whole":
                st.caption(f"**Total personas available:** {len(personas)}")
                ages = [p["age"] for p in personas]
                st.caption(f"**Age range:** {min(ages)}-{max(ages)} (avg {sum(ages)/len(ages):.0f})")
                provinces = Counter(p["province"] for p in personas)
                st.caption("**Top provinces:**")
                for prov, count in provinces.most_common(5):
                    st.caption(f"  {prov}: {count} ({count/len(personas)*100:.0f}%)")
                ethnicities = Counter(p["ethnicity"] for p in personas)
                st.caption("**Ethnicities:**")
                for eth, count in ethnicities.most_common(6):
                    st.caption(f"  {eth}: {count} ({count/len(personas)*100:.0f}%)")
            elif panel in ("RBC DS - All", "RBC DS - PIM Licensed"):
                st.caption(f"**Total advisors available:** {len(personas)}")
                provinces = Counter(p.get("province", "Unknown") for p in personas)
                st.caption("**Top provinces:**")
                for prov, count in provinces.most_common(8):
                    st.caption(f"  {prov}: {count} ({count/len(personas)*100:.0f}%)")
                if panel == "RBC DS - All":
                    pim_counts = Counter(p.get("pim_licensed", "Unknown") for p in personas)
                    st.caption("**PIM / Discretionary:**")
                    for pim, count in pim_counts.most_common():
                        label = {"Y": "PIM Licensed", "N": "Non-PIM", "Unknown": "Unclassified"}.get(pim, pim)
                        st.caption(f"  {label}: {count} ({count/len(personas)*100:.0f}%)")
                years = [p.get("years_in_business", 0) for p in personas]
                yrs_groups = Counter(("Junior (0-5)" if y <= 5 else "Mid (6-15)" if y <= 15 else "Senior (16+)") for y in years)
                st.caption("**Experience (est.):**")
                for g, count in yrs_groups.most_common():
                    st.caption(f"  {g}: {count} ({count/len(personas)*100:.0f}%)")
                # Top practice focus
                focuses = Counter(p.get("practice_focus", "") for p in personas if p.get("practice_focus"))
                if focuses:
                    st.caption("**Top practice focus areas:**")
                    for focus, count in focuses.most_common(5):
                        if focus:
                            st.caption(f"  {focus[:60]}: {count}")
                designations_flat = Counter()
                for p in personas:
                    for d in (p.get("designations") or []):
                        designations_flat[d] += 1
                if designations_flat:
                    st.caption("**Top designations:**")
                    for des, count in designations_flat.most_common(6):
                        st.caption(f"  {des}: {count} ({count/len(personas)*100:.0f}%)")
            else:
                st.caption(f"**Total advisors available:** {len(personas)}")
                ages = [p["age"] for p in personas]
                st.caption(f"**Age range:** {min(ages)}-{max(ages)} (avg {sum(ages)/len(ages):.0f})")
                firms = Counter(p["firm_type"] for p in personas)
                st.caption("**Firm types:**")
                for firm, count in firms.most_common(6):
                    st.caption(f"  {firm}: {count} ({count/len(personas)*100:.0f}%)")
                focuses = Counter(p["practice_focus"] for p in personas)
                st.caption("**Practice focus:**")
                for focus, count in focuses.most_common(5):
                    st.caption(f"  {focus}: {count} ({count/len(personas)*100:.0f}%)")
                maturity = Counter(p["business_maturity"] for p in personas)
                st.caption("**Business maturity:**")
                for mat, count in maturity.most_common():
                    st.caption(f"  {mat}: {count} ({count/len(personas)*100:.0f}%)")

        st.divider()

        # --- Model picker ---
        st.markdown("**Model**")
        # Group options by provider for clarity
        provider_order = ["google", "openai", "anthropic"]
        model_ids_ordered = []
        model_labels = []
        for prov in provider_order:
            for mid, cfg in MODEL_CONFIGS.items():
                if cfg["provider"] == prov:
                    model_ids_ordered.append(mid)
                    cost_per_50 = (1100 * 50 * cfg["input_cost_per_1m"] / 1_000_000
                                    + 500 * 50 * cfg["output_cost_per_1m"] / 1_000_000)
                    model_labels.append(f"{cfg['label']} — {cfg['tier']} (~${cost_per_50:.2f}/50 personas)")

        try:
            default_idx = model_ids_ordered.index(DEFAULT_MODEL_ID)
        except ValueError:
            default_idx = 0

        sel = st.selectbox(
            "Choose a model",
            options=range(len(model_ids_ordered)),
            format_func=lambda i: model_labels[i],
            index=default_idx,
            label_visibility="collapsed",
            help="Compare outputs across providers. Re-run with a different model to see how responses differ."
        )
        selected_model_id = model_ids_ordered[sel]
        selected_cfg = MODEL_CONFIGS[selected_model_id]
        st.caption(f"_{selected_cfg['description']}_")

        # Show cost estimate for current sample size
        est_cost, _ = estimate_run_cost(selected_model_id, sample_size, "reaction")
        st.caption(f"**Est. cost for {sample_size} personas:** ~${est_cost:.3f}")

        # Warn if API key missing
        if selected_cfg["provider"] == "openai" and not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY not set in .streamlit/secrets.toml — this model will fail.", icon="⚠️")
        elif selected_cfg["provider"] == "anthropic" and not ANTHROPIC_API_KEY:
            st.warning("ANTHROPIC_API_KEY not set in .streamlit/secrets.toml — this model will fail.", icon="⚠️")

        st.divider()
        st.caption(f"Powered by {selected_cfg['label']}")

    return sample_size, mode, panel, personas, selected_model_id


# ============================================================
# MODE: IDEA REACTOR
# ============================================================

def run_reactor_mode(personas, sample_size, is_advisor=False, panel_key="consumer", model_id=None):
    idea_type = st.radio(
        "Idea Type",
        ["Investment Idea", "General Idea"],
        horizontal=True,
        key=f"{panel_key}_idea_type",
        help="Investment Idea focuses on financial products. General Idea tests any concept, proposal, or idea.",
    )
    idea_type_key = "investment" if idea_type == "Investment Idea" else "general"

    # Determine the panel label for titles
    if panel_key == "rbc_pim":
        advisor_audience = "RBC DS PIM-Licensed Advisors"
    elif panel_key == "rbc_ds":
        advisor_audience = "RBC Dominion Securities Advisors"
    else:
        advisor_audience = "Canadian Financial Advisors"

    if idea_type_key == "general":
        if is_advisor:
            st.title(f"Test Your Idea with {advisor_audience}")
            st.markdown(f"Describe your idea below — e.g. an event concept, a piece of content, a new offering — and get professional reactions from a panel of {advisor_audience}.")
        else:
            st.title("Test Your Idea")
            st.markdown("Describe your idea below and get reactions from a demographically representative panel of Canadians.")
        idea_label = "Your Idea / Concept / Proposal"
        if panel_key in ("rbc_ds", "rbc_pim"):
            placeholder = "Example: A half-day event featuring a Bank of Canada economist, a private equity fund manager, and a tax planning expert — aimed at UHNW client prospecting..."
        else:
            placeholder = "Example: A community-based car sharing program for suburban neighbourhoods, where residents share vehicles through a mobile app with insurance included..."
    else:
        if is_advisor:
            st.title(f"Test Your Idea with {advisor_audience}")
            st.markdown(f"Describe your idea below and get professional reactions from a panel of {advisor_audience}.")
            idea_label = "Idea / Product / Concept"
            if panel_key in ("rbc_ds", "rbc_pim"):
                placeholder = "Example: A new alternative investment platform offering private credit and real estate debt funds with quarterly liquidity, aimed at discretionary PIM books..."
            else:
                placeholder = "Example: A new alternative investment platform offering private credit and real estate debt funds with quarterly liquidity, targeting accredited investors with $100K+ minimums..."
        else:
            st.title("Test Your Investment Idea")
            st.markdown("Describe your investment idea below and get reactions from a demographically representative panel of Canadian investors.")
            idea_label = "Investment Idea"
            placeholder = "Example: A mobile app that lets Canadians invest spare change from everyday purchases into diversified ETF portfolios, with automatic TFSA contribution tracking and a built-in financial literacy program..."

    context = render_context_panel(panel_key)

    idea = st.text_area(idea_label, height=150, placeholder=placeholder)

    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("Test This Idea", type="primary", use_container_width=True, disabled=not idea.strip())
    with col2:
        if idea.strip():
            panel_label = "advisors" if is_advisor else "personas"
            st.caption(f"Will test against {sample_size} {panel_label} using stratified sampling")

    reactions_key = f"{panel_key}_reactions"
    idea_key = f"{panel_key}_idea"

    if run_button and idea.strip():
        sample_fn = (stratified_sample_rbc if panel_key in ("rbc_ds", "rbc_pim") else (stratified_sample_advisors if is_advisor else stratified_sample))
        sampled = sample_fn(personas, sample_size)
        panel_label = "advisors" if is_advisor else "personas"
        st.info(f"Testing against {len(sampled)} {panel_label} (stratified sample from {len(personas)})")

        with st.spinner("Collecting reactions..."):
            reactions = collect_reactions(sampled, idea.strip(), is_advisor=is_advisor, context=context, idea_type=idea_type_key, model_id=model_id)

        st.session_state[reactions_key] = reactions
        st.session_state[idea_key] = idea.strip()
        st.session_state[f"{panel_key}_idea_type_saved"] = idea_type_key

        errors = sum(1 for r in reactions if "error" in r)
        valid = [r for r in reactions if "error" not in r]
        avg_score = sum(r.get("interest_score", 0) for r in valid) / len(valid) if valid else 0
        log_usage({
            "mode": "Idea Reactor",
            "panel": panel_key,
            "model_id": model_id or DEFAULT_MODEL_ID,
            "sample_size": len(sampled),
            "api_calls": len(sampled),
            "errors": errors,
            "idea_preview": idea.strip()[:100],
            "key_results": {"avg_interest_score": round(avg_score, 1), "valid_responses": len(valid)},
        })

    if reactions_key in st.session_state:
        reactions = st.session_state[reactions_key]
        build_fn = build_advisor_analysis if is_advisor else build_analysis
        df, analysis = build_fn(reactions)

        if df is None or df.empty:
            st.error("No valid reactions received. Check your API key and try again.")
            return

        st.divider()
        st.header("Results")
        idea_text = st.session_state.get(idea_key, "")
        st.caption(f"Idea: *{idea_text[:150]}...*" if len(idea_text) > 150 else f"Idea: *{idea_text}*")

        # Excel export
        try:
            _idea_type = st.session_state.get(f"{panel_key}_idea_type_saved", "investment")
            excel_bytes = build_excel_export_reactor(
                df, analysis, idea_text,
                is_advisor=is_advisor,
                idea_type=_idea_type,
                context_text=st.session_state.get(f"{panel_key}_context_area", ""),
            )
            render_excel_export_button(
                "Export All Results to Excel",
                excel_bytes,
                f"idea_reactor_{panel_key}",
                key=f"{panel_key}_export_reactor",
            )
        except Exception as _e:
            st.caption(f"Excel export unavailable: {str(_e)[:100]}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Demographics", "Insights", "Verbatim Quotes", "Raw Data"
        ])

        if is_advisor:
            with tab1:
                show_advisor_overview(df)
            with tab2:
                show_advisor_demographics(df)
            with tab3:
                show_insights(df, analysis)
            with tab4:
                show_advisor_quotes(df)
            with tab5:
                show_advisor_data(df)
        else:
            with tab1:
                show_overview(df)
            with tab2:
                show_demographics(df)
            with tab3:
                show_insights(df, analysis)
            with tab4:
                show_quotes(df)
            with tab5:
                show_data(df)


# ============================================================
# MODE: A/B TEST
# ============================================================

def show_ab_comparison(df_a, df_b, analysis_a, analysis_b, idea_a, idea_b):
    st.subheader("Head-to-Head Comparison")

    # Key metrics
    avg_a = df_a["interest_score"].mean()
    avg_b = df_b["interest_score"].mean()
    invest_a = df_a["would_invest"].mean() * 100
    invest_b = df_b["would_invest"].mean() * 100
    median_a = df_a["interest_score"].median()
    median_b = df_b["interest_score"].median()
    sent_a = df_a["sentiment"].mode().iloc[0] if not df_a["sentiment"].mode().empty else "N/A"
    sent_b = df_b["sentiment"].mode().iloc[0] if not df_b["sentiment"].mode().empty else "N/A"

    col_a, col_vs, col_b = st.columns([5, 1, 5])
    with col_a:
        st.markdown("#### Variant A")
        st.caption(f"*{idea_a[:120]}{'...' if len(idea_a) > 120 else ''}*")
    with col_vs:
        st.markdown("<div style='text-align:center; padding-top:20px; font-size:1.5em; font-weight:bold;'>vs</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown("#### Variant B")
        st.caption(f"*{idea_b[:120]}{'...' if len(idea_b) > 120 else ''}*")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        winner = "A" if avg_a > avg_b else ("B" if avg_b > avg_a else "Tie")
        st.metric("Avg Interest Score", f"{avg_a:.1f} vs {avg_b:.1f}", delta=f"{'A' if winner == 'A' else 'B'} wins by {abs(avg_a - avg_b):.1f}" if winner != "Tie" else "Tied")
    with m2:
        winner = "A" if invest_a > invest_b else ("B" if invest_b > invest_a else "Tie")
        st.metric("Would Invest", f"{invest_a:.0f}% vs {invest_b:.0f}%", delta=f"{'A' if winner == 'A' else 'B'} +{abs(invest_a - invest_b):.0f}pp" if winner != "Tie" else "Tied")
    with m3:
        st.metric("Top Sentiment", f"{sent_a.title()} vs {sent_b.title()}")
    with m4:
        st.metric("Median Score", f"{median_a:.0f} vs {median_b:.0f}")

    st.divider()

    # Interest score distribution overlay
    st.subheader("Interest Score Distribution")
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_a["interest_score"], name="Variant A", opacity=0.7, marker_color="#2d5a87", nbinsx=10))
    fig.add_trace(go.Histogram(x=df_b["interest_score"], name="Variant B", opacity=0.7, marker_color="#e67e22", nbinsx=10))
    fig.update_layout(barmode="overlay", xaxis_title="Interest Score", yaxis_title="Count", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True, key="ab_score_dist")

    # Sentiment comparison
    st.subheader("Sentiment Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        sent_a_counts = df_a["sentiment"].value_counts()
        fig = px.pie(values=sent_a_counts.values, names=sent_a_counts.index, title="Variant A", color=sent_a_counts.index, color_discrete_map=SENTIMENT_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True, key="ab_sent_a")
    with c2:
        sent_b_counts = df_b["sentiment"].value_counts()
        fig = px.pie(values=sent_b_counts.values, names=sent_b_counts.index, title="Variant B", color=sent_b_counts.index, color_discrete_map=SENTIMENT_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True, key="ab_sent_b")

    # Investment amount comparison
    if "investment_amount" in df_a.columns and "investment_amount" in df_b.columns:
        st.subheader("Investment Amount Willingness")
        amount_order = ["none", "small ($100-$1K)", "moderate ($1K-$10K)", "significant ($10K-$50K)", "major ($50K+)"]
        amt_a = df_a["investment_amount"].value_counts().reindex(amount_order, fill_value=0)
        amt_b = df_b["investment_amount"].value_counts().reindex(amount_order, fill_value=0)
        amt_df = pd.DataFrame({"Amount": amount_order * 2, "Count": list(amt_a.values) + list(amt_b.values), "Variant": ["A"] * len(amount_order) + ["B"] * len(amount_order)})
        fig = px.bar(amt_df, x="Amount", y="Count", color="Variant", barmode="group", color_discrete_map={"A": "#2d5a87", "B": "#e67e22"})
        st.plotly_chart(fig, use_container_width=True, key="ab_amounts")

    # Would invest by age group
    st.subheader("% Would Invest by Demographics")
    c1, c2 = st.columns(2)
    with c1:
        age_a = df_a.groupby("age_group", observed=True)["would_invest"].mean().mul(100).reset_index()
        age_a.columns = ["Age Group", "% Would Invest"]
        age_a["Variant"] = "A"
        age_b = df_b.groupby("age_group", observed=True)["would_invest"].mean().mul(100).reset_index()
        age_b.columns = ["Age Group", "% Would Invest"]
        age_b["Variant"] = "B"
        age_df = pd.concat([age_a, age_b])
        fig = px.bar(age_df, x="Age Group", y="% Would Invest", color="Variant", barmode="group", title="By Age Group", color_discrete_map={"A": "#2d5a87", "B": "#e67e22"})
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, key="ab_age")

    with c2:
        inc_a = df_a.groupby("income_bracket", observed=True)["would_invest"].mean().mul(100).reset_index()
        inc_a.columns = ["Income Bracket", "% Would Invest"]
        inc_a["Variant"] = "A"
        inc_b = df_b.groupby("income_bracket", observed=True)["would_invest"].mean().mul(100).reset_index()
        inc_b.columns = ["Income Bracket", "% Would Invest"]
        inc_b["Variant"] = "B"
        inc_df = pd.concat([inc_a, inc_b])
        fig = px.bar(inc_df, x="Income Bracket", y="% Would Invest", color="Variant", barmode="group", title="By Income Bracket", color_discrete_map={"A": "#2d5a87", "B": "#e67e22"})
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, key="ab_income")

    # Risk tolerance comparison
    risk_a = df_a.groupby("risk_tolerance_profile", observed=True)["would_invest"].mean().mul(100).reset_index()
    risk_a.columns = ["Risk Tolerance", "% Would Invest"]
    risk_a["Variant"] = "A"
    risk_b = df_b.groupby("risk_tolerance_profile", observed=True)["would_invest"].mean().mul(100).reset_index()
    risk_b.columns = ["Risk Tolerance", "% Would Invest"]
    risk_b["Variant"] = "B"
    risk_df = pd.concat([risk_a, risk_b])
    risk_order = ["Very Conservative", "Conservative", "Moderate", "Growth", "Aggressive"]
    risk_df["Risk Tolerance"] = pd.Categorical(risk_df["Risk Tolerance"], categories=risk_order, ordered=True)
    risk_df = risk_df.sort_values("Risk Tolerance")
    fig = px.bar(risk_df, x="Risk Tolerance", y="% Would Invest", color="Variant", barmode="group", title="By Risk Tolerance", color_discrete_map={"A": "#2d5a87", "B": "#e67e22"})
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, key="ab_risk")

    # Top concerns and appeals side by side
    st.subheader("Concerns & Appeals")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Concerns — Variant A**")
        if analysis_a["top_concerns"]:
            cdf = pd.DataFrame(analysis_a["top_concerns"][:8], columns=["Concern", "Count"])
            fig = px.bar(cdf.iloc[::-1], x="Count", y="Concern", orientation="h", color_discrete_sequence=["#2d5a87"])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="ab_concerns_a")

        st.markdown("**Top Appeals — Variant A**")
        if analysis_a["top_appeals"]:
            adf = pd.DataFrame(analysis_a["top_appeals"][:8], columns=["Appeal", "Count"])
            fig = px.bar(adf.iloc[::-1], x="Count", y="Appeal", orientation="h", color_discrete_sequence=["#2d5a87"])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="ab_appeals_a")

    with c2:
        st.markdown("**Top Concerns — Variant B**")
        if analysis_b["top_concerns"]:
            cdf = pd.DataFrame(analysis_b["top_concerns"][:8], columns=["Concern", "Count"])
            fig = px.bar(cdf.iloc[::-1], x="Count", y="Concern", orientation="h", color_discrete_sequence=["#e67e22"])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="ab_concerns_b")

        st.markdown("**Top Appeals — Variant B**")
        if analysis_b["top_appeals"]:
            adf = pd.DataFrame(analysis_b["top_appeals"][:8], columns=["Appeal", "Count"])
            fig = px.bar(adf.iloc[::-1], x="Count", y="Appeal", orientation="h", color_discrete_sequence=["#e67e22"])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="ab_appeals_b")

    # Statistical summary
    st.divider()
    diff = avg_a - avg_b
    pct_diff = abs(diff) / min(avg_a, avg_b) * 100 if min(avg_a, avg_b) > 0 else 0
    winner_label = "Variant A" if diff > 0 else "Variant B" if diff < 0 else "Neither"
    n = min(len(df_a), len(df_b))

    try:
        from scipy import stats
        # Merge on persona_id for paired comparison
        merged = df_a[["persona_id", "interest_score"]].merge(
            df_b[["persona_id", "interest_score"]], on="persona_id", suffixes=("_a", "_b")
        )
        if len(merged) > 1:
            t_stat, p_value = stats.ttest_rel(merged["interest_score_a"], merged["interest_score_b"])
            sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
            st.info(f"**{winner_label}** scored higher by {abs(diff):.1f} points ({pct_diff:.0f}%). "
                    f"Paired t-test: t={t_stat:.2f}, p={p_value:.3f} — {sig} at 95% confidence (n={len(merged)} paired responses).")
        else:
            st.info(f"**{winner_label}** scored higher by {abs(diff):.1f} points ({pct_diff:.0f}%) across {n} personas.")
    except ImportError:
        st.info(f"**{winner_label}** scored higher by {abs(diff):.1f} points ({pct_diff:.0f}%) across {n} personas.")


def run_ab_test_mode(personas, sample_size, is_advisor=False, panel_key="consumer", model_id=None):
    st.title("A/B Test")
    panel_label = "advisors" if is_advisor else "personas"
    st.markdown(f"Compare two ideas or two ways of messaging the same idea. "
                f"Both variants are tested against the **same** {panel_label} sample for a fair comparison.")

    context = render_context_panel(panel_key)

    c1, c2 = st.columns(2)
    with c1:
        idea_a = st.text_area(
            "Variant A", height=150,
            placeholder="Enter the first variant of your idea...",
            key=f"{panel_key}_ab_idea_a_input",
        )
    with c2:
        idea_b = st.text_area(
            "Variant B", height=150,
            placeholder="Enter the second variant of your idea...",
            key=f"{panel_key}_ab_idea_b_input",
        )

    both_filled = idea_a.strip() and idea_b.strip()

    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("Run A/B Test", type="primary", use_container_width=True, disabled=not both_filled)
    with col2:
        if both_filled:
            st.caption(f"Will test both variants against the same {sample_size} {panel_label}")

    rxa_key = f"{panel_key}_ab_reactions_a"
    rxb_key = f"{panel_key}_ab_reactions_b"
    ida_key = f"{panel_key}_ab_idea_a"
    idb_key = f"{panel_key}_ab_idea_b"

    if run_button and both_filled:
        sample_fn = (stratified_sample_rbc if panel_key in ("rbc_ds", "rbc_pim") else (stratified_sample_advisors if is_advisor else stratified_sample))
        sampled = sample_fn(personas, sample_size)
        st.info(f"Testing both variants against {len(sampled)} {panel_label}...")

        st.markdown("**Testing Variant A...**")
        reactions_a = collect_reactions(sampled, idea_a.strip(), is_advisor=is_advisor, context=context, model_id=model_id)

        st.markdown("**Testing Variant B...**")
        reactions_b = collect_reactions(sampled, idea_b.strip(), is_advisor=is_advisor, context=context, model_id=model_id)

        st.session_state[rxa_key] = reactions_a
        st.session_state[rxb_key] = reactions_b
        st.session_state[ida_key] = idea_a.strip()
        st.session_state[idb_key] = idea_b.strip()

        errors_a = sum(1 for r in reactions_a if "error" in r)
        errors_b = sum(1 for r in reactions_b if "error" in r)
        log_usage({
            "mode": "A/B Test",
            "panel": panel_key,
            "model_id": model_id or DEFAULT_MODEL_ID,
            "sample_size": len(sampled),
            "api_calls": len(sampled) * 2,
            "errors": errors_a + errors_b,
            "idea_preview": f"A: {idea_a.strip()[:50]} | B: {idea_b.strip()[:50]}",
            "key_results": {"valid_a": len(sampled) - errors_a, "valid_b": len(sampled) - errors_b},
        })

    if rxa_key in st.session_state and rxb_key in st.session_state:
        reactions_a = st.session_state[rxa_key]
        reactions_b = st.session_state[rxb_key]
        idea_a_text = st.session_state.get(ida_key, "Variant A")
        idea_b_text = st.session_state.get(idb_key, "Variant B")

        build_fn = build_advisor_analysis if is_advisor else build_analysis
        df_a, analysis_a = build_fn(reactions_a)
        df_b, analysis_b = build_fn(reactions_b)

        if (df_a is None or df_a.empty) and (df_b is None or df_b.empty):
            st.error("No valid reactions received for either variant. Check your API key and try again.")
            return

        st.divider()
        st.header("A/B Test Results")

        # Excel export
        try:
            excel_bytes = build_excel_export_ab(
                df_a, df_b, analysis_a, analysis_b,
                idea_a_text, idea_b_text,
                is_advisor=is_advisor,
                context_text=st.session_state.get(f"{panel_key}_context_area", ""),
            )
            render_excel_export_button(
                "Export A/B Test Results to Excel",
                excel_bytes,
                f"ab_test_{panel_key}",
                key=f"{panel_key}_export_ab",
            )
        except Exception as _e:
            st.caption(f"Excel export unavailable: {str(_e)[:100]}")

        tab1, tab2, tab3 = st.tabs(["Comparison", "Variant A Detail", "Variant B Detail"])

        with tab1:
            if df_a is not None and df_b is not None and not df_a.empty and not df_b.empty:
                if is_advisor:
                    show_advisor_ab_comparison(df_a, df_b, analysis_a, analysis_b, idea_a_text, idea_b_text)
                else:
                    show_ab_comparison(df_a, df_b, analysis_a, analysis_b, idea_a_text, idea_b_text)
            else:
                st.warning("One variant had no valid reactions. Cannot show comparison.")

        with tab2:
            if df_a is not None and not df_a.empty:
                st.caption(f"Idea: *{idea_a_text[:150]}*")
                if is_advisor:
                    show_advisor_overview(df_a, key_suffix="_ab_a")
                    show_advisor_demographics(df_a, key_suffix="_ab_a")
                    show_insights(df_a, analysis_a)
                    show_advisor_quotes(df_a, key_suffix="_ab_a")
                    show_advisor_data(df_a, key_suffix="_ab_a")
                else:
                    show_overview(df_a)
                    show_demographics(df_a)
                    show_insights(df_a, analysis_a)
                    show_quotes(df_a, key_suffix="_ab_a")
                    show_data(df_a, key_suffix="_ab_a")
            else:
                st.warning("No valid reactions for Variant A.")

        with tab3:
            if df_b is not None and not df_b.empty:
                st.caption(f"Idea: *{idea_b_text[:150]}*")
                if is_advisor:
                    show_advisor_overview(df_b, key_suffix="_ab_b")
                    show_advisor_demographics(df_b, key_suffix="_ab_b")
                    show_insights(df_b, analysis_b)
                    show_advisor_quotes(df_b, key_suffix="_ab_b")
                    show_advisor_data(df_b, key_suffix="_ab_b")
                else:
                    show_overview(df_b)
                    show_demographics(df_b)
                    show_insights(df_b, analysis_b)
                    show_quotes(df_b, key_suffix="_ab_b")
                    show_data(df_b, key_suffix="_ab_b")
            else:
                st.warning("No valid reactions for Variant B.")


# ============================================================
# MODE: SURVEY
# ============================================================

def run_survey_mode(personas, sample_size, is_advisor=False, panel_key="consumer", model_id=None):
    if panel_key == "rbc_pim":
        st.title("Survey the RBC DS PIM-Licensed Panel")
        st.markdown("Build your survey below. Great for open-ended questions like \"what content would you want to see at an event?\" or \"what are the best 5 strategies to prospect this group?\"")
    elif panel_key == "rbc_ds":
        st.title("Survey the RBC Dominion Securities Panel")
        st.markdown("Build your survey below. Great for open-ended questions like \"what content would you want to see at an event?\" or \"what are the best 5 strategies to prospect this group?\"")
    elif is_advisor:
        st.title("Survey Your Advisor Panel")
        st.markdown("Build your survey below. Add open-ended or multiple-choice questions using the buttons.")
    else:
        st.title("Survey Your Persona Panel")
        st.markdown("Build your survey below. Add open-ended or multiple-choice questions using the buttons.")

    context = render_context_panel(panel_key)

    questions = render_question_builder(panel_key)

    panel_label = "advisors" if is_advisor else "personas"
    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("Run Survey", type="primary", use_container_width=True, disabled=len(questions) == 0)
    with col2:
        if questions:
            st.caption(f"Will survey {sample_size} {panel_label} with {len(questions)} question(s)")

    resp_key = f"{panel_key}_survey_responses"
    q_key = f"{panel_key}_survey_questions"

    if run_button and questions:
        sample_fn = (stratified_sample_rbc if panel_key in ("rbc_ds", "rbc_pim") else (stratified_sample_advisors if is_advisor else stratified_sample))
        sampled = sample_fn(personas, sample_size)
        st.info(f"Surveying {len(sampled)} {panel_label} with {len(questions)} questions...")

        with st.spinner("Collecting survey responses..."):
            responses = collect_survey_responses(sampled, questions, is_advisor=is_advisor, context=context, model_id=model_id)

        st.session_state[resp_key] = responses
        st.session_state[q_key] = questions

        errors = sum(1 for r in responses if "error" in r)
        log_usage({
            "mode": "Survey",
            "panel": panel_key,
            "model_id": model_id or DEFAULT_MODEL_ID,
            "sample_size": len(sampled),
            "api_calls": len(sampled),
            "errors": errors,
            "idea_preview": f"Survey: {len(questions)} questions",
            "key_results": {"questions": len(questions), "valid_responses": len(sampled) - errors},
        })

    if resp_key in st.session_state and q_key in st.session_state:
        responses = st.session_state[resp_key]
        questions = st.session_state[q_key]
        # Legacy guard: convert old string format to structured dicts
        if questions and isinstance(questions[0], str):
            questions = [{"type": "open", "text": q} for q in questions]

        build_survey_fn = build_advisor_survey_analysis if is_advisor else build_survey_analysis
        df, per_question = build_survey_fn(responses, questions)

        if df is None or df.empty:
            st.error("No valid survey responses received. Check your API key and try again.")
            return

        st.divider()
        st.header("Survey Results")
        st.caption(f"{df['persona_id'].nunique()} respondents, {len(questions)} questions")

        # Excel export
        try:
            excel_bytes = build_excel_export_survey(
                df, per_question, questions,
                is_advisor=is_advisor,
                context_text=st.session_state.get(f"{panel_key}_context_area", ""),
            )
            render_excel_export_button(
                "Export Survey Results to Excel",
                excel_bytes,
                f"survey_{panel_key}",
                key=f"{panel_key}_export_survey",
            )
        except Exception as _e:
            st.caption(f"Excel export unavailable: {str(_e)[:100]}")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview", "Per-Question Analysis", "Individual Responses", "Raw Data & Export"
        ])

        if is_advisor:
            with tab1:
                show_advisor_survey_overview(df, per_question)
            with tab2:
                show_advisor_per_question(df, per_question)
            with tab3:
                show_advisor_survey_responses(df, per_question)
            with tab4:
                show_advisor_survey_data(df)
        else:
            with tab1:
                show_survey_overview(df, per_question, questions)
            with tab2:
                show_per_question_analysis(df, per_question, questions)
            with tab3:
                show_survey_responses(df, questions)
            with tab4:
                show_survey_data(df)



# ============================================================
# ADMIN DASHBOARD
# ============================================================




def run_admin_dashboard():
    st.title("📊 Usage Report")
    st.caption("Admin dashboard — persistent usage tracking across sessions")

    if not os.path.exists(USAGE_LOG_FILE):
        st.info("No usage data yet. Run some tests first.")
        return

    with open(USAGE_LOG_FILE, "r", encoding="utf-8") as f:
        log = json.load(f)

    if not log:
        st.info("No usage data yet. Run some tests first.")
        return

    df = pd.DataFrame(log)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # --- Summary Metrics ---
    st.header("Summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Tests Run", len(df))
    with m2:
        st.metric("Total API Calls", f"{df['api_calls'].sum():,}")
    with m3:
        st.metric("Estimated Total Cost", f"${df['estimated_cost_usd'].sum():.2f}")
    with m4:
        error_rate = df["errors"].sum() / df["api_calls"].sum() * 100 if df["api_calls"].sum() > 0 else 0
        st.metric("Error Rate", f"{error_rate:.1f}%")

    st.divider()

    # --- Charts ---
    st.header("Usage Trends")

    col1, col2 = st.columns(2)

    with col1:
        # Tests per day
        daily = df.groupby("date").size().reset_index(name="tests")
        daily["date"] = pd.to_datetime(daily["date"])
        fig = px.bar(daily, x="date", y="tests", title="Tests Per Day",
                     labels={"date": "Date", "tests": "Tests Run"},
                     color_discrete_sequence=["#2d5a87"])
        fig.update_layout(xaxis_title="", yaxis_title="Tests")
        st.plotly_chart(fig, use_container_width=True, key="admin_daily")

    with col2:
        # API calls per day
        daily_calls = df.groupby("date")["api_calls"].sum().reset_index()
        daily_calls["date"] = pd.to_datetime(daily_calls["date"])
        fig = px.bar(daily_calls, x="date", y="api_calls", title="API Calls Per Day",
                     labels={"date": "Date", "api_calls": "API Calls"},
                     color_discrete_sequence=["#e67e22"])
        fig.update_layout(xaxis_title="", yaxis_title="API Calls")
        st.plotly_chart(fig, use_container_width=True, key="admin_api_daily")

    col3, col4 = st.columns(2)

    with col3:
        # By mode
        mode_counts = df["mode"].value_counts()
        fig = px.pie(values=mode_counts.values, names=mode_counts.index,
                     title="Usage by Mode",
                     color_discrete_sequence=["#2d5a87", "#e67e22", "#27ae60"])
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True, key="admin_mode_pie")

    with col4:
        # By panel
        panel_counts = df["panel"].value_counts()
        panel_labels = panel_counts.index.map(lambda x: "Financial Advisors" if x == "advisor" else "Canadians as a Whole")
        fig = px.pie(values=panel_counts.values, names=panel_labels,
                     title="Usage by Panel",
                     color_discrete_sequence=["#3498db", "#9b59b6"])
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True, key="admin_panel_pie")

    st.divider()

    # --- Cost breakdown ---
    st.header("Cost Breakdown")
    cost_by_mode = df.groupby("mode")["estimated_cost_usd"].sum().reset_index()
    cost_by_mode.columns = ["Mode", "Cost (USD)"]
    cost_by_panel = df.groupby("panel")["estimated_cost_usd"].sum().reset_index()
    cost_by_panel.columns = ["Panel", "Cost (USD)"]
    cost_by_panel["Panel"] = cost_by_panel["Panel"].map(lambda x: "Financial Advisors" if x == "advisor" else "Canadians as a Whole")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By Mode")
        st.dataframe(cost_by_mode, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("By Panel")
        st.dataframe(cost_by_panel, use_container_width=True, hide_index=True)

    st.divider()

    # --- Activity Log ---
    st.header("Activity Log")
    display_df = df[["timestamp", "mode", "panel", "sample_size", "api_calls", "errors", "estimated_cost_usd", "idea_preview"]].copy()
    display_df.columns = ["Timestamp", "Mode", "Panel", "Sample Size", "API Calls", "Errors", "Cost (USD)", "Idea Preview"]
    display_df["Panel"] = display_df["Panel"].map(lambda x: "Advisors" if x == "advisor" else "Consumers")
    display_df = display_df.sort_values("Timestamp", ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    # --- Download ---
    csv = display_df.to_csv(index=False)
    st.download_button(
        "Download Usage Log (CSV)",
        csv,
        f"usage_log_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        key="admin_download",
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Admin dashboard - accessible via ?admin=<secret>
    params = st.query_params
    if params.get("admin") == ADMIN_SECRET:
        run_admin_dashboard()
        return

    consumer_personas = load_personas()
    advisor_personas = load_advisor_personas()
    rbc_ds_personas = load_rbc_ds_personas()
    rbc_pim_personas = load_rbc_pim_personas()
    sample_size, mode, panel, personas, model_id = render_sidebar(
        consumer_personas, advisor_personas, rbc_ds_personas, rbc_pim_personas
    )

    advisor_panels = {"Financial Advisors", "RBC DS - All", "RBC DS - PIM Licensed"}
    is_advisor = panel in advisor_panels
    if panel == "Financial Advisors":
        panel_key = "advisor"
    elif panel == "RBC DS - All":
        panel_key = "rbc_ds"
    elif panel == "RBC DS - PIM Licensed":
        panel_key = "rbc_pim"
    else:
        panel_key = "consumer"

    if mode == "Idea Reactor":
        run_reactor_mode(personas, sample_size, is_advisor, panel_key, model_id)
    elif mode == "A/B Test":
        run_ab_test_mode(personas, sample_size, is_advisor, panel_key, model_id)
    else:
        run_survey_mode(personas, sample_size, is_advisor, panel_key, model_id)


if __name__ == "__main__":
    main()
