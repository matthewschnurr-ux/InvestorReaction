#!/usr/bin/env python3
"""
Canadian Persona Reactor & Survey Tool - Streamlit Web App
Test investment ideas and run custom surveys against 1,000 synthetic Canadian personas.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime

from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================

# API key loaded from .streamlit/secrets.toml (local) or Streamlit Cloud secrets
API_KEY = st.secrets["GEMINI_API_KEY"]

MODEL_NAME = "gemini-2.5-flash"
MAX_WORKERS = 10
PERSONAS_FILE = os.path.join(os.path.dirname(__file__), "personas.json")

REACTION_PROMPT = """You are roleplaying as this Canadian investor persona. Stay fully in character. Think and respond as this specific person would, given their background, financial situation, knowledge level, and life circumstances.

PERSONA:
{persona}

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

You are being surveyed. Answer each question honestly and naturally from this persona's perspective. Consider your age, income, education, family situation, values, location, and life experiences.

QUESTIONS:
{questions_formatted}

Respond ONLY with valid JSON in this exact format (no markdown, no code fences):
{{
  "answers": [
    {{
      "question_number": 1,
      "answer": "<your natural, in-character answer in 1-3 sentences>",
      "sentiment": "<positive|neutral|negative|mixed>",
      "confidence": "<low|medium|high>",
      "key_themes": ["<theme1>", "<theme2>"]
    }}
  ]
}}

Include one entry in the "answers" array for each question, in order."""


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Persona Reactor",
    page_icon=":flag-ca:",
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


def make_error_result(persona, error_msg):
    """Create an error result dict for a persona."""
    return {
        "persona_id": persona["id"],
        "persona_name": f"{persona['first_name']} {persona['last_name']}",
        "error": error_msg,
    }


# ============================================================
# API CALLS - REACTOR
# ============================================================

def get_reaction(client, persona, idea):
    prompt = REACTION_PROMPT.format(persona=persona["persona_summary"], idea=idea)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.8,
                    max_output_tokens=1000,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            reaction = json.loads(text)
            return attach_persona_metadata(reaction, persona)
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(1)
                continue
            return make_error_result(persona, "JSON parse error")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep((attempt + 1) * 10)
                continue
            if attempt < 2:
                time.sleep(2)
                continue
            return make_error_result(persona, str(e)[:100])


def collect_reactions(personas, idea):
    client = genai.Client(api_key=API_KEY)
    total = len(personas)
    reactions = []
    completed = 0
    errors = 0
    delay = 60.0 / 450

    progress_bar = st.progress(0, text=f"Starting... 0/{total}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_persona = {}
        for persona in personas:
            future = executor.submit(get_reaction, client, persona, idea)
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
                text=f"Collected {completed}/{total} reactions ({errors} errors)"
            )

    progress_bar.progress(1.0, text=f"Done! {total} reactions collected ({errors} errors)")
    return reactions


# ============================================================
# API CALLS - SURVEY
# ============================================================

def get_survey_response(client, persona, questions):
    questions_formatted = "\n".join(f"Q{i}. {q}" for i, q in enumerate(questions, 1))
    prompt = SURVEY_PROMPT.format(
        persona=persona["persona_summary"],
        questions_formatted=questions_formatted,
    )
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.8,
                    max_output_tokens=min(300 * len(questions), 8000),
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            result = json.loads(text)
            return attach_persona_metadata(result, persona)
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(1)
                continue
            return make_error_result(persona, "JSON parse error")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep((attempt + 1) * 10)
                continue
            if attempt < 2:
                time.sleep(2)
                continue
            return make_error_result(persona, str(e)[:100])


def collect_survey_responses(personas, questions):
    client = genai.Client(api_key=API_KEY)
    total = len(personas)
    responses = []
    completed = 0
    errors = 0
    delay = 60.0 / 450

    progress_bar = st.progress(0, text=f"Starting survey... 0/{total}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_persona = {}
        for persona in personas:
            future = executor.submit(get_survey_response, client, persona, questions)
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
                text=f"Surveyed {completed}/{total} personas ({errors} errors)"
            )

    progress_bar.progress(1.0, text=f"Done! {total} personas surveyed ({errors} errors)")
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
                    "question_text": questions[qnum - 1],
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
        q_df = df[df["question_number"] == qnum]
        all_themes = []
        for themes in q_df["key_themes"]:
            if isinstance(themes, list):
                all_themes.extend(themes)
        per_question[qnum] = {
            "question": questions[qnum - 1],
            "response_count": len(q_df),
            "sentiment_counts": q_df["sentiment"].value_counts().to_dict(),
            "confidence_counts": q_df["confidence"].value_counts().to_dict(),
            "top_themes": Counter(all_themes).most_common(15),
        }

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


def show_quotes(df):
    st.subheader("Verbatim Reactions")

    c1, c2, c3 = st.columns(3)
    with c1:
        filter_sentiment = st.selectbox("Filter by sentiment", ["All", "positive", "negative", "neutral", "mixed"])
    with c2:
        filter_invest = st.selectbox("Filter by would invest", ["All", "Yes", "No"])
    with c3:
        sort_by = st.selectbox("Sort by", ["Interest Score (High to Low)", "Interest Score (Low to High)", "Random"])

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


def show_data(df):
    st.subheader("Raw Data")

    display_cols = [
        "persona_name", "age", "gender", "province", "income", "net_worth",
        "risk_tolerance_profile", "interest_score", "sentiment", "would_invest",
        "investment_amount", "gut_reaction",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].sort_values("interest_score", ascending=False), use_container_width=True, height=500)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full Results as CSV",
        data=csv,
        file_name=f"reactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ============================================================
# DISPLAY - SURVEY
# ============================================================

def show_survey_overview(df, per_question, questions):
    st.subheader("Survey Overview")

    col1, col2, col3 = st.columns(3)
    n_respondents = df["persona_id"].nunique()
    col1.metric("Respondents", n_respondents)
    col2.metric("Questions", len(questions))
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
    for qnum in range(1, len(questions) + 1):
        qa = per_question.get(qnum, {})
        sc = qa.get("sentiment_counts", {})
        label = f"Q{qnum}: {questions[qnum-1][:80]}"
        heatmap_data.append({
            "Question": label,
            "Positive": sc.get("positive", 0),
            "Neutral": sc.get("neutral", 0),
            "Mixed": sc.get("mixed", 0),
            "Negative": sc.get("negative", 0),
        })
    heatmap_df = pd.DataFrame(heatmap_data).set_index("Question")
    st.dataframe(
        heatmap_df.style.background_gradient(cmap="RdYlGn", axis=1),
        use_container_width=True,
    )


def show_per_question_analysis(df, per_question, questions):
    st.subheader("Per-Question Breakdown")

    for qnum in range(1, len(questions) + 1):
        qa = per_question.get(qnum, {})
        q_df = df[df["question_number"] == qnum]

        with st.expander(f"Q{qnum}: {questions[qnum-1]}", expanded=(qnum == 1)):
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
                st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig, use_container_width=True)

            # Themes
            if qa.get("top_themes"):
                themes_df = pd.DataFrame(qa["top_themes"], columns=["Theme", "Count"])
                fig = px.bar(
                    themes_df.iloc[::-1], x="Count", y="Theme",
                    orientation="h", title="Top Themes",
                    color_discrete_sequence=["#2d5a87"],
                )
                fig.update_layout(height=max(300, len(themes_df) * 28))
                st.plotly_chart(fig, use_container_width=True)

            # Sentiment by age group
            st.markdown("**Sentiment by Age Group**")
            age_sent = q_df.groupby(["age_group", "sentiment"], observed=True).size().reset_index(name="count")
            if not age_sent.empty:
                fig = px.bar(
                    age_sent, x="age_group", y="count", color="sentiment",
                    barmode="group", color_discrete_map=SENTIMENT_COLORS,
                    labels={"age_group": "Age Group", "count": "Count"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Sentiment by income
            st.markdown("**Sentiment by Income Bracket**")
            inc_sent = q_df.groupby(["income_bracket", "sentiment"], observed=True).size().reset_index(name="count")
            if not inc_sent.empty:
                fig = px.bar(
                    inc_sent, x="income_bracket", y="count", color="sentiment",
                    barmode="group", color_discrete_map=SENTIMENT_COLORS,
                    labels={"income_bracket": "Income Bracket", "count": "Count"},
                )
                st.plotly_chart(fig, use_container_width=True)


def show_survey_responses(df, questions):
    st.subheader("Individual Responses")

    selected_q = st.selectbox(
        "Select question",
        [f"Q{i}: {q}" for i, q in enumerate(questions, 1)],
        key="survey_q_select",
    )
    qnum = int(selected_q.split(":")[0][1:])
    q_df = df[df["question_number"] == qnum].copy()

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
        "question_number", "question_text", "persona_name", "age", "gender",
        "province", "income", "life_stage", "answer", "sentiment",
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
# SIDEBAR
# ============================================================

def render_sidebar(personas):
    with st.sidebar:
        st.title(":flag-ca: Persona Reactor")
        st.caption("Test ideas and run surveys against synthetic Canadian personas")
        st.divider()

        mode = st.radio(
            "Mode",
            ["Idea Reactor", "Survey"],
            horizontal=True,
            help="Idea Reactor tests investment ideas. Survey sends custom questions to the persona panel.",
        )

        st.divider()

        sample_size = st.slider(
            "Sample Size",
            min_value=10, max_value=len(personas), value=50, step=10,
            help="Number of personas to test. Use 50-100 for quick feedback, 500-1000 for thorough analysis."
        )

        st.divider()

        with st.expander("Persona Demographics"):
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

        st.divider()
        st.caption("Powered by Gemini 2.5 Flash")

    return sample_size, mode


# ============================================================
# MODE: IDEA REACTOR
# ============================================================

def run_reactor_mode(personas, sample_size):
    st.title("Test Your Investment Idea")
    st.markdown("Describe your investment idea below and get reactions from a demographically representative panel of Canadian investors.")

    idea = st.text_area(
        "Investment Idea",
        height=150,
        placeholder="Example: A mobile app that lets Canadians invest spare change from everyday purchases into diversified ETF portfolios, with automatic TFSA contribution tracking and a built-in financial literacy program...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("Test This Idea", type="primary", use_container_width=True, disabled=not idea.strip())
    with col2:
        if idea.strip():
            st.caption(f"Will test against {sample_size} personas using stratified sampling")

    if run_button and idea.strip():
        sampled = stratified_sample(personas, sample_size)
        st.info(f"Testing against {len(sampled)} personas (stratified sample from {len(personas)})")

        with st.spinner("Collecting reactions..."):
            reactions = collect_reactions(sampled, idea.strip())

        st.session_state["reactions"] = reactions
        st.session_state["idea"] = idea.strip()

    if "reactions" in st.session_state:
        reactions = st.session_state["reactions"]
        df, analysis = build_analysis(reactions)

        if df is None or df.empty:
            st.error("No valid reactions received. Check your API key and try again.")
            return

        st.divider()
        st.header("Results")
        st.caption(f"Idea: *{st.session_state.get('idea', '')[:150]}...*" if len(st.session_state.get("idea", "")) > 150 else f"Idea: *{st.session_state.get('idea', '')}*")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", "Demographics", "Insights", "Verbatim Quotes", "Raw Data"
        ])

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
# MODE: SURVEY
# ============================================================

def run_survey_mode(personas, sample_size):
    st.title("Survey Your Persona Panel")
    st.markdown("Enter custom questions to ask the persona panel. One question per line. Questions can be about anything \u2014 product feedback, policy opinions, lifestyle, financial habits, etc.")

    questions_text = st.text_area(
        "Survey Questions (one per line)",
        height=200,
        placeholder="Example:\nWould you consider switching banks for better digital features?\nHow important is ESG / responsible investing to you?\nWhat financial topic do you wish you understood better?",
    )

    # Parse questions
    questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]

    if questions:
        st.caption(f"{len(questions)} question(s) detected:")
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}.** {q}")
        if len(questions) > 10:
            st.warning("More than 10 questions may produce longer response times and occasional truncation.")

    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("Run Survey", type="primary", use_container_width=True, disabled=len(questions) == 0)
    with col2:
        if questions:
            st.caption(f"Will survey {sample_size} personas with {len(questions)} question(s)")

    if run_button and questions:
        sampled = stratified_sample(personas, sample_size)
        st.info(f"Surveying {len(sampled)} personas with {len(questions)} questions...")

        with st.spinner("Collecting survey responses..."):
            responses = collect_survey_responses(sampled, questions)

        st.session_state["survey_responses"] = responses
        st.session_state["survey_questions"] = questions

    if "survey_responses" in st.session_state and "survey_questions" in st.session_state:
        responses = st.session_state["survey_responses"]
        questions = st.session_state["survey_questions"]
        df, per_question = build_survey_analysis(responses, questions)

        if df is None or df.empty:
            st.error("No valid survey responses received. Check your API key and try again.")
            return

        st.divider()
        st.header("Survey Results")
        st.caption(f"{df['persona_id'].nunique()} respondents, {len(questions)} questions")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview", "Per-Question Analysis", "Individual Responses", "Raw Data & Export"
        ])

        with tab1:
            show_survey_overview(df, per_question, questions)
        with tab2:
            show_per_question_analysis(df, per_question, questions)
        with tab3:
            show_survey_responses(df, questions)
        with tab4:
            show_survey_data(df)


# ============================================================
# MAIN APP
# ============================================================

def main():
    personas = load_personas()
    sample_size, mode = render_sidebar(personas)

    if mode == "Idea Reactor":
        run_reactor_mode(personas, sample_size)
    else:
        run_survey_mode(personas, sample_size)


if __name__ == "__main__":
    main()
