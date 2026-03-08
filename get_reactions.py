#!/usr/bin/env python3
"""
Synthetic Persona Reaction Collector
Uses Gemini 2.5 Flash to collect reactions from synthetic Canadian investor personas.

Usage:
  1. Set your API key:  set GEMINI_API_KEY=your_key_here
     Or create a file called .gemini_key with just the key in it.

  2. Run:
     python get_reactions.py "Your investment idea description here"
     python get_reactions.py --sample 100 "Your idea here"
     python get_reactions.py --idea-file idea.txt
     python get_reactions.py --idea-file idea.txt --sample 200
"""

import os
import sys
import json
import csv
import time
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime

import google.generativeai as genai


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-2.5-flash-preview-05-20"
MAX_WORKERS = 10          # Concurrent API calls
REQUESTS_PER_MINUTE = 450  # Stay under 500 RPM free tier limit
PERSONAS_FILE = "personas.json"

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


# ============================================================
# API KEY LOADING
# ============================================================

def load_api_key():
    """Load API key from environment variable or .gemini_key file."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()

    key_file = os.path.join(os.path.dirname(__file__), ".gemini_key")
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            return f.read().strip()

    print("ERROR: No API key found.")
    print("Set it with:  set GEMINI_API_KEY=your_key_here")
    print("Or create a file called .gemini_key with just the key in it.")
    sys.exit(1)


# ============================================================
# PERSONA LOADING & SAMPLING
# ============================================================

def load_personas(filepath):
    """Load personas from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_sample(personas, n):
    """Take a stratified sample ensuring representation across key dimensions."""
    import random
    random.seed(42)

    if n >= len(personas):
        return personas

    # Create strata based on key dimensions
    strata = {}
    for p in personas:
        # Create a composite key for stratification
        age_group = "18-34" if p["age"] < 35 else ("35-54" if p["age"] < 55 else "55+")
        income_group = "low" if p["household_income"] < 60000 else ("mid" if p["household_income"] < 120000 else "high")
        key = (age_group, income_group, p["risk_tolerance"])
        if key not in strata:
            strata[key] = []
        strata[key].append(p)

    # Sample proportionally from each stratum
    sampled = []
    total = len(personas)
    for key, members in strata.items():
        stratum_n = max(1, round(len(members) / total * n))
        sampled.extend(random.sample(members, min(stratum_n, len(members))))

    # If we have too many or too few, adjust
    if len(sampled) > n:
        sampled = random.sample(sampled, n)
    elif len(sampled) < n:
        remaining = [p for p in personas if p not in sampled]
        sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))

    return sampled[:n]


# ============================================================
# API CALLS
# ============================================================

def get_reaction(model, persona, idea, persona_index, total):
    """Get a single persona's reaction via Gemini API."""
    prompt = REACTION_PROMPT.format(persona=persona["persona_summary"], idea=idea)

    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.8,
                    max_output_tokens=500,
                ),
            )

            # Parse JSON response
            text = response.text.strip()
            # Clean potential markdown fences
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            reaction = json.loads(text)

            # Add persona metadata
            reaction["persona_id"] = persona["id"]
            reaction["persona_name"] = f"{persona['first_name']} {persona['last_name']}"
            reaction["age"] = persona["age"]
            reaction["gender"] = persona["gender"]
            reaction["ethnicity"] = persona["ethnicity"]
            reaction["province"] = persona["province"]
            reaction["income"] = persona["household_income"]
            reaction["net_worth"] = persona["net_worth"]
            reaction["risk_tolerance_profile"] = persona["risk_tolerance"]
            reaction["life_stage"] = persona["life_stage"]
            reaction["investment_knowledge"] = persona["investment_knowledge"]

            return reaction

        except json.JSONDecodeError as e:
            if attempt < 2:
                time.sleep(1)
                continue
            return {
                "persona_id": persona["id"],
                "persona_name": f"{persona['first_name']} {persona['last_name']}",
                "error": f"JSON parse error: {str(e)}",
                "raw_response": response.text[:500] if response else "No response",
            }
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = (attempt + 1) * 10
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt < 2:
                time.sleep(2)
                continue
            return {
                "persona_id": persona["id"],
                "persona_name": f"{persona['first_name']} {persona['last_name']}",
                "error": str(e),
            }


def collect_reactions(personas, idea, model):
    """Collect reactions from all personas with concurrency."""
    total = len(personas)
    reactions = []
    errors = 0
    completed = 0

    # Calculate delay between batches to stay under rate limit
    delay_per_request = 60.0 / REQUESTS_PER_MINUTE

    print(f"\nCollecting reactions from {total} personas...")
    print(f"Model: {MODEL_NAME}")
    print(f"Concurrency: {MAX_WORKERS} workers")
    print(f"Estimated time: {total * delay_per_request / MAX_WORKERS:.0f}-{total * delay_per_request / MAX_WORKERS * 2:.0f} seconds\n")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_persona = {}
        for i, persona in enumerate(personas):
            future = executor.submit(get_reaction, model, persona, idea, i, total)
            future_to_persona[future] = persona
            # Small delay between submissions to avoid bursting
            time.sleep(delay_per_request)

        # Collect results
        for future in as_completed(future_to_persona):
            completed += 1
            result = future.result()
            reactions.append(result)

            if "error" in result:
                errors += 1
                print(f"  [{completed}/{total}] ERROR for {result['persona_name']}: {result['error'][:60]}")
            else:
                score = result.get("interest_score", "?")
                sentiment = result.get("sentiment", "?")
                name = result["persona_name"]
                print(f"  [{completed}/{total}] {name}: score={score}/10, {sentiment}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({errors} errors out of {total})")

    return reactions


# ============================================================
# ANALYSIS & OUTPUT
# ============================================================

def analyze_reactions(reactions):
    """Analyze aggregate results."""
    valid = [r for r in reactions if "error" not in r]
    if not valid:
        print("No valid reactions to analyze.")
        return {}

    n = len(valid)

    # Overall metrics
    scores = [r["interest_score"] for r in valid]
    avg_score = sum(scores) / n
    sentiments = Counter(r["sentiment"] for r in valid)
    would_invest = sum(1 for r in valid if r.get("would_invest"))
    amounts = Counter(r.get("investment_amount", "unknown") for r in valid)

    # All concerns and appeals flattened
    all_concerns = []
    all_appeals = []
    for r in valid:
        all_concerns.extend(r.get("key_concerns", []))
        all_appeals.extend(r.get("appeal_factors", []))
    top_concerns = Counter(all_concerns).most_common(15)
    top_appeals = Counter(all_appeals).most_common(15)

    # Segment analysis
    def segment_score(key, value):
        seg = [r for r in valid if r.get(key) == value]
        if not seg:
            return None
        return sum(r["interest_score"] for r in seg) / len(seg), len(seg)

    # Analysis by age group
    age_analysis = {}
    for r in valid:
        age = r["age"]
        group = "18-34" if age < 35 else ("35-54" if age < 55 else "55+")
        if group not in age_analysis:
            age_analysis[group] = {"scores": [], "would_invest": 0, "count": 0}
        age_analysis[group]["scores"].append(r["interest_score"])
        age_analysis[group]["would_invest"] += 1 if r.get("would_invest") else 0
        age_analysis[group]["count"] += 1

    # Analysis by risk tolerance
    risk_analysis = {}
    for r in valid:
        rt = r.get("risk_tolerance_profile", "Unknown")
        if rt not in risk_analysis:
            risk_analysis[rt] = {"scores": [], "would_invest": 0, "count": 0}
        risk_analysis[rt]["scores"].append(r["interest_score"])
        risk_analysis[rt]["would_invest"] += 1 if r.get("would_invest") else 0
        risk_analysis[rt]["count"] += 1

    # Analysis by income bracket
    income_analysis = {}
    for r in valid:
        inc = r.get("income", 0)
        bracket = "Under $60K" if inc < 60000 else ("$60K-$120K" if inc < 120000 else "$120K+")
        if bracket not in income_analysis:
            income_analysis[bracket] = {"scores": [], "would_invest": 0, "count": 0}
        income_analysis[bracket]["scores"].append(r["interest_score"])
        income_analysis[bracket]["would_invest"] += 1 if r.get("would_invest") else 0
        income_analysis[bracket]["count"] += 1

    # Analysis by province (top 5)
    province_analysis = {}
    for r in valid:
        prov = r.get("province", "Unknown")
        if prov not in province_analysis:
            province_analysis[prov] = {"scores": [], "would_invest": 0, "count": 0}
        province_analysis[prov]["scores"].append(r["interest_score"])
        province_analysis[prov]["would_invest"] += 1 if r.get("would_invest") else 0
        province_analysis[prov]["count"] += 1

    # Print report
    print(f"\n{'='*70}")
    print(f"  REACTION ANALYSIS REPORT")
    print(f"  Valid responses: {n} / {len(reactions)}")
    print(f"{'='*70}")

    print(f"\n  OVERALL METRICS:")
    print(f"    Average Interest Score:  {avg_score:.1f} / 10")
    print(f"    Would Invest:            {would_invest}/{n} ({would_invest/n*100:.1f}%)")
    print(f"    Sentiment:               ", end="")
    for sent, count in sentiments.most_common():
        print(f"{sent}: {count} ({count/n*100:.0f}%)  ", end="")
    print()

    print(f"\n  INVESTMENT AMOUNT WILLINGNESS:")
    for amt, count in amounts.most_common():
        bar = "#" * int(count / n * 40)
        print(f"    {amt:30s} {count:4d} ({count/n*100:5.1f}%) {bar}")

    print(f"\n  BY AGE GROUP:")
    for group in ["18-34", "35-54", "55+"]:
        if group in age_analysis:
            data = age_analysis[group]
            avg = sum(data["scores"]) / data["count"]
            inv_pct = data["would_invest"] / data["count"] * 100
            print(f"    {group:10s}  avg score: {avg:.1f}  would invest: {inv_pct:.0f}%  (n={data['count']})")

    print(f"\n  BY RISK TOLERANCE:")
    for rt in ["Very Conservative", "Conservative", "Moderate", "Growth", "Aggressive"]:
        if rt in risk_analysis:
            data = risk_analysis[rt]
            avg = sum(data["scores"]) / data["count"]
            inv_pct = data["would_invest"] / data["count"] * 100
            print(f"    {rt:22s}  avg score: {avg:.1f}  would invest: {inv_pct:.0f}%  (n={data['count']})")

    print(f"\n  BY INCOME BRACKET:")
    for bracket in ["Under $60K", "$60K-$120K", "$120K+"]:
        if bracket in income_analysis:
            data = income_analysis[bracket]
            avg = sum(data["scores"]) / data["count"]
            inv_pct = data["would_invest"] / data["count"] * 100
            print(f"    {bracket:15s}  avg score: {avg:.1f}  would invest: {inv_pct:.0f}%  (n={data['count']})")

    print(f"\n  BY PROVINCE (top 5):")
    sorted_provs = sorted(province_analysis.items(), key=lambda x: -x[1]["count"])[:5]
    for prov, data in sorted_provs:
        avg = sum(data["scores"]) / data["count"]
        inv_pct = data["would_invest"] / data["count"] * 100
        print(f"    {prov:30s}  avg score: {avg:.1f}  would invest: {inv_pct:.0f}%  (n={data['count']})")

    print(f"\n  TOP CONCERNS:")
    for concern, count in top_concerns[:10]:
        print(f"    - {concern} ({count} mentions)")

    print(f"\n  TOP APPEAL FACTORS:")
    for appeal, count in top_appeals[:10]:
        print(f"    - {appeal} ({count} mentions)")

    # Sample verbatim quotes
    print(f"\n  SAMPLE VERBATIM QUOTES:")
    import random
    random.seed(99)
    quote_sample = random.sample(valid, min(5, len(valid)))
    for r in quote_sample:
        print(f"    [{r['persona_name']}, age {r['age']}, {r['province']}]")
        print(f"    \"{r.get('verbatim_quote', 'N/A')}\"")
        print()

    return {
        "avg_interest_score": round(avg_score, 2),
        "would_invest_pct": round(would_invest / n * 100, 1),
        "sentiment_distribution": dict(sentiments),
        "amount_distribution": dict(amounts),
        "age_analysis": {k: {"avg_score": round(sum(v["scores"])/v["count"], 2), "would_invest_pct": round(v["would_invest"]/v["count"]*100, 1), "n": v["count"]} for k, v in age_analysis.items()},
        "risk_analysis": {k: {"avg_score": round(sum(v["scores"])/v["count"], 2), "would_invest_pct": round(v["would_invest"]/v["count"]*100, 1), "n": v["count"]} for k, v in risk_analysis.items()},
        "top_concerns": [{"concern": c, "count": n} for c, n in top_concerns[:10]],
        "top_appeals": [{"appeal": a, "count": n} for a, n in top_appeals[:10]],
    }


def write_results(reactions, analysis, idea, output_prefix):
    """Write results to CSV and JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"{output_prefix}_{timestamp}.csv"
    json_file = f"{output_prefix}_{timestamp}.json"

    # CSV - flatten for easy analysis
    valid = [r for r in reactions if "error" not in r]
    if valid:
        csv_fields = [
            "persona_id", "persona_name", "age", "gender", "ethnicity", "province",
            "income", "net_worth", "risk_tolerance_profile", "life_stage",
            "investment_knowledge", "interest_score", "sentiment", "gut_reaction",
            "key_concerns", "appeal_factors", "would_invest", "investment_amount",
            "what_would_help", "verbatim_quote",
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for r in valid:
                row = dict(r)
                for key in ("key_concerns", "appeal_factors"):
                    if isinstance(row.get(key), list):
                        row[key] = "; ".join(row[key])
                writer.writerow(row)

    # JSON - full data with analysis
    output = {
        "idea": idea,
        "timestamp": timestamp,
        "total_personas": len(reactions),
        "valid_responses": len(valid),
        "analysis": analysis,
        "reactions": reactions,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to:")
    print(f"  CSV:  {csv_file}")
    print(f"  JSON: {json_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Collect synthetic persona reactions to investment ideas")
    parser.add_argument("idea", nargs="?", help="The investment idea to test (in quotes)")
    parser.add_argument("--idea-file", help="Read idea from a text file instead")
    parser.add_argument("--sample", type=int, default=0, help="Use a stratified sample of N personas (default: all 1000)")
    parser.add_argument("--output", default="reactions", help="Output file prefix (default: reactions)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Concurrent API workers (default: {MAX_WORKERS})")
    parser.add_argument("--personas", default=PERSONAS_FILE, help=f"Personas JSON file (default: {PERSONAS_FILE})")
    args = parser.parse_args()

    # Get the idea
    if args.idea_file:
        with open(args.idea_file, "r", encoding="utf-8") as f:
            idea = f.read().strip()
    elif args.idea:
        idea = args.idea
    else:
        print("Enter your investment idea (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                if lines:
                    break
                continue
            lines.append(line)
        idea = "\n".join(lines)

    if not idea:
        print("ERROR: No idea provided.")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  SYNTHETIC PERSONA REACTION COLLECTOR")
    print(f"{'='*70}")
    print(f"\nIDEA: {idea[:200]}{'...' if len(idea) > 200 else ''}")

    # Load API key
    api_key = load_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    # Load personas
    personas = load_personas(args.personas)
    print(f"Loaded {len(personas)} personas from {args.personas}")

    # Sample if requested
    if args.sample > 0:
        personas = stratified_sample(personas, args.sample)
        print(f"Using stratified sample of {len(personas)} personas")

    # Update workers if specified
    global MAX_WORKERS
    MAX_WORKERS = args.workers

    # Collect reactions
    reactions = collect_reactions(personas, idea, model)

    # Analyze
    analysis = analyze_reactions(reactions)

    # Write output
    write_results(reactions, analysis, idea, args.output)

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
