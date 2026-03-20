import json
import os
import argparse
from collections import defaultdict
from openai import OpenAI

# Setup Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(AGENT_DIR, "data")
SKILLS_DB_PATH = os.path.join(AGENT_DIR, "src", "skills_db.json")

# Load API key from environment or .env file
def load_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = os.path.join(AGENT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.strip().split("=", 1)[1]
    raise ValueError("OPENAI_API_KEY not found. Set it in environment or .env file.")

PROPOSAL_API_KEY = load_api_key()
os.environ["OPENAI_API_KEY"] = PROPOSAL_API_KEY
client = OpenAI(api_key=PROPOSAL_API_KEY)

# ===== META_PROMPT VARIANT 1: GRADUAL GUIDANCE (V0A Track) =====
META_PROMPT_GRADUAL = """
You are an AI Recommender System Architect designing an evolutionary recommendation agent.
Your agent started with a SINGLE, GENERIC, UNIVERSAL seed skill that treats all users identically.
Now, through failure analysis, you must discover latent behavioral patterns and evolve specialized skills.

# Evolution Context
Used Skill: {used_skill}
Target Domain: {domain}
Failure Cluster Size: {failure_count} users

# Representative Failed Case:
- User Behavioral Context: {user_profile}
- Generic Skill Predicted: {predicted_item}
- Ground Truth (what user actually wanted): {ground_truth_item}
- LLM Reasoning: {raw_response}

# Your Task: Pattern Discovery
1. **Identify Behavioral Patterns**: What latent preference patterns link this failure cluster?
   - Do these users prioritize data-driven decision making (specs, comparisons, value analysis)?
   - Do these users prioritize experiential or aesthetic aspects (novelty, design, atmosphere)?
   - Do these users prioritize social validation (popularity, consensus, trends)?
   - Is this a cold-start cluster (no historical data available)?

2. **Counterfactual Reflection**: Why did the GENERIC skill fail for this behavioral type?
   What specific signals were ignored or misweighted?

3. **Mutate Specialized Skill**: Design a NEW skill targeting this behavioral pattern.
   - Name the skill descriptively based on discovered pattern (e.g., "Skill_Data_Driven_Comparer", "Skill_Experience_Seeker", "Skill_Consensus_Validator")
   - Description: Target behavioral profile and context
   - Prompt: Specialized instructions aligned with the pattern
   - Data Strategy: Add specific fields while PRESERVING foundational ones

# CRITICAL CONSTRAINTS:
1. **Anti-Forgetting**: ALWAYS retain foundational fields: `average_rating`, `review_count`, `price`, `stars`, `rating_number`, `ratings_count`
2. **Schema Whitelist**: ONLY use valid field names:
   - Amazon (product): `title`, `description`, `categories`, `price`, `average_rating`, `rating_number`, `attributes`
   - Yelp (business): `name`, `categories`, `attributes`, `stars`, `review_count`
   - Goodreads (book): `title`, `description`, `popular_shelves`, `average_rating`, `ratings_count`
3. **Descriptive Naming**: Name skills to reflect discovered patterns (avoid generic names)

Output JSON:
{{
    "skill_name": "Skill_[Pattern]_[Focus]",
    "discovered_pattern": "brief description of behavioral pattern identified",
    "description": "Target: [behavioral type]. Context: [when to use].",
    "prompt": "You are a [Type] Specialist. [Instructions matching pattern].",
    "data_strategy": ["foundational_fields", "...", "specialized_fields"]
}}
"""

# ===== META_PROMPT VARIANT 2: COGNITIVE GUIDANCE (V0B Track) =====
META_PROMPT_COGNITIVE = """
You are an AI Recommender System Architect designing an evolutionary recommendation agent.
Your agent started with a SINGLE, GENERIC, UNIVERSAL seed skill that treats all users identically.
Now, through failure analysis, you must discover latent COGNITIVE patterns and evolve specialized skills.

# Evolution Context
Used Skill: {used_skill}
Target Domain: {domain}
Failure Cluster Size: {failure_count} users

# Cognitive Profile Analysis
The following cluster shows a dominant cognitive pattern:
{cognitive_summary}

# Representative Failed Case:
- User Cognitive Scores: Analytical={analytical_score:.2f}, Exploratory={exploratory_score:.2f}, Social={social_score:.2f}
- User Behavioral Context: {user_profile}
- Generic Skill Predicted: {predicted_item}
- Ground Truth (what user actually wanted): {ground_truth_item}
- LLM Reasoning: {raw_response}

# Your Task: Cognitive Pattern Discovery
1. **Identify Cognitive Pattern**: What latent psychological preference pattern links this failure cluster?
   - Do these users show HIGH analytical scores (>0.7)? They need spec-focused, comparison-driven ranking.
   - Do these users show HIGH exploratory scores (>0.7)? They need novelty/aesthetic-driven ranking.
   - Do these users show HIGH social scores (>0.7)? They need popularity/consensus-driven ranking.
   - Is this a cold-start cluster (no reviews)? They need safe-bet strategies.

2. **Counterfactual Reflection**: Why did the GENERIC skill fail for this cognitive type?
   What cognitive signals were ignored?

3. **Mutate Specialized Skill**: Design a NEW skill that targets this cognitive pattern.
   Use explicit cognitive naming conventions (see below).

# CRITICAL CONSTRAINTS:
1. **Anti-Forgetting**: ALWAYS retain foundational fields: `average_rating`, `review_count`, `price`, `stars`, `rating_number`, `ratings_count`
2. **Schema Whitelist**: ONLY use valid field names:
   - Amazon (product): `title`, `description`, `categories`, `price`, `average_rating`, `rating_number`, `attributes`
   - Yelp (business): `name`, `categories`, `attributes`, `stars`, `review_count`
   - Goodreads (book): `title`, `description`, `popular_shelves`, `average_rating`, `ratings_count`
3. **Cognitive Naming Convention**: Name skills based on their target cognitive type:
   - Analytical patterns → "Skill_Analytical_[specific_focus]" (e.g., Skill_Analytical_Spec_Comparer)
   - Exploratory patterns → "Skill_Exploratory_[specific_focus]" (e.g., Skill_Exploratory_Aesthetic_Discoverer)
   - Social patterns → "Skill_Social_[specific_focus]" (e.g., Skill_Social_Trend_Follower)
   - Balanced patterns → "Skill_Balanced_[specific_focus]" (for users with no dominant type)
   - Cold-start patterns → "Skill_ColdStart_[specific_strategy]" (e.g., Skill_ColdStart_SafeBet)

Output JSON (MUST include cognitive_type for Router matching):
{{
    "skill_name": "Skill_[Cognitive]_[Specific]",
    "cognitive_type": "analytical|exploratory|social|balanced|cold_start",
    "description": "Target: [cognitive type users]. Context: [when to use].",
    "prompt": "You are a [Cognitive Type] Specialist. [Instructions].",
    "data_strategy": ["foundational_fields", "...", "specialized_fields"]
}}

IMPORTANT: The "cognitive_type" field MUST be one of: "analytical", "exploratory", "social", "balanced", or "cold_start" (lowercase).
This field is used by the Router to match users to skills based on their cognitive profile.
"""


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_cluster_cognitive_pattern(cluster_failures, profiles):
    """Identify dominant cognitive pattern in a failure cluster (for V0B track)."""
    analytical_scores = []
    exploratory_scores = []
    social_scores = []
    cold_start_count = 0

    for f in cluster_failures:
        user_id = f.get('user_id')
        profile = profiles.get(user_id, {})
        analytical_scores.append(profile.get('analytical_score', 0.5))
        exploratory_scores.append(profile.get('exploratory_score', 0.5))
        social_scores.append(profile.get('social_score', 0.5))
        if '[COLD START]' in profile.get('reasoning', ''):
            cold_start_count += 1

    if not analytical_scores:  # Empty cluster
        return {
            "dominant_pattern": "Unknown",
            "avg_analytical": 0.5,
            "avg_exploratory": 0.5,
            "avg_social": 0.5,
            "summary": "Unknown cluster pattern"
        }

    avg_analytical = sum(analytical_scores) / len(analytical_scores)
    avg_exploratory = sum(exploratory_scores) / len(exploratory_scores)
    avg_social = sum(social_scores) / len(social_scores)

    # Determine dominant pattern
    if cold_start_count > len(cluster_failures) * 0.7:
        dominant = "ColdStart"
    elif avg_analytical > 0.7:
        dominant = "Analytical"
    elif avg_exploratory > 0.7:
        dominant = "Exploratory"
    elif avg_social > 0.7:
        dominant = "Social"
    else:
        dominant = "Balanced"

    return {
        "dominant_pattern": dominant,
        "avg_analytical": avg_analytical,
        "avg_exploratory": avg_exploratory,
        "avg_social": avg_social,
        "summary": f"{dominant} cluster (Analytical:{avg_analytical:.2f}, Exploratory:{avg_exploratory:.2f}, Social:{avg_social:.2f})"
    }


def run_evolution(failed_cases_path, output_path=None, use_gradual_prompt=False):
    print(f"Loading failed cases from {failed_cases_path}...")
    failures = load_json(failed_cases_path)
    if not failures:
        print("No failures to process.")
        return

    profiles = load_json(os.path.join(DATA_DIR, "user_profiles.json"))

    # Load current skills_db - ALWAYS preserve existing skills (including Universal seed)
    if output_path and os.path.exists(output_path):
        skills_db = load_json(output_path)  # Load from output path if exists
    else:
        skills_db = load_json(SKILLS_DB_PATH)  # Otherwise load from default

    # Ensure Universal seed is always present
    universal_seed_path = os.path.join(AGENT_DIR, "src", "skills_db_v0_universal_seed.json")
    if os.path.exists(universal_seed_path):
        universal_skills = load_json(universal_seed_path)
        for skill_name, skill_data in universal_skills.items():
            if skill_name not in skills_db:
                skills_db[skill_name] = skill_data
                print(f"Preserved Universal seed: {skill_name}")

    # Select META_PROMPT variant based on experimental track
    if use_gradual_prompt:
        META_PROMPT = META_PROMPT_GRADUAL
        print("Using GRADUAL GUIDANCE META_PROMPT (V0A/V1A track)")
    else:
        META_PROMPT = META_PROMPT_COGNITIVE
        print("Using COGNITIVE GUIDANCE META_PROMPT (V0B/V1B track)")

    # 1. Batch Clustering: Group by (used_skill, domain)
    clusters = defaultdict(list)
    for f in failures:
        domain = f.get('domain') or 'general'
        used_skill = f.get('used_skill') or 'Unknown'
        clusters[(used_skill, domain)].append(f)

    print(f"Found {len(clusters)} failure clusters.")

    # 2. Mutate Skills
    new_skills_count = 0
    total_tokens_used = 0

    for (used_skill, domain), cluster_failures in clusters.items():
        if len(cluster_failures) < 2:
            continue  # Only evolve if there's a pattern (at least 2 failures)

        print(f"\nEvolving for Skill: {used_skill} | Domain: {domain} | Failures: {len(cluster_failures)}")

        # Analyze cognitive pattern (for V0B track)
        cognitive_summary_data = analyze_cluster_cognitive_pattern(cluster_failures, profiles)

        # Take the most representative failure
        sample = cluster_failures[0]
        sample_profile = profiles.get(sample['user_id'], {})
        user_profile = sample_profile.get('reasoning', 'Unknown Profile')

        if use_gradual_prompt:
            # V0A: No cognitive scores in prompt
            prompt = META_PROMPT.format(
                used_skill=used_skill,
                domain=domain,
                failure_count=len(cluster_failures),
                user_profile=user_profile,
                predicted_item=sample.get('predicted_top_1', 'N/A'),
                ground_truth_item=sample.get('ground_truth_item', 'N/A'),
                raw_response=sample.get('raw_response', 'N/A')[:500]
            )
        else:
            # V0B: Include cognitive scores and cluster summary
            prompt = META_PROMPT.format(
                used_skill=used_skill,
                domain=domain,
                failure_count=len(cluster_failures),
                cognitive_summary=cognitive_summary_data['summary'],
                analytical_score=sample_profile.get('analytical_score', 0.5),
                exploratory_score=sample_profile.get('exploratory_score', 0.5),
                social_score=sample_profile.get('social_score', 0.5),
                user_profile=user_profile,
                predicted_item=sample.get('predicted_top_1', 'N/A'),
                ground_truth_item=sample.get('ground_truth_item', 'N/A'),
                raw_response=sample.get('raw_response', 'N/A')[:500]
            )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use gpt-4o-mini for all experiments
                messages=[
                    {"role": "system", "content": "You are a senior AI system architect."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            # Track token usage
            total_tokens_used += response.usage.total_tokens

            new_skill = json.loads(response.choices[0].message.content)
            skill_name = new_skill.pop("skill_name")

            # Remove temporary metadata fields (NOT cognitive_type - it's needed for routing)
            new_skill.pop("discovered_pattern", None)
            new_skill.pop("cognitive_pattern_discovered", None)

            # Normalize cognitive_type to lowercase for consistent matching
            if "cognitive_type" in new_skill:
                new_skill["cognitive_type"] = new_skill["cognitive_type"].lower().replace("-", "_")
            else:
                # Infer cognitive_type from cluster analysis if not provided
                new_skill["cognitive_type"] = cognitive_summary_data['dominant_pattern'].lower().replace("-", "_")

            # 3. Add to Skill Library
            if skill_name not in skills_db:
                skills_db[skill_name] = new_skill
                new_skills_count += 1
                print(f"  -> Generated new skill: {skill_name}")
                print(f"     Cognitive Type: {new_skill.get('cognitive_type', 'unknown')}")
            else:
                print(f"  -> Skill {skill_name} already exists. Skipping.")

        except Exception as e:
            print(f"  -> Failed to evolve skill: {e}")

    # Save updated skills
    output_file = output_path if output_path else SKILLS_DB_PATH
    if new_skills_count > 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(skills_db, f, indent=4, ensure_ascii=False)
        print(f"\nEvolution complete. Added {new_skills_count} new skills to {output_file}.")
        print(f"Total GPT-4 tokens used: {total_tokens_used}")
    else:
        print("\nEvolution complete. No new skills were added (clusters might be too small).")


def main():
    parser = argparse.ArgumentParser(description="Evolve specialized skills from failure clusters")
    parser.add_argument("--version", default="v0", help="Version of failure logs (e.g., v0a_gradual, v0b_cognitive)")
    parser.add_argument("--domain", default="amazon", help="Domain for evolution (amazon, yelp, goodreads)")
    parser.add_argument("--failed_cases", help="Path to failed cases JSON file (overrides --version and --domain)")
    parser.add_argument("--output_dir", default="src", help="Output directory for evolved skills (default: src)")
    args = parser.parse_args()

    # Determine input path
    if args.failed_cases:
        failed_cases_path = args.failed_cases
    else:
        failed_cases_path = os.path.join(DATA_DIR, f"failed_cases_{args.version}_{args.domain}.json")

    # Domain-specific output path: src/skills_db_{domain}.json
    output_path = os.path.join(AGENT_DIR, args.output_dir, f"skills_db_{args.domain}.json")

    # Determine if using gradual or cognitive prompt
    use_gradual = os.environ.get("USE_GRADUAL_PROMPT", "false").lower() == "true"

    print(f"Evolution for domain: {args.domain}")
    print(f"Input: {failed_cases_path}")
    print(f"Output: {output_path}")

    run_evolution(failed_cases_path, output_path=output_path, use_gradual_prompt=use_gradual)


if __name__ == "__main__":
    main()
