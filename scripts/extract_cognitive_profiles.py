import argparse
import glob
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

from openai import OpenAI
from tqdm import tqdm


# PAE_katana structure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(AGENT_DIR, "dataset")
REVIEW_JSON_PATH = os.path.join(DATASET_PATH, "review.json")
TRACK2_DIR = os.path.join(AGENT_DIR, "tasks", "track2")
DEV_TASKS_DIR = os.path.join(AGENT_DIR, "data", "problem_splits", "classic_3000")
OUTPUT_DIR = os.path.join(AGENT_DIR, "data")
PROFILES_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "user_profiles.json")

DEFAULT_DOMAINS = ["amazon", "yelp", "goodreads"]
DEFAULT_SOURCES = ["train", "track2"]
MAX_REVIEWS_PER_USER = 10
PROFILE_VERSION = "hybrid_cached_v1"
RATE_LIMIT_PATTERNS = ("429", "too many requests", "rate limit", "rate_limit")
MAX_QUOTES_PER_DIMENSION = 2
MAX_SAMPLE_INTERACTIONS = 3

DIMENSION_KEYWORDS = {
    "analytical": [
        "accuracy",
        "affordable",
        "battery",
        "budget",
        "build quality",
        "compare",
        "comparison",
        "cost",
        "detailed",
        "durable",
        "feature",
        "material",
        "performance",
        "price",
        "quality",
        "reliable",
        "size",
        "spec",
        "technical",
        "value",
        "worth",
    ],
    "exploratory": [
        "aesthetic",
        "atmosphere",
        "beautiful",
        "creative",
        "cute",
        "different",
        "discover",
        "distinct",
        "emotional",
        "exciting",
        "experiment",
        "explore",
        "fresh",
        "fun",
        "interesting",
        "new",
        "novel",
        "special",
        "style",
        "unique",
        "vibe",
    ],
    "social": [
        "best seller",
        "bestseller",
        "crowd",
        "everyone",
        "famous",
        "family",
        "friends",
        "highly rated",
        "popular",
        "rating",
        "ratings",
        "recommend",
        "recommended",
        "review",
        "reviews",
        "social proof",
        "top rated",
        "trend",
        "trending",
    ],
}

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

REQUEST_LOCK = Lock()
NEXT_REQUEST_TS = 0.0

SYSTEM_PROMPT = """
You are an Expert Consumer Psychologist refining a theory-grounded Consumer Decision Profile.

You are NOT reading raw full reviews. Instead, you are given a compact evidence summary produced from historical reviews.
Use that compact evidence plus the heuristic prior to infer three literature-grounded behavioral proxy scores:

1. analytical_score
   Proxy for Need for Cognition / elaborative processing tendency.
2. exploratory_score
   Proxy for Consumers' Need for Uniqueness / exploratory buying tendency.
3. social_score
   Proxy for Susceptibility to Interpersonal Influence / social-proof reliance.

Rules:
- Output three independent scores between 0.00 and 1.00.
- Respect the compact evidence; do not invent unsupported traits.
- The heuristic prior is informative but not binding.
- If evidence is weak or mixed, keep scores moderate.
- Keep reasoning concise, concrete, and grounded in the provided evidence summary.

Output MUST be valid JSON:
{
  "analytical_score": float,
  "exploratory_score": float,
  "social_score": float,
  "confidence": float,
  "reasoning": "2-3 sentence explanation"
}
"""


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_profiles(path, profiles):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(profiles, file, indent=4, ensure_ascii=False)


def clamp_score(value):
    return max(0.0, min(1.0, round(float(value), 3)))


def normalize_text(text):
    return " ".join(str(text or "").strip().lower().split())


def shorten_text(text, max_chars=160):
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract cognitive profiles for train/track2 users from official dataset reviews.")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS, choices=DEFAULT_DOMAINS)
    parser.add_argument(
        "--task-root",
        default=DEV_TASKS_DIR,
        help="Task root that contains train/ and optionally val/ subdirectories.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        choices=["train", "val", "dev_train", "dev_val", "track2"],
        help="Which task sources to cover when collecting target users.",
    )
    parser.add_argument("--output", default=PROFILES_OUTPUT_PATH)
    parser.add_argument("--max-reviews-per-user", type=int, default=MAX_REVIEWS_PER_USER)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--min-request-interval", type=float, default=0.8)
    parser.add_argument("--rate-limit-base-sleep", type=float, default=6.0)
    parser.add_argument("--rate-limit-max-sleep", type=float, default=45.0)
    parser.add_argument("--llm-refine-threshold", type=float, default=0.72)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--limit-users", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--report-only", action="store_true", help="Only print/write coverage report without calling the API.")
    parser.add_argument("--coverage-report", default=None, help="Optional JSON path to save coverage metadata.")
    return parser.parse_args()


def get_cold_start_profile():
    return {
        "analytical_score": 0.5,
        "exploratory_score": 0.5,
        "social_score": 0.5,
        "reasoning": "[COLD START] The user has 0 historical reviews. Reverting to a balanced default across elaborative, exploratory, and social-influence tendencies.",
        "confidence": 0.15,
        "source": "cold_start",
    }


def collect_users_from_task_dir(tasks_dir):
    users = set()
    if not os.path.isdir(tasks_dir):
        return users
    for task_path in sorted(glob.glob(os.path.join(tasks_dir, "task_*.json"))):
        with open(task_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        user_id = payload.get("user_id")
        if user_id:
            users.add(user_id)
    return users


def normalize_source_name(source_name):
    if source_name == "dev_train":
        return "train"
    if source_name == "dev_val":
        return "val"
    return source_name


def collect_target_users(domains, sources, task_root):
    users = set()
    coverage = {}

    for domain in domains:
        source_map = {}
        normalized_sources = [normalize_source_name(source_name) for source_name in sources]
        if "train" in normalized_sources:
            source_map["train"] = collect_users_from_task_dir(os.path.join(task_root, "train", domain, "tasks"))
        if "val" in normalized_sources:
            source_map["val"] = collect_users_from_task_dir(os.path.join(task_root, "val", domain, "tasks"))
        if "track2" in normalized_sources:
            source_map["track2"] = collect_users_from_task_dir(os.path.join(TRACK2_DIR, domain, "tasks"))

        domain_users = set().union(*source_map.values()) if source_map else set()
        coverage[domain] = {
            "sources": {source_name: len(source_users) for source_name, source_users in source_map.items()},
            "unique_users": len(domain_users),
        }
        users.update(domain_users)

    return users, coverage


def review_sort_key(review):
    candidates = [
        review.get("timestamp"),
        review.get("time"),
        review.get("date"),
        review.get("date_updated"),
        review.get("date_added"),
        review.get("read_at"),
        review.get("started_at"),
    ]
    values = []
    for value in candidates:
        if value is None or value == "":
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
            continue
        text = str(value).strip()
        if text.isdigit():
            values.append(float(text))
            continue
        try:
            values.append(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
            continue
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%a %b %d %H:%M:%S %z %Y"):
            try:
                values.append(datetime.strptime(text, fmt).timestamp())
                break
            except Exception:
                continue
    if values:
        return max(values)
    return float("-inf")


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_user_reviews(target_users):
    print(f"Loading reviews from {REVIEW_JSON_PATH}...")
    if not os.path.exists(REVIEW_JSON_PATH):
        raise FileNotFoundError(f"Review file not found: {REVIEW_JSON_PATH}")

    user_reviews = defaultdict(list)
    with open(REVIEW_JSON_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = record.get("user_id")
            if user_id in target_users:
                user_reviews[user_id].append(record)
    return user_reviews


def get_recent_reviews(reviews, max_reviews_per_user):
    return sorted(reviews, key=review_sort_key, reverse=True)[:max_reviews_per_user]


def build_review_signature(reviews, max_reviews_per_user):
    recent_reviews = get_recent_reviews(reviews, max_reviews_per_user)
    normalized = []
    for review in recent_reviews:
        normalized.append(
            {
                "source": str(review.get("source", "")),
                "type": str(review.get("type", "")),
                "title": shorten_text(review.get("title", ""), max_chars=180),
                "text": shorten_text(review.get("text", ""), max_chars=400),
                "stars": review.get("stars"),
                "time": review_sort_key(review),
            }
        )
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scan_dimension_hits(text, dimension):
    hits = Counter()
    for keyword in DIMENSION_KEYWORDS[dimension]:
        count = text.count(keyword)
        if count > 0:
            hits[keyword] = count
    return hits


def build_compact_evidence(reviews, max_reviews_per_user):
    recent_reviews = get_recent_reviews(reviews, max_reviews_per_user)
    rating_values = []
    source_counter = Counter()
    dimension_keywords = {dimension: Counter() for dimension in DIMENSION_KEYWORDS}
    dimension_review_counts = {dimension: 0 for dimension in DIMENSION_KEYWORDS}
    dimension_quotes = {dimension: [] for dimension in DIMENSION_KEYWORDS}
    sample_interactions = []

    for review in recent_reviews:
        rating_value = safe_float(review.get("stars"))
        if rating_value is not None:
            rating_values.append(rating_value)

        source_name = str(review.get("source", "unknown"))
        source_counter[source_name] += 1

        title_text = str(review.get("title", ""))
        body_text = str(review.get("text", ""))
        merged_text = normalize_text(f"{title_text} {body_text}")

        if len(sample_interactions) < MAX_SAMPLE_INTERACTIONS:
            sample_interactions.append(
                {
                    "source": source_name,
                    "rating": rating_value,
                    "snippet": shorten_text(f"{title_text} {body_text}", max_chars=180),
                }
            )

        for dimension in DIMENSION_KEYWORDS:
            review_hits = scan_dimension_hits(merged_text, dimension)
            if review_hits:
                dimension_review_counts[dimension] += 1
                dimension_keywords[dimension].update(review_hits)
                if len(dimension_quotes[dimension]) < MAX_QUOTES_PER_DIMENSION:
                    dimension_quotes[dimension].append(shorten_text(body_text or title_text, max_chars=180))

    average_rating = round(sum(rating_values) / len(rating_values), 3) if rating_values else None
    if rating_values:
        variance = sum((value - average_rating) ** 2 for value in rating_values) / len(rating_values)
        rating_variance = round(variance, 3)
    else:
        rating_variance = None

    dimension_evidence = {}
    for dimension in DIMENSION_KEYWORDS:
        top_keywords = [keyword for keyword, _ in dimension_keywords[dimension].most_common(6)]
        total_hits = sum(dimension_keywords[dimension].values())
        dimension_evidence[dimension] = {
            "total_hits": total_hits,
            "matched_review_count": dimension_review_counts[dimension],
            "top_keywords": top_keywords,
            "quotes": dimension_quotes[dimension],
        }

    return {
        "review_count_total": len(reviews),
        "review_count_used": len(recent_reviews),
        "average_rating_used": average_rating,
        "rating_variance_used": rating_variance,
        "source_distribution": dict(source_counter.most_common()),
        "dimension_evidence": dimension_evidence,
        "sample_interactions": sample_interactions,
    }


def build_heuristic_profile(compact_evidence):
    if compact_evidence["review_count_total"] == 0:
        return get_cold_start_profile()

    raw_scores = {}
    total_hits = 0
    for dimension, evidence in compact_evidence["dimension_evidence"].items():
        raw_value = evidence["total_hits"] + 0.6 * evidence["matched_review_count"]
        raw_scores[dimension] = raw_value
        total_hits += evidence["total_hits"]

    total_raw = sum(raw_scores.values())
    if total_raw <= 0:
        return {
            "analytical_score": 0.5,
            "exploratory_score": 0.5,
            "social_score": 0.5,
            "confidence": clamp_score(0.22 + 0.03 * min(compact_evidence["review_count_used"], 4)),
            "source": "heuristic_direct",
            "reasoning": "[HEURISTIC] The available review evidence is sparse or neutral, so the user is treated as broadly balanced across analytical, exploratory, and social tendencies.",
        }

    scores = {
        dimension: clamp_score(0.18 + 0.64 * (raw_scores[dimension] / total_raw))
        for dimension in raw_scores
    }
    sorted_dimensions = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    dominant_dimension = sorted_dimensions[0][0]
    second_value = sorted_dimensions[1][1] if len(sorted_dimensions) > 1 else 0.0
    margin = (sorted_dimensions[0][1] - second_value) / total_raw if total_raw else 0.0
    confidence = clamp_score(
        0.3
        + 0.035 * min(total_hits, 12)
        + 0.03 * min(compact_evidence["review_count_used"], 6)
        + 0.25 * margin
    )

    dominant_keywords = compact_evidence["dimension_evidence"][dominant_dimension]["top_keywords"][:3]
    dominant_keywords_text = ", ".join(dominant_keywords) if dominant_keywords else "mixed evidence"
    reasoning = (
        f"[HEURISTIC] The user most strongly shows {dominant_dimension} tendencies. "
        f"Recent reviews repeatedly emphasize {dominant_keywords_text}, which is why this dimension receives the highest prior score."
    )

    return {
        "analytical_score": scores["analytical"],
        "exploratory_score": scores["exploratory"],
        "social_score": scores["social"],
        "confidence": confidence,
        "source": "heuristic_direct",
        "reasoning": reasoning,
    }


def should_refine_with_llm(heuristic_profile, compact_evidence, llm_refine_threshold):
    if heuristic_profile["source"] == "cold_start":
        return False

    evidence_map = compact_evidence["dimension_evidence"]
    total_hits = sum(evidence["total_hits"] for evidence in evidence_map.values())
    ordered_scores = sorted(
        [
            heuristic_profile["analytical_score"],
            heuristic_profile["exploratory_score"],
            heuristic_profile["social_score"],
        ],
        reverse=True,
    )
    margin = ordered_scores[0] - ordered_scores[1]

    if heuristic_profile["confidence"] >= llm_refine_threshold and total_hits >= 4 and margin >= 0.08:
        return False
    return True


def is_rate_limit_error(error):
    error_text = str(error).lower()
    return any(pattern in error_text for pattern in RATE_LIMIT_PATTERNS)


def wait_for_request_slot(min_request_interval):
    global NEXT_REQUEST_TS
    if min_request_interval <= 0:
        return

    with REQUEST_LOCK:
        now = time.monotonic()
        if now < NEXT_REQUEST_TS:
            time.sleep(NEXT_REQUEST_TS - now)
        NEXT_REQUEST_TS = time.monotonic() + min_request_interval


def finalize_profile(core_profile, signature, review_count, source_override=None):
    final_source = source_override or core_profile.get("source", "unknown")
    return {
        "analytical_score": clamp_score(core_profile.get("analytical_score", 0.5)),
        "exploratory_score": clamp_score(core_profile.get("exploratory_score", 0.5)),
        "social_score": clamp_score(core_profile.get("social_score", 0.5)),
        "reasoning": str(core_profile.get("reasoning", "")),
        "confidence": clamp_score(core_profile.get("confidence", 0.5)),
        "source": final_source,
        "profile_version": PROFILE_VERSION,
        "review_signature": signature,
        "review_count_used": int(review_count),
    }


def migrate_cached_profile(profile, reviews, signature, max_reviews_per_user):
    if not profile:
        return None

    review_count_used = min(len(reviews), max_reviews_per_user)
    reasoning = str(profile.get("reasoning", ""))
    if not reviews:
        cold_profile = get_cold_start_profile()
        return finalize_profile(cold_profile, signature, review_count_used, source_override="cold_start")

    if reasoning.startswith("[COLD START]") or reasoning.startswith("[HEURISTIC FALLBACK]") or reasoning.startswith("[RATE LIMIT FALLBACK]"):
        return profile

    migrated = dict(profile)
    migrated["confidence"] = clamp_score(migrated.get("confidence", 0.82))
    migrated["source"] = str(migrated.get("source", "legacy_cached"))
    migrated["profile_version"] = str(migrated.get("profile_version", PROFILE_VERSION))
    migrated["review_signature"] = signature
    migrated["review_count_used"] = review_count_used
    return migrated


def profile_needs_refresh(user_id, profile, reviews, signature, force_refresh):
    if profile is None:
        return True

    reasoning = str(profile.get("reasoning", ""))
    has_reviews = bool(reviews)

    if force_refresh:
        return True
    if not has_reviews:
        return reasoning.startswith("[COLD START]") is False
    if reasoning.startswith("[COLD START]") or reasoning.startswith("[HEURISTIC FALLBACK]") or reasoning.startswith("[RATE LIMIT FALLBACK]"):
        return True

    cached_signature = profile.get("review_signature")
    if cached_signature and cached_signature != signature:
        return True
    return False


def extract_profile_with_retry(plan, max_retries, min_request_interval, rate_limit_base_sleep, rate_limit_max_sleep):
    compact_payload = {
        "heuristic_prior": {
            "analytical_score": plan["heuristic_profile"]["analytical_score"],
            "exploratory_score": plan["heuristic_profile"]["exploratory_score"],
            "social_score": plan["heuristic_profile"]["social_score"],
            "confidence": plan["heuristic_profile"]["confidence"],
        },
        "compact_evidence": plan["compact_evidence"],
    }
    user_content = json.dumps(compact_payload, ensure_ascii=False, indent=2)

    for attempt in range(max_retries):
        try:
            wait_for_request_slot(min_request_interval)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            profile_json = json.loads(response.choices[0].message.content)
            final_profile = finalize_profile(
                {
                    "analytical_score": profile_json.get("analytical_score", plan["heuristic_profile"]["analytical_score"]),
                    "exploratory_score": profile_json.get("exploratory_score", plan["heuristic_profile"]["exploratory_score"]),
                    "social_score": profile_json.get("social_score", plan["heuristic_profile"]["social_score"]),
                    "confidence": profile_json.get("confidence", max(plan["heuristic_profile"]["confidence"], 0.7)),
                    "reasoning": profile_json.get("reasoning", plan["heuristic_profile"]["reasoning"]),
                    "source": "llm_refined",
                },
                plan["signature"],
                plan["review_count_used"],
                source_override="llm_refined",
            )
            return plan["user_id"], final_profile
        except Exception as error:
            if is_rate_limit_error(error):
                sleep_seconds = min(
                    rate_limit_max_sleep,
                    rate_limit_base_sleep * (2 ** attempt) + random.uniform(0, 1.5),
                )
                print(
                    f"\n[RateLimit] user={plan['user_id']} attempt={attempt + 1}/{max_retries} "
                    f"sleep={sleep_seconds:.1f}s error={error}"
                )
                time.sleep(sleep_seconds)
            else:
                print(f"\n[Error] Failed to refine profile for user {plan['user_id']}: {error}")
                fallback = finalize_profile(plan["heuristic_profile"], plan["signature"], plan["review_count_used"])
                fallback["source"] = "heuristic_fallback"
                fallback["reasoning"] = "[HEURISTIC FALLBACK] " + plan["heuristic_profile"]["reasoning"].replace("[HEURISTIC] ", "")
                return plan["user_id"], fallback

    print(
        f"\n[Error] Failed to refine profile for user {plan['user_id']} after {max_retries} retries due to rate limits. "
        "Using heuristic fallback."
    )
    fallback = finalize_profile(plan["heuristic_profile"], plan["signature"], plan["review_count_used"])
    fallback["source"] = "heuristic_fallback"
    fallback["reasoning"] = "[RATE LIMIT FALLBACK] " + plan["heuristic_profile"]["reasoning"].replace("[HEURISTIC] ", "")
    return plan["user_id"], fallback


def build_coverage_report(target_users, domain_coverage, profiles, user_reviews, args):
    covered_users = sum(1 for user_id in target_users if user_id in profiles)
    users_with_reviews = sum(1 for user_id in target_users if user_reviews.get(user_id))
    source_breakdown = Counter()
    for user_id in target_users:
        profile = profiles.get(user_id)
        if profile:
            source_breakdown[str(profile.get("source", "unknown"))] += 1

    report = {
        "domains": args.domains,
        "sources": [normalize_source_name(source_name) for source_name in args.sources],
        "task_root": os.path.abspath(args.task_root),
        "target_user_count": len(target_users),
        "existing_profile_count": covered_users,
        "existing_profile_coverage": covered_users / len(target_users) if target_users else 0.0,
        "users_with_reviews": users_with_reviews,
        "users_with_reviews_ratio": users_with_reviews / len(target_users) if target_users else 0.0,
        "cold_start_candidates": len(target_users) - users_with_reviews,
        "profile_source_breakdown": dict(source_breakdown),
        "profile_version": PROFILE_VERSION,
        "per_domain": domain_coverage,
    }
    return report


def main():
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    target_users, domain_coverage = collect_target_users(args.domains, args.sources, args.task_root)
    target_users = sorted(target_users)
    if args.limit_users is not None:
        target_users = target_users[: args.limit_users]

    print(f"Target users collected: {len(target_users)}")
    for domain, meta in domain_coverage.items():
        print(f"  {domain}: unique={meta['unique_users']} sources={meta['sources']}")

    profiles = load_json(args.output)
    if profiles:
        print(f"Existing profiles loaded: {len(profiles)}")

    user_reviews = load_user_reviews(set(target_users))

    heuristic_direct_count = 0
    llm_refine_count = 0
    stale_count = 0
    users_to_process = []

    for user_id in target_users:
        reviews = user_reviews.get(user_id, [])
        signature = build_review_signature(reviews, args.max_reviews_per_user)
        migrated_profile = migrate_cached_profile(profiles.get(user_id), reviews, signature, args.max_reviews_per_user)
        if migrated_profile is not None:
            profiles[user_id] = migrated_profile

        if not profile_needs_refresh(user_id, profiles.get(user_id), reviews, signature, args.force_refresh):
            continue

        stale_count += int(user_id in profiles)
        review_count_used = min(len(reviews), args.max_reviews_per_user)

        if not reviews:
            profiles[user_id] = finalize_profile(get_cold_start_profile(), signature, review_count_used, source_override="cold_start")
            heuristic_direct_count += 1
            continue

        compact_evidence = build_compact_evidence(reviews, args.max_reviews_per_user)
        heuristic_profile = build_heuristic_profile(compact_evidence)
        heuristic_profile = finalize_profile(heuristic_profile, signature, review_count_used)

        if should_refine_with_llm(heuristic_profile, compact_evidence, args.llm_refine_threshold):
            users_to_process.append(
                {
                    "user_id": user_id,
                    "compact_evidence": compact_evidence,
                    "heuristic_profile": heuristic_profile,
                    "signature": signature,
                    "review_count_used": review_count_used,
                }
            )
            llm_refine_count += 1
        else:
            profiles[user_id] = heuristic_profile
            heuristic_direct_count += 1

    coverage_report = build_coverage_report(target_users, domain_coverage, profiles, user_reviews, args)
    print(
        "Coverage snapshot: "
        f"profiles={coverage_report['existing_profile_count']}/{coverage_report['target_user_count']} "
        f"({coverage_report['existing_profile_coverage']:.1%}), "
        f"reviews={coverage_report['users_with_reviews']}/{coverage_report['target_user_count']} "
        f"({coverage_report['users_with_reviews_ratio']:.1%}), "
        f"cold_start={coverage_report['cold_start_candidates']}"
    )
    print(
        "Hybrid plan: "
        f"heuristic_direct={heuristic_direct_count}, "
        f"llm_refine={llm_refine_count}, "
        f"stale_refresh={stale_count}"
    )

    if args.coverage_report:
        with open(args.coverage_report, "w", encoding="utf-8") as file:
            json.dump(coverage_report, file, indent=4, ensure_ascii=False)
        print(f"Coverage report saved to: {args.coverage_report}")

    save_profiles(args.output, profiles)

    if args.report_only:
        print("Report-only mode enabled. No API calls made.")
        return

    print(f"Remaining users to process via LLM refinement: {len(users_to_process)}")

    if users_to_process:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_user = {
                executor.submit(
                    extract_profile_with_retry,
                    plan,
                    args.max_retries,
                    args.min_request_interval,
                    args.rate_limit_base_sleep,
                    args.rate_limit_max_sleep,
                ): plan["user_id"]
                for plan in users_to_process
            }

            for processed_count, future in enumerate(
                tqdm(as_completed(future_to_user), total=len(users_to_process), desc="Refining Profiles"),
                start=1,
            ):
                user_id = future_to_user[future]
                try:
                    _, profile = future.result()
                    profiles[user_id] = profile
                except Exception as error:
                    print(f"\nUnhandled exception for user {user_id}: {error}")
                    fallback_plan = next(plan for plan in users_to_process if plan["user_id"] == user_id)
                    fallback = finalize_profile(
                        fallback_plan["heuristic_profile"],
                        fallback_plan["signature"],
                        fallback_plan["review_count_used"],
                    )
                    fallback["source"] = "heuristic_fallback"
                    fallback["reasoning"] = "[HEURISTIC FALLBACK] " + fallback_plan["heuristic_profile"]["reasoning"].replace("[HEURISTIC] ", "")
                    profiles[user_id] = fallback

                if processed_count % args.save_every == 0:
                    save_profiles(args.output, profiles)

    save_profiles(args.output, profiles)
    print(f"\nSuccessfully generated profiles for {len(profiles)} users.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
