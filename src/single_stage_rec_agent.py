import os
import re
import json
import time
import logging
import math
import tiktoken
from typing import List, Dict

from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase
from src.personal_rec_router import DynamicIntentRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def num_tokens_from_string(string: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))
    except Exception:
        return len(string) // 4


def sanitize_text(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.encode("utf-8", "replace").decode("utf-8")


def sanitize_jsonable(value):
    if isinstance(value, dict):
        return {sanitize_text(k): sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, str):
        return sanitize_text(value)
    return value

class SingleStageRecAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase, domain=None):
        """
        Initialize agent with optional domain for domain-specific skills.

        Args:
            llm: LLM client instance
            domain: Domain name (amazon, yelp, goodreads) for domain-specific skills
        """
        super().__init__(llm=llm)
        self.domain = domain
        self.router = DynamicIntentRouter(domain=domain)
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "execution_logs.jsonl")

    def _fetch_candidates_data(self, candidate_list: List[str], data_strategy: List[str], max_tokens: int = 40000) -> str:
        candidates_str = "Candidate Items:\n"
        for item_id in candidate_list:
            item_data = self.interaction_tool.get_item(item_id)
            if not item_data: continue
            
            filtered_data = {}
            for k, v in item_data.items():
                if k in data_strategy or k == "item_id":
                    filtered_data[k] = sanitize_jsonable(v)
            filtered_data['item_id'] = item_id
            
            entry = json.dumps(filtered_data, ensure_ascii=False, allow_nan=False, default=str) + "\n"
            if num_tokens_from_string(candidates_str + entry) > max_tokens: 
                logger.warning("Max tokens reached in candidate fetching. Truncating candidates list.")
                break
            candidates_str += entry
            
        return candidates_str

    def _call_llm_and_parse(self, system_prompt, user_prompt, candidate_list, expected_count):
        system_prompt = sanitize_text(system_prompt)
        user_prompt = sanitize_text(user_prompt)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        max_retries = 15
        response_text = None
        if not os.environ.get("OPENAI_API_KEY"): 
            return candidate_list[:expected_count], "NO_API_KEY"
        
        prompt_tokens = num_tokens_from_string(system_prompt + user_prompt)
        # Dynamic delay based on TPM (Assuming ~100k TPM limit for Tier 1)
        dynamic_delay = max(1.0, (prompt_tokens / 100000.0) * 60.0)
        
        for attempt in range(max_retries):
            try:
                response_text = self.llm(messages=messages, temperature=0.1, max_tokens=1000)
                time.sleep(dynamic_delay) 
                break
            except Exception as e:
                error_text = str(e)
                if "429" in error_text:
                    sleep_time = (attempt + 1) * 15 + (time.time() % 10)
                    logger.warning(f"Rate limit 429 hit. Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                elif "parse the JSON body of your request" in error_text:
                    sleep_time = min(5 * (attempt + 1), 30)
                    logger.warning(
                        "OpenAI request body parse failure. Retrying in %.2fs... (Attempt %s/%s, prompt_chars=%s)",
                        sleep_time,
                        attempt + 1,
                        max_retries,
                        len(system_prompt) + len(user_prompt),
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        "LLM Call failed: %s (prompt_chars=%s, prompt_tokens_est=%s)",
                        e,
                        len(system_prompt) + len(user_prompt),
                        prompt_tokens,
                    )
                    time.sleep(5)
                    break
                    
        if not response_text: 
            return candidate_list[:expected_count], "FAILED_CALL"

        found_ids = re.findall(r'([A-Za-z0-9_\-]+)', response_text)
        
        result = []
        seen = set()
        for id_str in found_ids:
            if id_str in candidate_list and id_str not in seen:
                result.append(id_str)
                seen.add(id_str)
                
        # Fill missing with original order
        for cand in candidate_list:
            if cand not in seen:
                result.append(cand)

        return result[:expected_count], response_text

    def _build_cognitive_context(self, profile: dict) -> str:
        """Build cognitive context string for LLM prompt (V0B/V1B variant)."""
        analytical = profile.get("analytical_score", 0.5)
        exploratory = profile.get("exploratory_score", 0.5)
        social = profile.get("social_score", 0.5)

        return f"""Cognitive Indicators (for reference only):
- Analytical tendency: {analytical:.2f} (focus on specs, comparisons, value)
- Exploratory tendency: {exploratory:.2f} (focus on novelty, aesthetics, experiences)
- Social tendency: {social:.2f} (focus on popularity, trends, consensus)

Note: These are behavioral indicators extracted from historical reviews."""

    def workflow(self) -> List[str]:
        user_id = self.task['user_id']
        candidate_list = self.task['candidate_list']
        domain_context = self.task.get('candidate_category', "")

        # Normalize domain from candidate_category (e.g., "amazon" from "amazon_product")
        normalized_domain = None
        for valid_domain in ["amazon", "yelp", "goodreads"]:
            if valid_domain in domain_context.lower():
                normalized_domain = valid_domain
                break

        # Reload router skills if domain changed
        if normalized_domain and normalized_domain != self.domain:
            self.domain = normalized_domain
            self.router.reload_skills(domain=normalized_domain)

        # Check experimental flags
        enable_evolution = os.environ.get("ENABLE_EVOLUTION", "false").lower() == "true"
        use_cognitive_guidance = os.environ.get("USE_COGNITIVE_GUIDANCE", "false").lower() == "true"

        skill_name, skill_data = self.router.route(user_id, domain_context, enable_evolution=enable_evolution)

        profile = self.router.profiles.get(user_id, {})
        user_profile_reasoning = profile.get('reasoning', '')

        rich_data = self._fetch_candidates_data(candidate_list, skill_data['data_strategy'], max_tokens=40000)

        sys_prompt = skill_data['prompt'] + "\nYou must rank the following 20 candidate items. Provide your final ranking by listing the item_ids in order of preference from best to worst."

        # Conditionally inject cognitive context based on experimental variant
        if use_cognitive_guidance and profile:
            cognitive_context = self._build_cognitive_context(profile)
            user_prompt = f"""User Profile Context: {user_profile_reasoning}

{cognitive_context}

Target Domain: {domain_context}

{rich_data}

Please output the ranked item IDs."""
        else:
            # V0A/V1A: No cognitive injection (Gradual Guidance track)
            user_prompt = f"""User Profile Context: {user_profile_reasoning}

Target Domain: {domain_context}

{rich_data}

Please output the ranked item IDs."""
        
        final_ranked_ids, raw_resp = self._call_llm_and_parse(sys_prompt, user_prompt, candidate_list, 20)

        # Logging
        log_entry = {
            "user_id": user_id,
            "domain": domain_context,
            "used_skill": skill_name,
            "predicted_top_1": final_ranked_ids[0] if final_ranked_ids else None,
            "predicted_top_5": final_ranked_ids[:5],
            "raw_response": raw_resp
        }
        import fcntl
        with open(self.log_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(sanitize_jsonable(log_entry), ensure_ascii=False, allow_nan=False, default=str) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
            
        return final_ranked_ids
