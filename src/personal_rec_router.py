import json
import os

class DynamicIntentRouter:
    """
    Dynamic Skill Router for PersonalRecAgent.

    Design:
    - V0 (enable_evolution=False): Always use Skill_Generic_Universal (single seed)
    - V1 (enable_evolution=True): Match evolved skills based on user cognitive profile

    Cognitive Dimensions:
    - analytical: Focus on specs, comparisons, value analysis
    - exploratory: Focus on novelty, aesthetics, experiences
    - social: Focus on popularity, trends, community consensus

    Domain-Specific Skills:
    - Each domain has its own skills file: skills_db_{domain}.json
    - The universal seed is always included in each domain's skills file
    """

    # Cognitive type thresholds
    HIGH_SCORE_THRESHOLD = 0.6

    # Valid domains
    VALID_DOMAINS = ["amazon", "yelp", "goodreads"]

    def __init__(self, domain=None, skills_file=None, profiles_file="data/user_profiles.json"):
        """
        Initialize router with optional domain-specific skills.

        Args:
            domain: Domain name (amazon, yelp, goodreads). If provided, loads domain-specific skills.
            skills_file: Explicit skills file path. Overrides domain-based loading if provided.
            profiles_file: Path to user profiles JSON.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Determine skills file path
        if skills_file:
            self.skills_path = os.path.join(base_dir, skills_file)
        elif domain and domain in self.VALID_DOMAINS:
            self.skills_path = os.path.join(base_dir, f"src/skills_db_{domain}.json")
        else:
            self.skills_path = os.path.join(base_dir, "src/skills_db.json")

        self.profiles_path = os.path.join(base_dir, profiles_file)
        self.domain = domain

        self.skills = self._load_json(self.skills_path)
        self.profiles = self._load_json(self.profiles_path)

    def _load_json(self, path):
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}. Returning empty dict.")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def reload_skills(self, domain=None):
        """
        Reload skills from file. If domain is provided and differs from current,
        switches to domain-specific skills file.

        Args:
            domain: Optional new domain to switch to
        """
        if domain and domain != self.domain and domain in self.VALID_DOMAINS:
            self.domain = domain
            self.skills_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                f"src/skills_db_{domain}.json"
            )
        self.skills = self._load_json(self.skills_path)

    def route(self, user_id, domain_context="", enable_evolution=False):
        """
        Unified routing method.

        V0 (enable_evolution=False): Use Skill_Generic_Universal only
        V1 (enable_evolution=True): Match evolved skills by cognitive profile
        """
        if not enable_evolution:
            # V0: Always use Universal seed
            return self._get_universal_seed()

        # V1: Match evolved skill based on user's cognitive profile
        evolved_name, evolved_skill = self._match_by_cognitive_profile(user_id)
        if evolved_skill:
            return evolved_name, evolved_skill

        # V1 Fallback: Use Universal seed if no evolved skill matched
        return self._get_universal_seed()

    def _get_universal_seed(self):
        """Get the single universal seed skill (V0 design)."""
        if "Skill_Generic_Universal" in self.skills:
            return "Skill_Generic_Universal", self.skills["Skill_Generic_Universal"]

        # Absolute fallback: First skill in database
        if self.skills:
            first_skill = list(self.skills.items())[0]
            return first_skill[0], first_skill[1]

        raise ValueError("No skills available in skills_db.json")

    def _get_user_cognitive_type(self, user_id):
        """
        Determine user's dominant cognitive type from profile.

        Returns: (dominant_type, scores_dict)
        - dominant_type: "analytical" | "exploratory" | "social" | "balanced" | "cold_start"
        - scores_dict: {analytical, exploratory, social}
        """
        profile = self.profiles.get(user_id, {})

        # Check for cold start
        if not profile or "[COLD START]" in profile.get("reasoning", ""):
            return "cold_start", {"analytical": 0.5, "exploratory": 0.5, "social": 0.5}

        scores = {
            "analytical": profile.get("analytical_score", 0.5),
            "exploratory": profile.get("exploratory_score", 0.5),
            "social": profile.get("social_score", 0.5),
        }

        # Find dominant type
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        if max_score >= self.HIGH_SCORE_THRESHOLD:
            return max_type, scores
        else:
            return "balanced", scores

    def _match_by_cognitive_profile(self, user_id):
        """
        Match user's cognitive profile to the best evolved skill.

        Evolved skills should have a 'cognitive_type' field indicating their target:
        - "analytical": For users with high analytical_score
        - "exploratory": For users with high exploratory_score
        - "social": For users with high social_score
        - "balanced": For users with no dominant type

        Returns (skill_name, skill_data) or (None, None) if no match.
        """
        user_type, scores = self._get_user_cognitive_type(user_id)

        # Find best matching evolved skill for this cognitive type
        best_skill = None
        best_skill_data = None
        best_score = 0

        for skill_name, skill_data in self.skills.items():
            # Skip the universal seed
            if skill_name == "Skill_Generic_Universal":
                continue

            skill_cognitive_type = skill_data.get("cognitive_type", "").lower()

            # Exact match with user's dominant type
            if skill_cognitive_type == user_type:
                type_match_score = 1.0
            # Partial match based on skill's target cognitive dimension
            elif skill_cognitive_type in scores:
                type_match_score = scores[skill_cognitive_type]
            else:
                type_match_score = 0.0

            if type_match_score > best_score:
                best_score = type_match_score
                best_skill = skill_name
                best_skill_data = skill_data

        # Only return if we have a reasonable match
        if best_skill and best_score >= 0.5:
            return best_skill, best_skill_data

        return None, None

