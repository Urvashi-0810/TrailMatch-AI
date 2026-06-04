import uuid
import json
import re
import httpx
from typing import Dict, Any, List, Optional
from loguru import logger
from utils.gemini_client import generate_json

# ClinicalTrials.gov API v2 base URL
CTGOV_API_BASE = "https://clinicaltrials.gov/api/v2"

FALLBACK_TRIALS_SYSTEM_PROMPT = """You are a clinical trial data specialist. When the ClinicalTrials.gov
API is unavailable, generate realistic mock clinical trial data for testing purposes.

You will receive a patient condition. Return a JSON array of 2-3 realistic trial objects:
[
  {
    "trial_id": "<NCTxxxxxxxx>",
    "trial_name": "<realistic trial name for the condition>",
    "description": "<realistic trial description>",
    "phase": "<Phase 2 or Phase 3>",
    "status": "recruiting",
    "included_conditions": ["<condition>"],
    "excluded_conditions": [],
    "age_min": <integer>,
    "age_max": <integer>,
    "gender": ["M", "F"],
    "location": "<realistic location>",
    "drug_name": "<realistic drug name>",
    "drug_class": "<drug class>",
    "side_effects": ["<side effect>", ...],
    "excluded_medications": [],
    "excluded_allergies": [],
    "enrollment_target": <integer>,
    "duration_months": <integer>,
    "sponsor": "<sponsor name>",
    "contact_email": null,
    "bmi_min": null,
    "bmi_max": null,
    "required_lab_tests": {},
    "source": "Mock Data"
  }
]

Rules:
- Make the trials medically realistic for the given condition.
- Use varied phases, locations, and drug names.
- Generate unique NCT-style IDs.
- Return ONLY a valid JSON array."""


class WebScrapingAgent:
    """
    Agent responsible for discovering clinical trials from ClinicalTrials.gov
    based on patient conditions. Falls back to Gemini LLM mock generation
    if the API is unreachable.
    """

    def __init__(self):
        self.logger = logger
        self.agent_id = "web_scraping_agent"
        self.role = "Clinical Trial Discovery Specialist"
        self.api_base = CTGOV_API_BASE
        self.max_results = 10
        self.timeout = 15  # seconds

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_age_limit(eligibility: dict, key: str) -> Optional[int]:
        """Pull min/max age from the eligibility block and convert to int years."""
        raw = eligibility.get(key)
        if not raw:
            return None
        match = re.search(r"(\d+)", str(raw))
        return int(match.group(1)) if match else None

    @staticmethod
    def _map_gender(sex: Optional[str]) -> List[str]:
        if not sex or sex.upper() == "ALL":
            return ["M", "F", "Other"]
        if sex.upper() == "MALE":
            return ["M"]
        if sex.upper() == "FEMALE":
            return ["F"]
        return ["M", "F", "Other"]

    @staticmethod
    def _map_phase(phases: Optional[list]) -> str:
        if not phases:
            return "Not specified"
        return ", ".join(p.replace("PHASE", "Phase ").strip() for p in phases)

    @staticmethod
    def _map_status(raw: Optional[str]) -> str:
        mapping = {
            "RECRUITING": "recruiting",
            "ACTIVE_NOT_RECRUITING": "active",
            "COMPLETED": "completed",
            "NOT_YET_RECRUITING": "not yet recruiting",
            "ENROLLING_BY_INVITATION": "enrolling by invitation",
        }
        return mapping.get((raw or "").upper(), (raw or "unknown").lower())

    def _api_study_to_trial(self, study: dict) -> Dict[str, Any]:
        """Convert a ClinicalTrials.gov v2 study object to the pipeline trial dict."""
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        eligibility = proto.get("eligibilityModule", {})
        desc_mod = proto.get("descriptionModule", {})
        contacts_mod = proto.get("contactsLocationsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        conditions_mod = proto.get("conditionsModule", {})

        # -- locations --
        locations = contacts_mod.get("locations", [])
        location_str = (
            f"{locations[0].get('city', '')}, {locations[0].get('state', '')}"
            if locations
            else "Not specified"
        )

        # -- contact email --
        central_contacts = contacts_mod.get("centralContacts", [])
        email = central_contacts[0].get("email") if central_contacts else None

        # -- drug / intervention info --
        interventions = arms_mod.get("interventions", [])
        drug_name = interventions[0].get("name", "Investigational Agent") if interventions else "Not specified"
        drug_class = interventions[0].get("type", "Other") if interventions else "Not specified"

        # -- sponsor --
        lead = sponsor_mod.get("leadSponsor", {})
        sponsor = lead.get("name", "Not specified")

        # -- eligibility criteria text (for downstream trial_parser_agent) --
        criteria_text = eligibility.get("eligibilityCriteria", "")

        # -- enrollment --
        enrollment_info = design.get("enrollmentInfo", {})
        enrollment_target = enrollment_info.get("count")

        return {
            "trial_id": ident.get("nctId", f"NCT-{uuid.uuid4().hex[:8]}"),
            "trial_name": ident.get("officialTitle") or ident.get("briefTitle", "Unnamed Trial"),
            "description": desc_mod.get("briefSummary", ""),
            "phase": self._map_phase(design.get("phases")),
            "status": self._map_status(status_mod.get("overallStatus")),
            "included_conditions": conditions_mod.get("conditions", []),
            "excluded_conditions": [],
            "age_min": self._extract_age_limit(eligibility, "minimumAge"),
            "age_max": self._extract_age_limit(eligibility, "maximumAge"),
            "gender": self._map_gender(eligibility.get("sex")),
            "location": location_str,
            "drug_name": drug_name,
            "drug_class": drug_class,
            "side_effects": [],
            "excluded_medications": [],
            "excluded_allergies": [],
            "bmi_min": None,
            "bmi_max": None,
            "required_lab_tests": {},
            "enrollment_target": int(enrollment_target) if enrollment_target else None,
            "duration_months": None,
            "sponsor": sponsor,
            "contact_email": email,
            "eligibility_criteria_text": criteria_text,
            "source": "ClinicalTrials.gov",
        }

    # ── main entry point ─────────────────────────────────────────────────

    async def scrape_clinical_trials(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("[WebScrapingAgent] Searching clinical trials from ClinicalTrials.gov")

        conditions = patient_data.get("conditions", [])
        if not conditions:
            return {
                "trials": [],
                "total_found": 0,
                "message": "No conditions provided",
            }

        primary_condition = conditions[0]
        self.logger.info(
            f"[WebScrapingAgent] Querying ClinicalTrials.gov for: {primary_condition}"
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.api_base}/studies",
                    params={
                        "query.cond": primary_condition,
                        "filter.overallStatus": "RECRUITING",
                        "pageSize": self.max_results,
                        "format": "json",
                    },
                )
                resp.raise_for_status()

            data = resp.json()
            studies = data.get("studies", [])

            trials = [self._api_study_to_trial(s) for s in studies]
            self.logger.info(f"[WebScrapingAgent] Found {len(trials)} trials from API")

            if not trials:
                self.logger.info("[WebScrapingAgent] API returned 0 results, using fallback")
                return self._create_fallback_trials(patient_data)

            return {
                "trials": trials,
                "total_found": len(trials),
                "search_condition": primary_condition,
                "source": "ClinicalTrials.gov",
            }

        except Exception as e:
            self.logger.warning(
                f"[WebScrapingAgent] API request failed, using fallback trials: {str(e)}"
            )
            return self._create_fallback_trials(patient_data)

    # ── fallback ─────────────────────────────────────────────────────────

    def _create_fallback_trials(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use Gemini LLM to generate realistic fallback trials."""
        conditions = patient_data.get("conditions", [])
        primary_condition = conditions[0] if conditions else "general condition"

        try:
            prompt = f"Generate realistic mock clinical trials for a patient with: {primary_condition}"
            mock_trials = generate_json(FALLBACK_TRIALS_SYSTEM_PROMPT, prompt)
            if not isinstance(mock_trials, list):
                mock_trials = [mock_trials]
        except Exception as e:
            self.logger.warning(f"[WebScrapingAgent] Fallback LLM failed: {str(e)}")
            mock_trials = [
                {
                    "trial_id": f"MOCK-{uuid.uuid4().hex[:6]}",
                    "trial_name": f"{primary_condition.title()} Treatment Study",
                    "description": f"Clinical trial studying {primary_condition}",
                    "phase": "Phase 3",
                    "status": "recruiting",
                    "included_conditions": [primary_condition],
                    "excluded_conditions": [],
                    "age_min": 18,
                    "age_max": 75,
                    "gender": ["M", "F"],
                    "location": "Multiple locations",
                    "drug_name": "Investigational Drug",
                    "drug_class": "Therapeutic",
                    "side_effects": ["Nausea", "Fatigue"],
                    "excluded_medications": [],
                    "excluded_allergies": [],
                    "bmi_min": None,
                    "bmi_max": None,
                    "required_lab_tests": {},
                    "enrollment_target": 200,
                    "duration_months": 24,
                    "sponsor": "Research Institute",
                    "contact_email": None,
                    "source": "Mock Data",
                }
            ]

        return {
            "trials": mock_trials,
            "total_found": len(mock_trials),
            "search_condition": primary_condition,
            "fallback": True,
        }

    def get_info(self) -> Dict[str, str]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "data_source": "ClinicalTrials.gov API",
            "method": "REST API (v2) + LLM fallback",
            "async_supported": True,
        }


__all__ = ["WebScrapingAgent"]