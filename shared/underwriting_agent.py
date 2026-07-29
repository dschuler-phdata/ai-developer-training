import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field


class SubmissionExtraction(BaseModel):
    """Extraction schema for underwriting submissions."""

    business_name: str
    industry: str
    locations: list[str]
    requested_limits: str | None = None
    notable_risks: list[str]
    missing_information: list[str]


EXTRACTION_SYSTEM_PROMPT = (
    "Extract underwriting information from the submission text. "
    "For the 'requested_limits' field, if no specific amount is mentioned, set it to null. "
    "For 'notable_risks' and 'missing_information', use empty lists if there are none. "
    "Do not invent values for missing fields."
)


def parse_dollar_amount(text: str | None) -> float | None:
    """Pull the first dollar figure out of a free-text limit string, e.g. '$3,000,000' -> 3000000.0."""
    if not text:
        return None
    match = re.search(r"\$?([\d,]+)", text)
    return float(match.group(1).replace(",", "")) if match else None


class RuleCheckInput(BaseModel):
    industry: str = Field(description="The business's primary industry")
    requested_limit: float | None = Field(default=None, description="Requested coverage limit in dollars")
    flood_zone: str | None = Field(default=None, description="FEMA flood zone letter, e.g. 'A' or 'V', if applicable")
    flood_mitigation_on_file: bool = Field(default=False, description="Whether flood mitigation documentation is on file")
    roof_age_years: int | None = Field(default=None, description="Age of the roof in years, if known")
    heavy_machinery: bool = Field(default=False, description="Whether the business operates heavy machinery (forklifts, industrial saws, commercial mowing equipment)")
    prior_liability_claims_5yr: int = Field(default=0, description="Number of liability claims in the past 5 years")


def check_underwriting_rules(
    industry: str,
    requested_limit: float | None = None,
    flood_zone: str | None = None,
    flood_mitigation_on_file: bool = False,
    roof_age_years: int | None = None,
    heavy_machinery: bool = False,
    prior_liability_claims_5yr: int = 0,
) -> dict:
    """Check known facts about a risk against explicit underwriting thresholds. No LLM call."""
    flags = []

    if requested_limit is not None and requested_limit > 5_000_000:
        flags.append("Requested limit exceeds standard $5,000,000 max - requires senior underwriter sign-off")

    if (
        flood_zone in ("A", "V")
        and requested_limit is not None
        and requested_limit > 2_000_000
        and not flood_mitigation_on_file
    ):
        flags.append(
            "Flood zone A/V with requested limit over $2,000,000 and no flood mitigation "
            "documentation on file - outside standard property appetite"
        )

    if heavy_machinery:
        flags.append("Heavy machinery present - supplemental safety questionnaire required before binding")

    if roof_age_years is not None and roof_age_years > 20:
        flags.append("Roof older than 20 years - current roof condition report required before binding")

    if prior_liability_claims_5yr >= 2:
        flags.append("2+ liability claims in past 5 years - outside preferred risk criteria")

    return {"flags": flags, "within_appetite": len(flags) == 0}


def check_underwriting_rules_v2(
    industry: str,
    requested_limit: float | None = None,
    flood_zone: str | None = None,
    flood_mitigation_on_file: bool = False,
    roof_age_years: int | None = None,
    heavy_machinery: bool = False,
    prior_liability_claims_5yr: int = 0,
) -> dict:
    """Same rules as check_underwriting_rules, but normalizes FEMA flood zone subzone
    codes (e.g. 'VE', 'AE', 'A99') to their base letter before matching, so the flood/
    limit rule fires correctly regardless of the exact subzone passed in.
    """
    normalized_flood_zone = flood_zone[:1].upper() if flood_zone else None
    return check_underwriting_rules(
        industry=industry,
        requested_limit=requested_limit,
        flood_zone=normalized_flood_zone,
        flood_mitigation_on_file=flood_mitigation_on_file,
        roof_age_years=roof_age_years,
        heavy_machinery=heavy_machinery,
        prior_liability_claims_5yr=prior_liability_claims_5yr,
    )


class SearchDocumentsInput(BaseModel):
    query: str = Field(description="Natural-language question to search the underwriting guideline manuals for")
    k: int = Field(default=3, description="Number of matching excerpts to return")


class ExtractSubmissionInput(BaseModel):
    submission_text: str = Field(description="Full raw text of the underwriting submission to extract structured facts from")


TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search the underwriting guideline manuals (general liability, property "
            "appetite, cyber exclusions, workers' comp classification) for text relevant "
            "to a natural-language question. Returns the top-k most relevant excerpts "
            "with their source document."
        ),
        "input_schema": SearchDocumentsInput.model_json_schema(),
    },
    {
        "name": "extract_submission_information",
        "description": (
            "Extract structured underwriting facts (business name, industry, locations, "
            "requested limits, notable risks, missing information) from the raw text of "
            "a new business submission."
        ),
        "input_schema": ExtractSubmissionInput.model_json_schema(),
    },
    {
        "name": "check_underwriting_rules",
        "description": (
            "Deterministically check known facts about a risk (requested limit, flood "
            "zone, roof age, heavy machinery, prior claims) against the underwriting "
            "manuals' explicit thresholds. Does not use an LLM - returns any rule "
            "violations found."
        ),
        "input_schema": RuleCheckInput.model_json_schema(),
    },
]

AGENT_SYSTEM_PROMPT = (
    "You are an underwriting assistant. You have three tools available: "
    "search_documents (search the underwriting guideline manuals), "
    "extract_submission_information (pull structured facts out of raw submission text), "
    "and check_underwriting_rules (check known facts against explicit underwriting thresholds). "
    "Given a new business submission, decide which tools you need and in what order. "
    "Gather enough information to determine whether the risk is within appetite before answering."
)


class UnderwritingDecision(BaseModel):
    """Final structured underwriting appetite decision."""

    decision: str = Field(description="One of: 'accept', 'decline', or 'refer to senior underwriter'")
    confidence: float = Field(description="Confidence in this decision, from 0.0 to 1.0")
    risk_factors: list[str] = Field(description="Specific risk factors identified from the gathered tool outputs")
    missing_information: list[str] = Field(description="Information that would be needed to be more certain")
    evidence: list[str] = Field(description="Specific tool outputs (guideline excerpts, rule flags, extracted facts) that support this decision")
    rationale: str = Field(description="Brief explanation tying the decision to the evidence above")


@dataclass
class UnderwritingAgent:
    """Underwriting tools and agent loop bound to a live LLM client and Chroma collection.

    `search_documents`/`extract_submission_information` need a live embedding client and
    a populated Chroma `collection` to query, so they can't be defined at import time -
    `build_underwriting_agent()` closes over the caller's own `client`/`collection` and
    returns everything bound and ready to call.
    """

    search_documents: callable
    extract_submission_information: callable
    run_agent_loop: callable
    run_full_agent: callable
    tool_functions: dict


def build_underwriting_agent(client, collection) -> UnderwritingAgent:
    """Build the underwriting tools and agent loop against a given LLM client and Chroma collection."""

    def retrieve(query: str, k: int = 3):
        query_embedding = client.embed([query])[0]
        return collection.query(query_embeddings=[query_embedding], n_results=k)

    def search_documents(query: str, k: int = 3) -> list[dict]:
        """Search the underwriting guideline manuals for text relevant to `query`."""
        results = retrieve(query, k=k)

        hits = []
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append(
                {
                    "source": metadata["source"],
                    "chunk_index": metadata["chunk_index"],
                    "text": text,
                    "distance": distance,
                }
            )

        return hits

    def extract_submission_information(submission_text: str) -> dict:
        """Pull structured facts out of raw submission text."""
        result = client.generate_structured(
            submission_text, SubmissionExtraction, system_prompt=EXTRACTION_SYSTEM_PROMPT
        )
        return result.model_dump()

    tool_functions = {
        "search_documents": search_documents,
        "extract_submission_information": extract_submission_information,
        "check_underwriting_rules": check_underwriting_rules,
    }

    def run_agent_loop(user_message: str, system_prompt: str, tools: list[dict] = TOOLS, max_iterations: int = 6):
        # Force search_documents on the very first turn instead of leaving it to the model's
        # judgment - a reasoning model has no temperature/seed dial to make that per-run
        # judgment call land the same way twice, so a prompt asking it to search "when needed"
        # was calling search_documents inconsistently. tool_choice is an API-enforced
        # constraint, not a suggestion, so this makes retrieval happen on every run. Every
        # turn after the first goes back to "auto" so the model stays free to call
        # check_underwriting_rules or finalize once retrieval has happened.
        response = client.generate_with_tools(
            user_message, tools, system_prompt=system_prompt, tool_choice="search_documents"
        )
        tool_call_log = []
        iterations = 0

        while response.tool_calls:
            if iterations >= max_iterations:
                break

            messages = response.messages
            for call in response.tool_calls:
                result = tool_functions[call.name](**call.arguments)
                tool_call_log.append({"name": call.name, "arguments": call.arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

            response = client.generate_with_tools(
                user_message, tools, system_prompt=system_prompt, messages=messages, tool_choice="auto"
            )
            iterations += 1

        return response, tool_call_log

    def run_full_agent(submission_text: str) -> tuple[UnderwritingDecision, list[dict]]:
        user_message = (
            f"Process this underwriting submission and determine whether it is within appetite:\n\n{submission_text}"
        )
        response, tool_call_log = run_agent_loop(user_message, AGENT_SYSTEM_PROMPT)

        tool_output_summary = "\n\n".join(
            f"Tool: {entry['name']}({entry['arguments']})\nResult: {entry['result']}"
            for entry in tool_call_log
        )
        decision_prompt = (
            f"Submission:\n{submission_text}\n\n"
            f"Tool outputs gathered while investigating this submission:\n{tool_output_summary}\n\n"
            "Using ONLY the information above, produce a final underwriting appetite decision. "
            "Every risk_factor and evidence entry must trace back to a specific tool output above - "
            "do not introduce new facts."
        )
        decision = client.generate_structured(decision_prompt, UnderwritingDecision, system_prompt=AGENT_SYSTEM_PROMPT)
        return decision, tool_call_log

    return UnderwritingAgent(
        search_documents=search_documents,
        extract_submission_information=extract_submission_information,
        run_agent_loop=run_agent_loop,
        run_full_agent=run_full_agent,
        tool_functions=tool_functions,
    )
