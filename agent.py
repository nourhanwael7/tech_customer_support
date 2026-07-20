"""Marketing Campaign Strategist AI Agent.

Pipeline: user message -> intent detection -> marketing tool decision -> RAG retrieval ->
Gemini reasoning -> structured, grounded response.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from rag import get_rag_pipeline
from tools import execute_tool

load_dotenv()

APP_NAME = "Marketing Campaign Strategist AI Agent"


def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


@dataclass
class AgentStep:
    step: str
    description: str
    result: Optional[str] = None


@dataclass
class AgentResponse:
    final_answer: str
    intent: str
    tool_used: Optional[str]
    tool_result: Optional[dict]
    rag_context: list[str]
    confidence_score: str = "Medium"
    reasoning_steps: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    error: Optional[str] = None


INTENT_CATEGORIES = {
    "campaign_idea": ["campaign", "idea", "launch", "promotion", "creative", "gen z", "awareness"],
    "segment_analysis": ["segment", "persona", "audience", "customer", "demographic", "behavior"],
    "report_summary": ["summarize", "report", "performance", "results", "analysis", "recap"],
    "kpi_explanation": ["kpi", "cac", "ctr", "roas", "conversion", "metric", "cpa", "open rate"],
    "trend_recommendation": ["trend", "improve", "recommend", "optimize", "seo", "social", "email"],
    "general": [],
}


def detect_intent(user_message: str) -> str:
    msg = user_message.lower()
    scores = {intent: sum(1 for kw in kws if kw in msg) for intent, kws in INTENT_CATEGORIES.items()}
    scores = {k: v for k, v in scores.items() if v > 0}
    return max(scores, key=scores.get) if scores else "general"


def decide_tool(intent: str, user_message: str) -> dict:
    """Reuse the existing deterministic tool-router pattern with marketing tools."""
    if intent == "campaign_idea":
        return {"tool": "CampaignIdeaGenerator", "params": {"brief": user_message}, "reasoning": "Generate a structured campaign concept."}
    if intent == "segment_analysis":
        return {"tool": "CustomerSegmentAnalyzer", "params": {"segment_description": user_message}, "reasoning": "Analyze customer persona and segment fit."}
    if intent == "report_summary":
        return {"tool": "MarketingReportSummarizer", "params": {"report_text": user_message}, "reasoning": "Summarize the marketing report request."}
    if intent == "kpi_explanation":
        return {"tool": "CampaignKPIExplainer", "params": {"metric_question": user_message}, "reasoning": "Explain relevant marketing KPIs."}
    if intent == "trend_recommendation":
        return {"tool": "TrendRecommendationTool", "params": {"trend_prompt": user_message}, "reasoning": "Recommend improvements from documented marketing practices."}
    return {"tool": None, "params": {}, "reasoning": "Answer only from retrieved marketing knowledge."}


FEW_SHOT_EXAMPLES = """
Example 1
User: Suggest a campaign for Gen Z coffee lovers.
Assistant:
Summary: Launch a short-form video and campus ambassador campaign for convenience, identity, and social discovery.
Insights: Gen Z responds to authentic creators, mobile-first content, and community participation.
Recommendations: Use TikTok/Reels, limited-time flavors, referral codes, and UGC challenges.
Supporting Evidence: Retrieved knowledge base sections about social media, personas, and campaign KPIs.
KPIs: CTR, conversion rate, ROAS, CAC, engagement rate.
Confidence Score: Medium — based on retrieved knowledge base.

Example 2
User: Analyze budget-conscious parents as a customer segment.
Assistant:
Summary: This segment prioritizes practical value, reliability, trust, and clear promotions.
Insights: Messaging should reduce perceived risk and emphasize savings.
Recommendations: Use email offers, comparison content, testimonials, and bundle pricing.
Supporting Evidence: Retrieved segmentation and funnel guidance.
Confidence Score: Medium.

Example 3
User: Explain CTR and conversion rate.
Assistant:
Summary: CTR measures ad or email click engagement; conversion rate measures desired action completion.
Insights: High CTR with low conversion may indicate landing-page or offer mismatch.
Recommendations: Align message, landing page, CTA, and audience targeting.
Supporting Evidence: Retrieved campaign metrics definitions.
Confidence Score: High.

Example 4
User: Summarize this report: email opens rose but sales fell.
Assistant:
Summary: Awareness engagement improved, but downstream conversion weakened.
Insights: Subject lines may be strong while offer, landing page, or audience intent is weak.
Recommendations: Review segmentation, CTA clarity, and landing page friction.
Supporting Evidence: Retrieved funnel and email marketing guidance.
Confidence Score: Medium.

Example 5
User: How can we improve SEO traffic?
Assistant:
Summary: Improve search visibility with intent-matched content, technical hygiene, and authority building.
Insights: SEO supports discovery and top-of-funnel demand.
Recommendations: Map keywords to funnel stages, refresh content, optimize metadata, and monitor organic conversion.
Supporting Evidence: Retrieved digital marketing guidance.
Confidence Score: Medium.
"""


SYSTEM_PROMPT = f"""You are {APP_NAME}, a professional marketing strategist.

Strict guardrails:
- Use retrieved context first and ground every answer in it.
- Answer questions only using retrieved knowledge and tool results supplied in the prompt.
- Never fabricate statistics, case studies, benchmarks, sources, or guaranteed outcomes.
- If evidence is missing, clearly say: "I don't have enough evidence in the knowledge base."
- Always include: "This recommendation is based on the retrieved knowledge base."
- Explain concise reasoning without exposing hidden chain-of-thought; use visible reasoning steps.
- Always produce these sections: Summary, Insights, Recommendations, Supporting Evidence, Confidence Score.
- Mention confidence level as High, Medium, or Low with a short justification.

Few-shot formatting examples:
{FEW_SHOT_EXAMPLES}
"""


def _confidence(rag_context: list[str], tool_result: Optional[dict]) -> str:
    if len(rag_context) >= 3 and (not tool_result or tool_result.get("success", True)):
        return "High"
    if rag_context:
        return "Medium"
    return "Low"


def generate_final_response(user_message: str, intent: str, rag_context: list[str], tool_result: Optional[dict], tool_name: Optional[str], conversation_history: list[dict], model) -> str:
    context_block = "\n---\n".join(rag_context[:4]) if rag_context else "NO_RETRIEVED_CONTEXT"
    tool_block = json.dumps(tool_result or {}, indent=2)
    recent_history = "\n".join(f"{m['role']}: {m['content']}" for m in conversation_history[-6:])
    confidence = _confidence(rag_context, tool_result)
    if not rag_context:
        return ("Summary\nI don't have enough evidence in the knowledge base.\n\n"
                "Insights\nNo retrieved marketing context was available for this question.\n\n"
                "Recommendations\nPlease upload or add relevant marketing knowledge before relying on a recommendation. This recommendation is based on the retrieved knowledge base.\n\n"
                "Supporting Evidence\nNo supporting chunks were retrieved.\n\n"
                "Confidence Score\nLow — no retrieved evidence was available.")
    prompt = f"""{SYSTEM_PROMPT}

Conversation history:
{recent_history}

Retrieved knowledge base context:
{context_block}

Tool result from {tool_name or 'no tool'}:
{tool_block}

User message: {user_message}
Intent: {intent}
Estimated confidence before generation: {confidence}

Create the final structured answer now."""
    try:
        text = model.generate_content(prompt).text.strip()
    except Exception as exc:
        text = f"Summary\nI encountered an LLM error: {exc}\n\nInsights\nThe RAG context was retrieved, but response generation failed.\n\nRecommendations\nRetry after checking Gemini configuration. This recommendation is based on the retrieved knowledge base.\n\nSupporting Evidence\n{context_block[:800]}\n\nConfidence Score\nLow — generation failed."
    if "This recommendation is based on the retrieved knowledge base." not in text:
        text += "\n\nThis recommendation is based on the retrieved knowledge base."
    return text


class MarketingStrategistAgent:
    """Orchestrates intent detection, RAG retrieval, marketing tools, and structured response generation."""

    def __init__(self):
        self.model = _configure_gemini()
        self.rag = get_rag_pipeline()
        self.conversation_history: list[dict] = []

    def run(self, user_message: str, show_steps: bool = True) -> AgentResponse:
        steps: list[AgentStep] = []
        intent = detect_intent(user_message)
        steps.append(AgentStep("1. Intent Detection", f"Detected marketing intent: **{intent}**", intent))
        rag_context = self.rag.retrieve(user_message, top_k=4) if self.rag.is_ready else []
        steps.append(AgentStep("2. RAG Retrieval", f"Retrieved {len(rag_context)} top-k chunks from FAISS.", f"{len(rag_context)} chunks"))
        decision = decide_tool(intent, user_message)
        tool_name = decision.get("tool")
        steps.append(AgentStep("3. Tool Decision", f"Tool selected: **{tool_name or 'None'}** — {decision.get('reasoning')}", tool_name or "skipped"))
        tool_result = execute_tool(tool_name, decision.get("params", {})) if tool_name else None
        steps.append(AgentStep("4. Tool Execution", "Executed marketing tool." if tool_name else "No tool required.", json.dumps(tool_result)[:300] if tool_result else "skipped"))
        answer = generate_final_response(user_message, intent, rag_context, tool_result, tool_name, self.conversation_history, self.model)
        confidence = _confidence(rag_context, tool_result)
        reasoning_steps = ["Classified the marketing intent.", "Retrieved relevant FAISS knowledge chunks.", "Applied the selected marketing tool when useful.", "Generated a structured answer with guardrails and confidence."]
        steps.append(AgentStep("5. Structured Response", "Generated grounded marketing response with confidence score.", "done"))
        self.conversation_history += [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}]
        return AgentResponse(answer, intent, tool_name, tool_result, rag_context, confidence, reasoning_steps, steps)

    def clear_history(self):
        self.conversation_history = []

    def add_document(self, text: str, source: str = "uploaded") -> int:
        return self.rag.add_document(text, source)

    def add_pdf_bytes(self, pdf_bytes: bytes, source: str = "uploaded.pdf") -> int:
        return self.rag.add_pdf_bytes(pdf_bytes, source)


# Backward-compatible alias for older imports/tests.
CustomerSupportAgent = MarketingStrategistAgent
