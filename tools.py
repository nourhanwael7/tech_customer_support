"""Marketing tool architecture for the Marketing Campaign Strategist AI Agent."""

import re
from collections import Counter
from datetime import datetime
from typing import Callable


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+-]*", text.lower())


def CampaignIdeaGenerator(brief: str) -> dict:
    """Generate a structured campaign idea scaffold from the user's brief."""
    channels = []
    lowered = brief.lower()
    for name, kws in {"Email": ["email", "newsletter"], "SEO": ["seo", "search"], "Social Media": ["social", "tiktok", "instagram", "gen z"], "Paid Search": ["ppc", "paid", "google"], "Content Marketing": ["blog", "content"]}.items():
        if any(k in lowered for k in kws):
            channels.append(name)
    if not channels:
        channels = ["Email", "Social Media", "SEO"]
    return {"success": True, "campaign_goal": "Drive awareness and qualified conversions", "target_audience": brief, "recommended_channels": channels, "content_angles": ["problem-solution messaging", "social proof", "clear CTA"], "kpis": ["CTR", "Conversion Rate", "CAC", "ROAS"], "guardrail": "No performance statistics are estimated without retrieved evidence."}


def CustomerSegmentAnalyzer(segment_description: str) -> dict:
    """Analyze a customer segment with needs, motivations, and channel fit."""
    words = Counter(_tokens(segment_description))
    return {"success": True, "segment_signals": [w for w, _ in words.most_common(8)], "likely_needs": ["relevance", "trust", "clear value proposition"], "positioning": "Match message and offer to persona motivations and funnel stage.", "recommended_channels": ["Email nurturing", "Social Media", "Search-led content"], "risks": ["Avoid stereotypes", "Validate assumptions with first-party data"]}


def MarketingReportSummarizer(report_text: str) -> dict:
    """Create an extractive summary scaffold for marketing report text."""
    sentences = re.split(r"(?<=[.!?])\s+", report_text.strip())
    return {"success": True, "summary_points": sentences[:4] if sentences else [report_text[:240]], "observed_metrics": [m.upper() for m in ["ctr", "cac", "roas", "conversion rate", "open rate"] if m in report_text.lower()], "next_steps": ["Compare funnel-stage KPIs", "Identify weakest conversion step", "Prioritize tests with measurable outcomes"]}


def CampaignKPIExplainer(metric_question: str) -> dict:
    """Explain common marketing KPIs without inventing benchmarks."""
    definitions = {"CAC": "Customer Acquisition Cost: spend required to acquire a customer.", "CTR": "Click-Through Rate: percentage of impressions or sends that produce clicks.", "ROAS": "Return on Ad Spend: revenue attributed to advertising divided by ad spend.", "Conversion Rate": "Percentage of visitors or leads completing the desired action.", "Open Rate": "Percentage of delivered emails opened."}
    requested = [k for k in definitions if k.lower() in metric_question.lower()]
    if not requested:
        requested = list(definitions)
    return {"success": True, "definitions": {k: definitions[k] for k in requested}, "interpretation_guardrail": "Use trends and campaign goals; do not claim universal benchmark targets without evidence."}


def TrendRecommendationTool(trend_prompt: str) -> dict:
    """Recommend marketing improvements aligned to documented channels and guardrails."""
    return {"success": True, "recommendations": ["Align content with persona and funnel stage", "Use SEO for durable discovery", "Use email segmentation for lifecycle nurturing", "Use social content for awareness and community", "Track CTR, conversion rate, CAC, and ROAS together"], "validation_plan": ["A/B test one variable at a time", "Review retrieved KPI definitions", "Document assumptions before launch"], "guardrail": "Recommendations must be validated against retrieved knowledge and actual campaign data."}


TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    "CampaignIdeaGenerator": CampaignIdeaGenerator,
    "CustomerSegmentAnalyzer": CustomerSegmentAnalyzer,
    "MarketingReportSummarizer": MarketingReportSummarizer,
    "CampaignKPIExplainer": CampaignKPIExplainer,
    "TrendRecommendationTool": TrendRecommendationTool,
}


def execute_tool(tool_name: str | None, params: dict) -> dict:
    if not tool_name:
        return {"success": True, "message": "No tool required."}
    if tool_name not in TOOL_REGISTRY:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    try:
        result = TOOL_REGISTRY[tool_name](**params)
        result["tool_executed_at"] = datetime.utcnow().isoformat() + "Z"
        return result
    except Exception as exc:
        return {"success": False, "error": f"Tool error: {exc}"}
