"""
agent.py — Multi-Step AI Customer Support Agent
Pipeline: User Message → Intent Detection → Tool Decision → RAG → Reason → Action → Response
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from rag import get_rag_pipeline
from tools import execute_tool

load_dotenv()

# ── Configure Gemini ───────────────────────────────────────────────────────────

def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


# ── Data Structures ────────────────────────────────────────────────────────────

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
    steps: list[AgentStep] = field(default_factory=list)
    error: Optional[str] = None


# ── Intent Detection ───────────────────────────────────────────────────────────

INTENT_CATEGORIES = {
    "order_status": [
        "order", "track", "tracking", "shipped", "delivery", "delivered",
        "where is", "package", "shipment", "when will", "arrive", "status"
    ],
    "refund_return": [
        "refund", "return", "money back", "cancel", "exchange", "damaged",
        "broken", "defective", "wrong item", "not working"
    ],
    "create_ticket": [
        "problem", "issue", "complaint", "help", "support", "ticket",
        "broken", "not working", "error", "fix", "resolve"
    ],
    "product_info": [
        "price", "cost", "available", "stock", "specs", "features",
        "warranty", "guarantee", "compare", "difference"
    ],
    "account": [
        "account", "password", "login", "sign in", "email", "profile",
        "billing", "payment", "subscription"
    ],
    "shipping": [
        "shipping", "ship", "delivery", "how long", "express", "overnight",
        "free shipping", "international", "address"
    ],
    "generate_report": [
        "report", "analytics", "statistics", "summary", "complaints",
        "generate report", "admin", "dashboard"
    ],
    "send_email": [
        "send email", "email me", "confirmation", "receipt", "notify",
        "send me an email", "email about", "mail me", "send mail",
        "email about my order", "order email", "send email about"
    ],
    "general": []
}


def detect_intent(user_message: str) -> str:
    """Simple keyword-based intent detection."""
    msg_lower = user_message.lower()
    scores: dict[str, int] = {}

    for intent, keywords in INTENT_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in msg_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "general"
    return max(scores, key=lambda k: scores[k])


# ── Regex helpers ──────────────────────────────────────────────────────────────

_ORDER_RE = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

def _extract_order_id(text: str) -> Optional[str]:
    m = _ORDER_RE.search(text)
    return m.group(0).upper() if m else None

def _extract_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None

def _scan_history(history: list, extractor) -> Optional[str]:
    """Scan conversation history newest-first for order ID or email."""
    for msg in reversed(history or []):
        val = extractor(msg.get("content", ""))
        if val:
            return val
    return None


# ── Tool Decision ──────────────────────────────────────────────────────────────

# Keywords that deterministically trigger email sending
_EMAIL_TRIGGERS = [
    "send email", "send me email", "send me an email",
    "email me", "mail me", "send mail",
    "email about", "send an email", "email about my order",
    "order email", "بعت ايميل", "ابعت ايميل",
]

# Keywords that trigger report generation
_REPORT_TRIGGERS = [
    "generate report", "create report", "show report",
    "make report", "pdf report", "support report", "ticket report",
]

# Keywords that trigger order lookup only
_ORDER_TRIGGERS = [
    "track", "order status", "where is my order", "order update",
    "check order", "my order", "shipment",
]


def _is_email_request(message: str) -> bool:
    msg = message.lower()
    return any(t in msg for t in _EMAIL_TRIGGERS)

def _is_report_request(message: str) -> bool:
    msg = message.lower()
    return any(t in msg for t in _REPORT_TRIGGERS)


def decide_tool(intent: str, user_message: str, model, conversation_history: list = None) -> dict:
    """
    DETERMINISTIC rule-based tool router — Gemini is NOT trusted for tool selection.
    Rules are evaluated in priority order; only falls back to Gemini for ambiguous cases.
    """
    history = conversation_history or []
    msg = user_message

    # ── Rule 1: Report request ─────────────────────────────────────────────────
    if _is_report_request(msg) or intent == "generate_report":
        return {
            "tool": "generate_report",
            "params": {},
            "reasoning": "User requested a report.",
            "followup_tool": None,
        }

    # ── Rule 2: Email request → deterministic chain ────────────────────────────
    if _is_email_request(msg) or intent == "send_email":
        # Find order ID from message or history
        order_id = _extract_order_id(msg) or _scan_history(history, _extract_order_id)
        email    = _extract_email(msg)    or _scan_history(history, _extract_email)

        if order_id or email:
            # We have an identifier — fetch order then email
            params = {}
            if order_id:
                params["order_id"] = order_id
            elif email:
                params["email"] = email
            return {
                "tool": "check_order_status",
                "params": params,
                "reasoning": f"Fetch order ({order_id or email}) then send email.",
                "followup_tool": {"tool": "send_email", "use_order_data": True},
            }
        else:
            # No identifier — ask Gemini to extract or fall back to asking user
            return {
                "tool": None,
                "params": {},
                "reasoning": "Email requested but no order ID or email address found.",
                "followup_tool": None,
                "needs_clarification": True,
            }

    # ── Rule 3: Order status lookup ────────────────────────────────────────────
    if intent == "order_status":
        order_id = _extract_order_id(msg) or _scan_history(history, _extract_order_id)
        email    = _extract_email(msg)    or _scan_history(history, _extract_email)
        if order_id or email:
            params = {}
            if order_id:
                params["order_id"] = order_id
            elif email:
                params["email"] = email
            return {
                "tool": "check_order_status",
                "params": params,
                "reasoning": f"Order lookup for {order_id or email}.",
                "followup_tool": None,
            }

    # ── Rule 4: Support ticket ─────────────────────────────────────────────────
    if intent == "create_ticket":
        order_id = _extract_order_id(msg) or _scan_history(history, _extract_order_id)
        email    = _extract_email(msg)    or _scan_history(history, _extract_email)
        return {
            "tool": "create_support_ticket",
            "params": {
                "customer_name":  "Customer",
                "customer_email": email or "unknown@customer.com",
                "issue_category": "General",
                "description":    msg[:300],
                "priority":       "Medium",
                "order_id":       order_id,
            },
            "reasoning": "User has an issue — creating support ticket.",
            "followup_tool": None,
        }

    # ── Rule 5: Fallback — ask Gemini only for non-action intents ──────────────
    # (general questions, policy lookups, etc. — no tool needed)
    return {
        "tool": None,
        "params": {},
        "reasoning": "No tool action required — answering from knowledge base.",
        "followup_tool": None,
    }


# ── Final Response Generator ───────────────────────────────────────────────────

def generate_final_response(
    user_message: str,
    intent: str,
    rag_context: list[str],
    tool_result: Optional[dict],
    tool_name: Optional[str],
    conversation_history: list[dict],
    model,
) -> str:
    """Use Gemini to generate the final customer-facing response."""

    context_block = ""
    if rag_context:
        context_block = "KNOWLEDGE BASE CONTEXT:\n" + "\n---\n".join(rag_context[:3])

    tool_block = ""
    if tool_result:
        tool_block = f"TOOL ({tool_name}) RESULT:\n{json.dumps(tool_result, indent=2)}"

    history_block = ""
    if conversation_history:
        recent = conversation_history[-6:]  # last 3 exchanges
        history_lines = []
        for msg in recent:
            role = "Customer" if msg["role"] == "user" else "Agent"
            history_lines.append(f"{role}: {msg['content']}")
        history_block = "CONVERSATION HISTORY:\n" + "\n".join(history_lines)

    system_prompt = """You are Alex, a helpful and friendly AI customer support agent for TechStore.

STRICT RULES — never break these:
1. ONLY say an email was sent if tool_result contains "email_sent" with success=true.
2. If email_sent.mode = "real"      → say "Email sent successfully to <address>. Message ID: <id>"
3. If email_sent.mode = "simulated" → say "Email logged (simulation mode — add SMTP config to .env to send real emails)"
4. If there is NO email_sent key    → do NOT mention email at all — do not say "I've sent" or "I've initiated"
5. Never invent ticket IDs, order IDs, or message IDs — only use what the tool returned.
6. If a tool returned success=false, apologise and explain what went wrong.
7. Format order details with clear bullet points.
8. Always end with an offer to help further.
"""

    full_prompt = f"""{system_prompt}

{history_block}

{context_block}

{tool_block}

Customer's current message: "{user_message}"
Detected intent: {intent}

Provide a helpful, natural response:"""

    try:
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error generating a response: {str(e)}. Please try again."


# ── Main Agent Orchestrator ────────────────────────────────────────────────────

class CustomerSupportAgent:
    """
    Multi-step agent that orchestrates:
    intent detection → tool selection → RAG retrieval → reasoning → response
    """

    def __init__(self):
        self.model = _configure_gemini()
        self.rag = get_rag_pipeline()
        self.conversation_history: list[dict] = []

    def run(self, user_message: str, show_steps: bool = True) -> AgentResponse:
        """
        Full agent pipeline. Returns an AgentResponse with all step details.
        Supports tool chaining: e.g. check_order_status → simulate_send_email
        """
        steps: list[AgentStep] = []

        # ── Step 1: Intent Detection ──────────────────────────────────────────
        intent = detect_intent(user_message)
        steps.append(AgentStep(
            step="1. Intent Detection",
            description=f"Detected intent: **{intent}**",
            result=intent
        ))

        # ── Step 2: RAG Retrieval ─────────────────────────────────────────────
        rag_context = []
        if self.rag.is_ready:
            rag_context = self.rag.retrieve(user_message, top_k=4)
        steps.append(AgentStep(
            step="2. RAG Retrieval",
            description=f"Retrieved {len(rag_context)} relevant chunks from knowledge base.",
            result=f"{len(rag_context)} chunks"
        ))

        # ── Step 3: Tool Decision (with history context) ──────────────────────
        tool_decision = decide_tool(
            intent, user_message, self.model, self.conversation_history
        )
        tool_name = tool_decision.get("tool")
        tool_params = tool_decision.get("params", {})
        tool_reasoning = tool_decision.get("reasoning", "")
        followup_tool = tool_decision.get("followup_tool")  # e.g. send email after order lookup

        steps.append(AgentStep(
            step="3. Tool Decision",
            description=f"Tool selected: **{tool_name or 'None'}** — {tool_reasoning}"
                        + (f" → followup: **{followup_tool.get('tool')}**" if followup_tool else ""),
            result=tool_name
        ))

        # ── Step 4: Tool Execution (with optional chaining) ───────────────────
        tool_result = None
        final_tool_name = tool_name  # track which tool result to surface

        if tool_name:
            tool_result = execute_tool(tool_name, tool_params)
            success = tool_result.get("success", False)
            steps.append(AgentStep(
                step="4a. Tool Execution",
                description=f"Executed `{tool_name}` — {'✅ Success' if success else '❌ Failed'}",
                result=json.dumps(tool_result, indent=2)[:300]
            ))

            # ── Tool Chaining: if followup_tool is set and first tool succeeded ──
            if followup_tool and followup_tool.get("use_order_data") and success:
                orders = tool_result.get("orders", [])
                if orders:
                    order = orders[0]
                    # Use email from order record (always present) — no guessing
                    customer_email = order.get("customer_email")
                    if not customer_email:
                        customer_email = self._extract_email_from_history()
                    if not customer_email:
                        customer_email = "customer@techstore.com"

                    email_body   = self._build_order_email(order)
                    email_result = execute_tool("send_email", {
                        "to_email": customer_email,
                        "subject":  f"Your TechStore Order Update — {order['order_id']}",
                        "body":     email_body,
                    })
                    tool_result["email_sent"] = email_result
                    final_tool_name = "check_order_status + send_email"
                    email_ok = email_result.get("success")
                    email_mode = email_result.get("mode", "unknown")
                    steps.append(AgentStep(
                        step="4b. send_email",
                        description=(
                            f"{'✅' if email_ok else '❌'} Email {'sent' if email_ok else 'FAILED'} "
                            f"to `{customer_email}` — mode: **{email_mode}** — "
                            f"ID: `{email_result.get('message_id','?')}`"
                        ),
                        result=json.dumps(email_result, indent=2)[:300]
                    ))
                else:
                    steps.append(AgentStep(
                        step="4b. send_email",
                        description="⚠️ Skipped — no order data returned to include in email.",
                        result="skipped"
                    ))

            # ── Direct simulate_send_email (no chaining) ──────────────────────
            elif tool_name == "simulate_send_email" and not tool_params.get("to_email"):
                # Params incomplete — try to fill from history
                email = self._extract_email_from_history()
                if email:
                    fixed_result = execute_tool("simulate_send_email", {
                        **tool_params,
                        "to_email": email,
                    })
                    tool_result = fixed_result

        else:
            steps.append(AgentStep(
                step="4. Tool Execution",
                description="No tool required — answering from knowledge base.",
                result="skipped"
            ))

        # ── Step 5: Response Generation ───────────────────────────────────────
        final_answer = generate_final_response(
            user_message=user_message,
            intent=intent,
            rag_context=rag_context,
            tool_result=tool_result,
            tool_name=final_tool_name,
            conversation_history=self.conversation_history,
            model=self.model,
        )
        steps.append(AgentStep(
            step="5. Response Generation",
            description="Generated final response using Gemini gemini-2.5-flash.",
            result="✅ Done"
        ))

        # ── Update History ────────────────────────────────────────────────────
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": final_answer})

        return AgentResponse(
            final_answer=final_answer,
            intent=intent,
            tool_used=final_tool_name,
            tool_result=tool_result,
            rag_context=rag_context,
            steps=steps,
        )

    # ── Helper Methods ─────────────────────────────────────────────────────────

    def _extract_email_from_history(self) -> Optional[str]:
        """Scan conversation history for an email address."""
        email_pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
        for msg in reversed(self.conversation_history):
            match = email_pattern.search(msg.get("content", ""))
            if match:
                return match.group(0)
        return None

    def _extract_order_id_from_history(self) -> Optional[str]:
        """Scan conversation history for an order ID like ORD-001."""
        order_pattern = re.compile(r"ORD-\d+", re.IGNORECASE)
        for msg in reversed(self.conversation_history):
            match = order_pattern.search(msg.get("content", ""))
            if match:
                return match.group(0).upper()
        return None

    @staticmethod
    def _build_order_email(order: dict) -> str:
        """Build a formatted email body from order data."""
        return f"""Dear {order.get('customer_name', 'Valued Customer')},

Thank you for shopping with TechStore! Here is your order update:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ORDER DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Order ID:           {order.get('order_id', 'N/A')}
  Product:            {order.get('product', 'N/A')}
  Quantity:           {order.get('quantity', 'N/A')}
  Order Total:        {order.get('amount', 'N/A')}
  Order Date:         {order.get('order_date', 'N/A')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SHIPPING STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Status:             {order.get('status', 'N/A')}
  Estimated Delivery: {order.get('estimated_delivery', 'N/A')}
  Tracking Number:    {order.get('tracking_number', 'N/A')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If you have any questions, reply to this email or contact us at:
support@techstore.com | 1-800-TECH-HELP

Best regards,
TechStore Customer Support Team
"""

    def clear_history(self):
        self.conversation_history = []

    def add_document(self, text: str, source: str = "uploaded") -> int:
        return self.rag.add_document(text, source)

    def add_pdf_bytes(self, pdf_bytes: bytes, source: str = "doc.pdf") -> int:
        return self.rag.add_pdf_bytes(pdf_bytes, source)
