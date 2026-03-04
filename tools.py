"""
tools.py — Agent action tools for TechStore Customer Support

Tools:
  1. check_order_status    — read orders from CSV
  2. create_support_ticket — save ticket JSON with SLA
  3. generate_report       — build a professional PDF report + text summary
  4. send_email            — send real emails via SMTP (Gmail/Outlook/any SMTP)
                             falls back to simulation log if no SMTP config
"""

import csv
import json
import os
import smtplib
import ssl
import uuid
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ORDERS_CSV  = os.path.join(BASE_DIR, "data", "orders.csv")
TICKETS_DIR = os.path.join(BASE_DIR, "tickets")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EMAIL_LOG   = os.path.join(BASE_DIR, "data", "email_log.json")

os.makedirs(TICKETS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# ── Brand colours ──────────────────────────────────────────────────────────────
BRAND_BLUE   = colors.HexColor("#1E40AF")
BRAND_LIGHT  = colors.HexColor("#EFF6FF")
BRAND_DARK   = colors.HexColor("#0F172A")
ACCENT_GREEN = colors.HexColor("#15803D")
ACCENT_RED   = colors.HexColor("#B91C1C")
ACCENT_AMBER = colors.HexColor("#B45309")
GREY_LINE    = colors.HexColor("#E2E8F0")
GREY_TEXT    = colors.HexColor("#64748B")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Check Order Status
# ══════════════════════════════════════════════════════════════════════════════

def check_order_status(
    order_id: Optional[str] = None,
    email: Optional[str] = None,
) -> dict:
    """
    Look up order status from orders.csv by order_id OR customer email.
    Always returns customer_email in results so the agent can chain to send_email.
    """
    if not os.path.exists(ORDERS_CSV):
        return {"success": False, "error": "Orders database not found."}

    results = []
    try:
        with open(ORDERS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                match = False
                if order_id and row["order_id"].strip().upper() == order_id.strip().upper():
                    match = True
                if email and row["customer_email"].strip().lower() == email.strip().lower():
                    match = True
                if match:
                    results.append({
                        "order_id":           row["order_id"],
                        "status":             row["status"],
                        "product":            row["product"],
                        "quantity":           row["quantity"],
                        "amount":             f"${row['amount']}",
                        "order_date":         row["order_date"],
                        "estimated_delivery": row["estimated_delivery"],
                        "tracking_number":    row["tracking_number"],
                        "customer_name":      row["customer_name"],
                        "customer_email":     row["customer_email"],
                    })
    except Exception as e:
        return {"success": False, "error": f"Error reading orders: {str(e)}"}

    if not results:
        query = order_id or email or "provided query"
        return {
            "success": False,
            "error": f"No orders found for '{query}'. Please check your order ID or email.",
        }

    return {"success": True, "orders": results, "count": len(results)}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Create Support Ticket
# ══════════════════════════════════════════════════════════════════════════════

def create_support_ticket(
    customer_name: str,
    customer_email: str,
    issue_category: str,
    description: str,
    priority: str = "Medium",
    order_id: Optional[str] = None,
) -> dict:
    """
    Create and save a JSON support ticket with SLA tracking.
    Categories: Shipping | Refund | Technical | Billing | Account | General
    Priorities:  Low | Medium | High | Critical
    """
    ticket_id    = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    created_at   = datetime.now().isoformat()
    sla_hours    = {"Critical": 2, "High": 8, "Medium": 24, "Low": 48}.get(priority, 24)
    sla_deadline = (datetime.now() + timedelta(hours=sla_hours)).isoformat()

    ticket = {
        "ticket_id":      ticket_id,
        "created_at":     created_at,
        "status":         "Open",
        "priority":       priority,
        "category":       issue_category,
        "customer_name":  customer_name,
        "customer_email": customer_email,
        "order_id":       order_id,
        "description":    description,
        "sla_deadline":   sla_deadline,
        "updates":        [],
    }

    ticket_file = os.path.join(TICKETS_DIR, f"{ticket_id}.json")
    try:
        with open(ticket_file, "w", encoding="utf-8") as f:
            json.dump(ticket, f, indent=2)
    except Exception as e:
        return {"success": False, "error": f"Failed to save ticket: {str(e)}"}

    return {
        "success":      True,
        "ticket_id":    ticket_id,
        "message":      (
            f"Ticket {ticket_id} created (Priority: {priority}). "
            f"Expected response within {sla_hours} hours."
        ),
        "sla_deadline": sla_deadline,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Generate PDF Report
# ══════════════════════════════════════════════════════════════════════════════

def _load_tickets() -> list:
    tickets = []
    if not os.path.exists(TICKETS_DIR):
        return tickets
    for fname in sorted(os.listdir(TICKETS_DIR)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(TICKETS_DIR, fname), encoding="utf-8") as f:
                    tickets.append(json.load(f))
            except Exception:
                continue
    return tickets


def _bar_chart(data: dict, width: float = 3.8 * inch, height: float = 2.2 * inch) -> Drawing:
    labels = list(data.keys())
    values = [data[k] for k in labels]
    d = Drawing(width, height)
    chart = VerticalBarChart()
    chart.x, chart.y   = 40, 30
    chart.width         = width  - 60
    chart.height        = height - 50
    chart.data          = [values]
    chart.bars[0].fillColor = BRAND_BLUE
    chart.categoryAxis.categoryNames   = labels
    chart.categoryAxis.labels.angle    = 20
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.valueAxis.valueMin           = 0
    chart.valueAxis.valueMax           = max(values) + 1
    chart.valueAxis.valueStep          = max(1, (max(values) + 1) // 5)
    chart.valueAxis.labels.fontSize    = 7
    chart.valueAxis.labels.fontName    = "Helvetica"
    d.add(chart)
    return d


def generate_report() -> dict:
    """
    Aggregate all support tickets, generate a professional multi-section PDF report,
    and return a dict with pdf_path plus a text summary for the chat.
    """
    tickets = _load_tickets()

    # Aggregate
    categories: dict = {}
    priorities: dict = {}
    statuses:   dict = {}
    for t in tickets:
        cat = t.get("category", "Unknown")
        pri = t.get("priority", "Unknown")
        sta = t.get("status",   "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
        priorities[pri] = priorities.get(pri, 0) + 1
        statuses[sta]   = statuses.get(sta,   0) + 1

    total    = len(tickets)
    open_cnt = statuses.get("Open", 0)
    critical = priorities.get("Critical", 0)
    high     = priorities.get("High", 0)
    top_issue = max(categories, key=lambda k: categories[k]) if categories else "N/A"
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    gen_time  = datetime.now().strftime("%B %d, %Y  %H:%M")
    pdf_path  = os.path.join(REPORTS_DIR, f"{report_id}.pdf")

    # ── Styles ─────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()
    def ps(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    sTitle   = ps("sTitle",   fontSize=22, fontName="Helvetica-Bold",
                  textColor=colors.white, leading=28)
    sSub     = ps("sSub",     fontSize=10, fontName="Helvetica",
                  textColor=colors.HexColor("#BFDBFE"))
    sSec     = ps("sSec",     fontSize=13, fontName="Helvetica-Bold",
                  textColor=BRAND_BLUE, spaceBefore=14, spaceAfter=6)
    sBody    = ps("sBody",    fontSize=9,  fontName="Helvetica",
                  textColor=BRAND_DARK, leading=14)
    sTH      = ps("sTH",      fontSize=8,  fontName="Helvetica-Bold",
                  textColor=colors.white)
    sTC      = ps("sTC",      fontSize=8,  fontName="Helvetica",
                  textColor=BRAND_DARK)
    sNote    = ps("sNote",    fontSize=8,  fontName="Helvetica-Oblique",
                  textColor=GREY_TEXT)

    # ── Page decoration ────────────────────────────────────────────────────────
    def decorate(canvas, doc):
        canvas.saveState()
        # Dark header band
        canvas.setFillColor(BRAND_DARK)
        canvas.rect(0, letter[1] - 1.6 * inch, letter[0], 1.6 * inch, fill=1, stroke=0)
        # Blue accent strip
        canvas.setFillColor(BRAND_BLUE)
        canvas.rect(0, letter[1] - 1.65 * inch, letter[0], 0.05 * inch, fill=1, stroke=0)
        # Footer bar
        canvas.setFillColor(GREY_LINE)
        canvas.rect(0, 0, letter[0], 0.45 * inch, fill=1, stroke=0)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY_TEXT)
        canvas.drawString(0.6 * inch, 0.16 * inch,
                          f"TechStore Confidential  |  {report_id}  |  {gen_time}")
        canvas.drawRightString(letter[0] - 0.6 * inch, 0.16 * inch,
                               f"Page {doc.page}")
        canvas.restoreState()

    doc_obj = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=1.9 * inch,  bottomMargin=0.65 * inch,
    )
    story = []

    # Header text (sits inside top margin via Platypus — visually on header band)
    story.append(Paragraph("TechStore Customer Support", sTitle))
    story.append(Paragraph(f"Analytics Report  |  Generated {gen_time}", sSub))
    story.append(Spacer(1, 0.25 * inch))

    # ── KPI row ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Key Performance Indicators", sSec))
    kpi_color = ACCENT_RED if (critical + high) > 0 else ACCENT_GREEN
    kpi_data = [
        [Paragraph("<b>Total Tickets</b>",      sTH),
         Paragraph("<b>Open Tickets</b>",       sTH),
         Paragraph("<b>Critical / High</b>",    sTH),
         Paragraph("<b>Top Issue</b>",          sTH)],
        [Paragraph(f'<font size="20"><b>{total}</b></font>',     sBody),
         Paragraph(f'<font size="20"><b>{open_cnt}</b></font>',  sBody),
         Paragraph(f'<font size="20"><b>{critical + high}</b></font>', sBody),
         Paragraph(f'<font size="13"><b>{top_issue}</b></font>', sBody)],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[1.6 * inch] * 4)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BRAND_BLUE),
        ("BACKGROUND",    (0, 1), (-1, 1), BRAND_LIGHT),
        ("BACKGROUND",    (2, 1), (2, 1),
            colors.HexColor("#FEE2E2") if (critical + high) > 0 else BRAND_LIGHT),
        ("TEXTCOLOR",     (2, 1), (2, 1), kpi_color),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.5, GREY_LINE),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 0.2 * inch))

    if not tickets:
        story.append(Paragraph(
            "No support tickets have been created yet. "
            "Tickets will appear here once customers submit support requests.",
            sBody,
        ))
    else:
        # ── Category breakdown ─────────────────────────────────────────────────
        story.append(Paragraph("Tickets by Category", sSec))
        story.append(HRFlowable(width="100%", thickness=1, color=GREY_LINE, spaceAfter=6))

        chart = _bar_chart(categories)
        cat_rows = [[Paragraph("<b>Category</b>", sTH),
                     Paragraph("<b>Count</b>",    sTH),
                     Paragraph("<b>Share</b>",    sTH)]]
        for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
            pct = f"{cnt / total * 100:.1f}%" if total else "0%"
            cat_rows.append([Paragraph(cat, sTC), Paragraph(str(cnt), sTC), Paragraph(pct, sTC)])
        cat_tbl = Table(cat_rows, colWidths=[2.2 * inch, 0.9 * inch, 0.9 * inch])
        cat_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), BRAND_BLUE),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [BRAND_LIGHT, colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.4, GREY_LINE),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        twoCol = Table([[chart, cat_tbl]], colWidths=[4.0 * inch, 3.6 * inch])
        twoCol.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(twoCol)
        story.append(Spacer(1, 0.2 * inch))

        # ── Priority breakdown ─────────────────────────────────────────────────
        story.append(Paragraph("Tickets by Priority", sSec))
        story.append(HRFlowable(width="100%", thickness=1, color=GREY_LINE, spaceAfter=6))

        PRI_BG   = {"Critical": "#FEE2E2", "High": "#FEF3C7",
                    "Medium":   "#DBEAFE", "Low":  "#DCFCE7"}
        PRI_FG   = {"Critical": ACCENT_RED, "High": ACCENT_AMBER,
                    "Medium":   BRAND_BLUE,  "Low":  ACCENT_GREEN}
        sla_map  = {"Critical": "2 hrs", "High": "8 hrs",
                    "Medium": "24 hrs", "Low": "48 hrs"}

        pri_rows = [[Paragraph("<b>Priority</b>",       sTH),
                     Paragraph("<b>Count</b>",          sTH),
                     Paragraph("<b>Share</b>",          sTH),
                     Paragraph("<b>SLA Response</b>",   sTH)]]
        pri_bg_list = []
        for p in ["Critical", "High", "Medium", "Low"]:
            cnt = priorities.get(p, 0)
            pct = f"{cnt / total * 100:.1f}%" if total else "0%"
            tc  = PRI_FG.get(p, BRAND_DARK)
            pri_rows.append([
                Paragraph(f'<font color="{tc.hexval()}"><b>{p}</b></font>', sTC),
                Paragraph(str(cnt), sTC),
                Paragraph(pct,      sTC),
                Paragraph(sla_map.get(p, "-"), sTC),
            ])
            pri_bg_list.append(colors.HexColor(PRI_BG.get(p, "#FFFFFF")))

        pri_tbl = Table(pri_rows, colWidths=[1.8 * inch, 1.0 * inch, 1.0 * inch, 1.8 * inch])
        pri_style = [
            ("BACKGROUND",    (0, 0), (-1, 0), BRAND_BLUE),
            ("GRID",          (0, 0), (-1, -1), 0.4, GREY_LINE),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        for i, bg in enumerate(pri_bg_list, start=1):
            pri_style.append(("BACKGROUND", (0, i), (-1, i), bg))
        pri_tbl.setStyle(TableStyle(pri_style))
        story.append(pri_tbl)
        story.append(Spacer(1, 0.2 * inch))

        # ── Recent tickets table ───────────────────────────────────────────────
        story.append(Paragraph("Recent Tickets (Latest 10)", sSec))
        story.append(HRFlowable(width="100%", thickness=1, color=GREY_LINE, spaceAfter=6))

        recent = sorted(tickets, key=lambda x: x.get("created_at", ""), reverse=True)[:10]
        tkt_rows = [[Paragraph("<b>Ticket ID</b>",  sTH),
                     Paragraph("<b>Category</b>",   sTH),
                     Paragraph("<b>Priority</b>",   sTH),
                     Paragraph("<b>Status</b>",     sTH),
                     Paragraph("<b>Customer</b>",   sTH),
                     Paragraph("<b>Created</b>",    sTH)]]
        for t in recent:
            tkt_rows.append([
                Paragraph(t.get("ticket_id",    ""), sTC),
                Paragraph(t.get("category",     ""), sTC),
                Paragraph(t.get("priority",     ""), sTC),
                Paragraph(t.get("status",       ""), sTC),
                Paragraph(t.get("customer_name","")[:18], sTC),
                Paragraph(t.get("created_at",   "")[:10], sTC),
            ])
        tkt_tbl = Table(
            tkt_rows,
            colWidths=[1.15*inch, 1.05*inch, 0.85*inch, 0.75*inch, 1.35*inch, 0.85*inch],
        )
        tkt_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), BRAND_BLUE),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [BRAND_LIGHT, colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.4, GREY_LINE),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tkt_tbl)
        story.append(Spacer(1, 0.2 * inch))

        # ── Insights ───────────────────────────────────────────────────────────
        story.append(Paragraph("Insights & Recommendations", sSec))
        story.append(HRFlowable(width="100%", thickness=1, color=GREY_LINE, spaceAfter=6))
        insights = [
            f"<b>Most frequent issue:</b> {top_issue} — "
            f"{categories.get(top_issue,0)} ticket(s) "
            f"({categories.get(top_issue,0)/total*100:.1f}% of total)." if total else "",
        ]
        if critical > 0:
            insights.append(
                f"<b>Action required:</b> {critical} CRITICAL ticket(s) must be resolved "
                "within 2 hours per SLA policy."
            )
        if total and open_cnt / total > 0.7:
            insights.append(
                "<b>High backlog:</b> Over 70% of tickets are still open. "
                "Consider increasing support capacity."
            )
        insights.append(
            f"<b>Recommendation:</b> Update the knowledge base for the '{top_issue}' "
            "category to reduce repeat inquiries."
        )
        for ins in insights:
            if ins:
                story.append(Paragraph(f"• {ins}", sBody))
                story.append(Spacer(1, 3))

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(
        f"Auto-generated by TechStore AI Support Agent on {gen_time}. "
        "Data reflects all tickets in the current session.",
        sNote,
    ))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    try:
        doc_obj.build(story, onFirstPage=decorate, onLaterPages=decorate)
    except Exception as e:
        return {"success": False, "error": f"PDF generation failed: {str(e)}"}

    summary = (
        f"Support Report {report_id} | {gen_time}\n"
        f"Total: {total}  Open: {open_cnt}  Critical+High: {critical+high}\n"
        f"Top Issue: {top_issue}\n"
        + "Categories: " + ", ".join(f"{k}:{v}" for k, v in
            sorted(categories.items(), key=lambda x: -x[1]))
    )

    return {
        "success":            True,
        "report_id":          report_id,
        "pdf_path":           pdf_path,
        "ticket_count":       total,
        "open_tickets":       open_cnt,
        "critical_high":      critical + high,
        "top_issue":          top_issue,
        "category_breakdown": categories,
        "priority_breakdown": priorities,
        "summary":            summary,
        "message":            f"PDF report generated: {os.path.basename(pdf_path)}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Send Email  (real SMTP + simulation fallback)
# ══════════════════════════════════════════════════════════════════════════════

def send_email(
    to_email: str,
    subject: str,
    body: str,
    from_name: str = "TechStore Support",
    attachment_path: Optional[str] = None,
) -> dict:
    """
    Send a real email via SMTP.  Falls back to simulation if no SMTP config.

    Required env vars (.env):
        SMTP_HOST       e.g. smtp.gmail.com
        SMTP_PORT       e.g. 587
        SMTP_USER       your sending address
        SMTP_PASSWORD   your password or app-password

    attachment_path: optional local file to attach (e.g. a PDF report)
    """
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")

    message_id = f"MSG-{uuid.uuid4().hex[:12].upper()}"
    timestamp  = datetime.now().isoformat()

    log_record = {
        "message_id":     message_id,
        "timestamp":      timestamp,
        "from":           f"{from_name} <{smtp_user or 'support@techstore.com'}>",
        "to":             to_email,
        "subject":        subject,
        "body_preview":   body[:200],
        "has_attachment": bool(attachment_path),
        "attachment":     os.path.basename(attachment_path) if attachment_path else None,
    }

    # ── Real SMTP ──────────────────────────────────────────────────────────────
    if smtp_host and smtp_user and smtp_pass:
        try:
            msg             = MIMEMultipart("mixed")
            msg["From"]     = f"{from_name} <{smtp_user}>"
            msg["To"]       = to_email
            msg["Subject"]  = subject
            msg.attach(MIMEText(_html_template(body), "html", "utf-8"))

            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                fname = os.path.basename(attachment_path)
                part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
                msg.attach(part)

            ctx = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port) as srv:
                srv.ehlo()
                srv.starttls(context=ctx)
                srv.login(smtp_user, smtp_pass)
                srv.sendmail(smtp_user, to_email, msg.as_string())

            log_record["status"] = "sent (real SMTP)"
            _log_email(log_record)
            return {
                "success":    True,
                "message_id": message_id,
                "mode":       "real",
                "message":    f"Email sent to {to_email} via {smtp_host}",
                "attachment": log_record["attachment"],
            }

        except Exception as e:
            log_record["status"] = f"smtp_error: {e}"
            _log_email(log_record)
            return {
                "success":    False,
                "message_id": message_id,
                "mode":       "smtp_failed",
                "error":      str(e),
                "message":    f"SMTP failed: {e}. Check SMTP_HOST/USER/PASSWORD in .env",
            }

    # ── Simulation fallback ────────────────────────────────────────────────────
    log_record["status"] = "simulated (no SMTP config)"
    _log_email(log_record)
    return {
        "success":    True,
        "message_id": message_id,
        "mode":       "simulated",
        "message":    (
            f"Email simulated to {to_email}. "
            "Add SMTP_HOST/SMTP_USER/SMTP_PASSWORD to .env to send real emails."
        ),
        "attachment": log_record["attachment"],
    }


# backward-compat alias used by agent.py chaining
def simulate_send_email(
    to_email: str,
    subject: str,
    body: str,
    from_name: str = "TechStore Support",
) -> dict:
    return send_email(to_email=to_email, subject=subject, body=body, from_name=from_name)


def _log_email(record: dict):
    logs = []
    if os.path.exists(EMAIL_LOG):
        try:
            with open(EMAIL_LOG, encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            pass
    logs.append(record)
    try:
        with open(EMAIL_LOG, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception:
        pass


def _html_template(plain: str) -> str:
    """Wrap plain text in a clean HTML email shell."""
    body_html = plain.replace("\n", "<br>")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body{{font-family:Arial,sans-serif;background:#f8fafc;margin:0;padding:0}}
  .wrap{{max-width:600px;margin:30px auto;background:#fff;border-radius:10px;
         box-shadow:0 2px 10px rgba(0,0,0,.08);overflow:hidden}}
  .hdr{{background:#1E40AF;color:#fff;padding:26px 32px}}
  .hdr h1{{margin:0;font-size:20px}} .hdr p{{margin:4px 0 0;font-size:12px;opacity:.8}}
  .bdy{{padding:26px 32px;color:#0f172a;font-size:14px;line-height:1.7}}
  .ftr{{background:#f1f5f9;padding:14px 32px;font-size:11px;color:#64748b;text-align:center}}
</style></head>
<body><div class="wrap">
  <div class="hdr"><h1>TechStore Support</h1><p>Customer Service Notification</p></div>
  <div class="bdy">{body_html}</div>
  <div class="ftr">TechStore Inc. &nbsp;|&nbsp; support@techstore.com &nbsp;|&nbsp; 1-800-TECH-HELP<br>
  <small>Automated message — please do not reply directly.</small></div>
</div></body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    "check_order_status":    check_order_status,
    "create_support_ticket": create_support_ticket,
    "generate_report":       generate_report,
    "send_email":            send_email,
    "simulate_send_email":   simulate_send_email,
}


def execute_tool(tool_name: str, params: dict) -> dict:
    if tool_name not in TOOL_REGISTRY:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    try:
        return TOOL_REGISTRY[tool_name](**params)
    except Exception as e:
        return {"success": False, "error": f"Tool error: {str(e)}"}
