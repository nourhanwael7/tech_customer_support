# TechStore Multi-Step AI Customer Support Agent

A production-style AI customer support system built with Google Gemini 2.5 Flash, FAISS-based RAG retrieval, real SMTP email delivery, PDF report generation, and a Streamlit interface.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Application](#running-the-application)
8. [Agent Pipeline](#agent-pipeline)
9. [Tools Reference](#tools-reference)
10. [RAG Pipeline](#rag-pipeline)
11. [Email Configuration](#email-configuration)
12. [Sample Data](#sample-data)
13. [Usage Examples](#usage-examples)
14. [Tech Stack](#tech-stack)

---

## Overview

This system implements a multi-step AI agent that processes customer support requests through a deterministic pipeline. Each user message passes through intent detection, document retrieval, tool execution, and response generation — giving full visibility into every decision the agent makes.

The agent is designed around a rule-based tool router, meaning tool selection is handled by Python logic rather than delegated entirely to the language model. This prevents hallucinated tool calls and ensures reliable, auditable behavior.

---

## Features

**AI and Reasoning**
- Google Gemini 2.5 Flash for natural language understanding and response generation
- Deterministic rule-based tool router — no LLM hallucinations on tool selection
- Conversation history awareness — extracts order IDs and emails from prior messages
- RAG retrieval with FAISS and SentenceTransformers for grounded responses

**Agent Tools**
- Order status lookup from a CSV database
- Support ticket creation with SLA tracking and JSON persistence
- PDF report generation with charts, breakdowns, and insights using ReportLab
- Real email delivery via SMTP with HTML templates and PDF attachment support

**Interface**
- Streamlit chat UI with user and agent message bubbles
- Agent Inspector panel showing pipeline steps, RAG chunks, and tool results
- PDF upload to extend the knowledge base at runtime
- One-click PDF report generation and download
- Conversation history export as JSON

---

## Project Structure

```
customer_support_agent/
├── app.py                      Streamlit UI — main entry point
├── agent.py                    Multi-step agent orchestrator
├── rag.py                      RAG pipeline (FAISS + SentenceTransformers)
├── tools.py                    Agent tools — order lookup, tickets, reports, email
├── requirements.txt            Python dependencies
├── .env.example                Environment variable template
│
├── data/
│   ├── orders.csv              Sample order database (12 orders)
│   └── email_log.json          Email delivery log (auto-created)
│
├── knowledge_base/
│   ├── faq.txt                 Company FAQ document
│   └── policies.txt            Company policies document
│
├── tickets/                    Support ticket JSON files (auto-created)
└── reports/                    Generated PDF reports (auto-created)
```

---

## Requirements

- Python 3.10 or higher
- A Google Gemini API key (free at aistudio.google.com)
- SMTP credentials for real email sending (optional — simulation mode works without them)

---

## Installation

**Step 1. Extract the project**

```bash
unzip customer_support_agent.zip
cd customer_support_agent
```

**Step 2. Create a virtual environment**

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Step 3. Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in your values. See the Configuration section below.

---

## Configuration

All configuration is handled through the `.env` file in the project root.

```env
# Required — Google Gemini API key
# Get yours free at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Optional — SMTP credentials for real email sending
# Without these, the agent runs in simulation mode
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

Simulation mode is the default when SMTP is not configured. The agent executes the full email pipeline, logs the message to `data/email_log.json`, and reports a message ID in the chat — everything functions except the email is not physically delivered.

---

## Running the Application

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501` by default.

On first launch, the agent loads the knowledge base from `knowledge_base/` and builds the FAISS vector index. This takes a few seconds. Subsequent launches are faster.

---

## Agent Pipeline

Every user message passes through five sequential steps. The Agent Inspector panel in the sidebar shows the result of each step in real time.

```
Step 1 — Intent Detection
         Keyword classifier assigns one of:
         order_status | send_email | generate_report | create_ticket |
         refund_return | shipping | product_info | account | general

Step 2 — RAG Retrieval
         FAISS vector search returns the top 4 most relevant chunks
         from the knowledge base to ground the response in facts.

Step 3 — Tool Decision  (deterministic Python router)
         Python rules — not the LLM — decide which tool to call
         and with what parameters. Order IDs and email addresses
         are extracted by regex from the current message and from
         conversation history.

Step 4 — Tool Execution
         The selected tool runs and returns structured data.
         If a followup tool is required (e.g. send_email after
         check_order_status), it executes immediately after.

Step 5 — Response Generation
         Gemini 2.5 Flash receives the tool result, RAG context,
         and conversation history, then generates the final reply.
```

**Tool chaining example**

User input: `"Send me an email about order ORD-001"`

```
Router detects: email request + order ID "ORD-001"

Step 4a:  check_order_status(order_id="ORD-001")
          -> customer_email = "john.smith@email.com"
          -> status = "Shipped", tracking = "TRK-789456123"

Step 4b:  send_email(
            to_email  = "john.smith@email.com",
            subject   = "Your TechStore Order Update — ORD-001",
            body      = <full order details>
          )
          -> mode = "real", message_id = "MSG-XXXXXXXXXXXXX"

Step 5:   Gemini generates:
          "Email sent to john.smith@email.com.
           Message ID: MSG-XXXXXXXXXXXXX. Is there anything else
           I can help you with?"
```

The router scans conversation history automatically. If the user says "Send me an email about my order" without an order ID, the router looks back through previous messages and extracts the last-mentioned order ID or email address.

---

## Tools Reference

### check_order_status

Looks up one or more orders from `data/orders.csv`.

| Parameter | Type     | Required | Description                          |
|-----------|----------|----------|--------------------------------------|
| order_id  | string   | No       | Order ID in format ORD-XXX           |
| email     | string   | No       | Customer email address               |

At least one parameter must be provided. Returns full order details including product, quantity, amount, status, tracking number, estimated delivery date, and customer email.

---

### create_support_ticket

Creates a JSON support ticket in the `tickets/` directory with SLA deadline tracking.

| Parameter        | Type   | Required | Description                                              |
|------------------|--------|----------|----------------------------------------------------------|
| customer_name    | string | Yes      | Customer full name                                       |
| customer_email   | string | Yes      | Customer email address                                   |
| issue_category   | string | Yes      | Shipping, Refund, Technical, Billing, Account, or General |
| description      | string | Yes      | Description of the issue                                 |
| priority         | string | No       | Low, Medium, High, or Critical (default: Medium)         |
| order_id         | string | No       | Related order ID if applicable                           |

SLA response times by priority:

| Priority | Response Time |
|----------|--------------|
| Critical | 2 hours      |
| High     | 8 hours      |
| Medium   | 24 hours     |
| Low      | 48 hours     |

---

### generate_report

Reads all tickets from `tickets/`, aggregates statistics, and generates a PDF report saved to `reports/`.

No parameters required.

The PDF report includes:
- KPI summary row — total tickets, open count, critical and high priority count, top issue category
- Tickets by category with vertical bar chart and percentage breakdown table
- Tickets by priority with colour-coded rows and SLA reference
- Recent tickets table showing the last 10 created
- Insights and recommendations section

Returns the file path for immediate download in the Streamlit UI.

---

### send_email

Sends an email via SMTP if configured, otherwise falls back to simulation mode.

| Parameter        | Type   | Required | Description                                            |
|------------------|--------|----------|--------------------------------------------------------|
| to_email         | string | Yes      | Recipient email address                                |
| subject          | string | Yes      | Email subject line                                     |
| body             | string | Yes      | Plain text body (wrapped in branded HTML template)     |
| from_name        | string | No       | Sender display name (default: TechStore Support)       |
| attachment_path  | string | No       | Absolute path to a file to attach (e.g. a PDF report)  |

Returns:
- `mode: "real"` when the email was delivered via SMTP
- `mode: "simulated"` when SMTP is not configured
- `mode: "smtp_failed"` when SMTP is configured but delivery failed, with an error description

---

## RAG Pipeline

The retrieval pipeline loads documents from `knowledge_base/`, splits them into overlapping chunks, embeds them with `all-MiniLM-L6-v2` from SentenceTransformers, and stores them in a FAISS `IndexFlatL2` index.

**Chunking parameters**
- Chunk size: ~400 words
- Overlap: 80 words — preserves context at chunk boundaries

**Retrieval**
- L2 nearest-neighbour search returns the top 4 most relevant chunks per query
- Falls back to keyword frequency scoring if the embedding model is unavailable

**Supported file types**
- `.txt` — plain text, loaded directly
- `.pdf` — text extracted with pypdf

**Adding documents at runtime**
Use the Upload section in the Streamlit sidebar. The document is chunked and embedded immediately without restarting the application. Uploaded documents persist for the duration of the session.

---

## Email Configuration

### Gmail

Gmail requires an App Password rather than your regular account password.

1. Enable 2-Step Verification at [myaccount.google.com/security](https://myaccount.google.com/security).
2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).
3. Create a new app password — set the name to "TechStore" or any label you prefer.
4. Copy the generated 16-character password into your `.env` file.

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop
```

The spaces in the app password are normal — include them exactly as shown.

### Outlook / Office 365

```env
SMTP_HOST=smtp.office365.com
SMTP_PORT=587
SMTP_USER=you@outlook.com
SMTP_PASSWORD=your_password
```

### Other Providers

Any provider that supports SMTP with STARTTLS on port 587 is compatible. Set `SMTP_HOST` to your provider's outbound mail server hostname.

---

## Sample Data

The project ships with 12 sample orders covering all status types.

| Order ID | Customer          | Product              | Status     |
|----------|-------------------|----------------------|------------|
| ORD-001  | John Smith        | Laptop Pro X1        | Shipped    |
| ORD-002  | Sarah Johnson     | Wireless Headphones  | Delivered  |
| ORD-003  | Mike Davis        | Smart Watch Ultra    | Processing |
| ORD-004  | Emily Chen        | Tablet 10 Pro        | Cancelled  |
| ORD-005  | Robert Wilson     | Gaming Mouse         | Shipped    |
| ORD-006  | Lisa Anderson     | 4K Monitor           | Delivered  |
| ORD-007  | David Brown       | Mechanical Keyboard  | Processing |
| ORD-008  | Jennifer Taylor   | USB-C Hub            | Shipped    |
| ORD-009  | Michael Martinez  | Webcam HD Pro        | Delivered  |
| ORD-010  | Amanda Garcia     | Laptop Stand         | Processing |
| ORD-011  | James Thompson    | External SSD 1TB     | Shipped    |
| ORD-012  | Patricia White    | Wireless Charger     | Delivered  |

---

## Usage Examples

**Order tracking**
```
"Track my order ORD-001"
"What is the status of ORD-007?"
"Check order for sarah.j@email.com"
```

**Email with order details**
```
"Send me an email about order ORD-001"
"Email me about my order"   (uses order ID from conversation history)
```

**Policy questions answered from knowledge base**
```
"What is your return policy?"
"How long does standard shipping take?"
"Do you offer price matching?"
"What is covered by the warranty?"
```

**Support ticket creation**
```
"My laptop is broken and I need help"
"I received the wrong item for order ORD-005"
"I was charged twice for my order"
```

**Admin report**
```
"Generate a PDF report"
"Show me the support analytics"
```

---

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Language Model  | Google Gemini 2.5 Flash             |
| Embeddings      | all-MiniLM-L6-v2 (SentenceTransformers) |
| Vector Store    | FAISS IndexFlatL2                   |
| PDF Generation  | ReportLab                           |
| PDF Parsing     | pypdf                               |
| Email Delivery  | Python smtplib + MIME (STARTTLS)    |
| UI Framework    | Streamlit                           |
| Data Storage    | CSV (orders), JSON (tickets, logs)  |
| Configuration   | python-dotenv                       |

---

## License

MIT License. Free to use, modify, and distribute.
