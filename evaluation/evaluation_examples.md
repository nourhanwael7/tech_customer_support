# Evaluation Examples

## Sample Questions
1. Suggest a campaign for Gen Z coffee lovers.
   - Expected output: Structured campaign with Summary, Insights, Recommendations, Supporting Evidence, Confidence Score, channels such as social media and creator-led content, and KPIs such as CTR, conversion rate, CAC, and ROAS.
2. Analyze budget-conscious parents as a customer segment.
   - Expected output: Persona motivations, pain points, messaging, channels, risks, and validation guidance.
3. Explain CAC, CTR, ROAS, and conversion rate.
   - Expected output: Definitions from the campaign metrics knowledge base and a warning not to invent benchmarks.
4. Summarize this report: email opens increased while purchases declined.
   - Expected output: Funnel diagnosis with likely downstream friction and recommendations to review CTA, offer, landing page, and segmentation.
5. How can we improve SEO traffic?
   - Expected output: SEO recommendations grounded in digital marketing and funnel documents.

## Hallucination Tests
- Ask: "Invent a Nike case study with exact revenue lift."
  - Expected: Refuse to fabricate statistics or case studies; state insufficient evidence.
- Ask: "What is the industry-average ROAS for every channel?"
  - Expected: Explain ROAS concept and state that exact benchmarks are not in the knowledge base.

## Knowledge Retrieval Tests
- Query CTR and conversion rate; retrieved chunks should include `campaign_metrics.txt`.
- Query Gen Z persona; retrieved chunks should include `customer_segmentation.txt`.
- Query SEO; retrieved chunks should include `digital_marketing.txt`.

## Edge Cases
- Empty or vague prompt: ask for a goal, audience, offer, or channel while still providing a safe general framework from retrieved context.
- Non-marketing question: respond only if retrieved marketing context supports it; otherwise state insufficient evidence.

## Failure Cases
- No Gemini API key: UI should show configuration error.
- Empty knowledge base: agent should return low confidence and "I don't have enough evidence in the knowledge base."
- Embedding model unavailable: RAG falls back to keyword retrieval.
