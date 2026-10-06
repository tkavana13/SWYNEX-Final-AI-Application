# Support Ticket Triage Assistant

A small AI application that reads an incoming customer-support message and decides how it
should be handled: what category it belongs to, how urgent it is, which team should own it,
whether a person needs to double-check it, and a first-reply draft to speed up the response.

Final deliverable for the SWYNEX AI Internship, bringing together Tasks 1–3 into one
runnable application.


---

## 1. The problem

Support teams receive a constant stream of tickets that all need the same first step:
figure out what the ticket is about and who should handle it. Done by hand, this is slow
and inconsistent — two agents can categorize the same ticket differently, and low-priority
questions can sit in the same queue as urgent outages.

**Target user:** a support team lead or triage agent who wants incoming tickets sorted and
routed automatically, with a clear signal for when to step in personally.

**Task:** given the raw text of a ticket, predict one of four categories — **Billing**,
**Technical**, **Account**, **General** — and turn that prediction into an actionable
routing decision.

## 2. Method

**Model:** a TF-IDF vectorizer (unigrams + bigrams) feeding a multinomial Logistic
Regression classifier — scikit-learn, no external API, no GPU required.

**Training data:** 160 labeled example tickets, 40 per category, written to reflect
realistic phrasing for each category while minimizing vocabulary overlap between
categories (see Task 3 for the failure-case work that shaped this).

**Evaluation:** 5-fold stratified cross-validation, since a single train/test split on a
dataset this size is too noisy to trust on its own.
- **Cross-validated macro F1: ~0.94**
- **Held-out split macro F1: ~0.93**

**The intelligent layer on top of the prediction** (`ticket_intelligence.py`):
- **Priority & SLA** — derived from category, escalated to High if urgency language
  ("urgent", "asap", "locked out", etc.) appears in the text
- **Routing team** — which internal team should own the ticket
- **Confidence-gated human review** — predictions below 40% confidence are flagged for a
  person instead of being auto-routed
- **Auto-response draft** — a first-reply template per category

**Error handling:** `classify_ticket()` never raises. Empty text, `None`, wrong types,
too-short text, and oversized text (truncated rather than rejected) all return a structured
result instead of crashing the app — verified with explicit test cases in
`demo_task3.ipynb` (carried over from Task 3).

## 3. The application

A Flask backend (`app.py`) serves a single-page interface (`static/`) where anyone can
paste a support message and see the full routing decision — no command line required.

**API:**
- `POST /api/classify` — `{"text": "..."}` → category, confidence, priority, SLA, routing
  team, human-review flag, and a draft reply
- `GET /api/metrics` — the model's cross-validated macro F1 and dataset size, so the
  evaluation numbers above are visible from the running app itself, not just this README

### Run it locally

```bash
pip install -r requirements.txt
python app.py        # or: py app.py  (Windows)
```

Then open **http://127.0.0.1:5000** in a browser. Click one of the example chips (or paste
your own message) and press **Classify ticket**.

### Demo

Take a look at the video below to see the project in action, including its main features, functionality, and overall user experience.

## 📹 Demo


https://github.com/user-attachments/assets/e872fd7a-11ae-48e8-9008-b25a02182799


## 4. Limitations

- **Small, synthetic-style training set.** 160 examples is enough to demonstrate the
  approach cleanly, but a production system would need real historical tickets — actual
  customer language is messier (typos, mixed languages, multiple issues in one message)
  than the training examples here.
- **English only.** No handling for other languages or heavily code-mixed text.
- **Single-label only.** A ticket that's genuinely both a Billing and a Technical issue
  gets forced into one category.
- **Static model.** The model is trained once at startup; it doesn't learn from corrections
  an agent makes, though `classify_ticket()`'s structured output is designed to make adding
  that feedback loop straightforward later.
- **The 0.94 macro F1 reflects this specific dataset.** On messier, real-world tickets
  with more overlapping vocabulary, accuracy would likely be lower — the honest failure
  analysis in Task 3 is the more realistic preview of where a larger real dataset would
  still trip the model up.

## 5. Ethics notes

- **Human-in-the-loop by design, not by accident.** The confidence-gated review flag exists
  specifically so low-confidence predictions reach a person before any automated action is
  taken on them — this app is built to assist triage, not to fully replace a human decision
  on ambiguous tickets.
- **No sensitive data is stored or logged.** Ticket text is processed in memory for a
  single request and not written to disk, a database, or any third-party service.
- **Transparency of confidence.** The interface always shows the model's confidence
  alongside its prediction, rather than presenting a single "answer" as if it were certain.
- **Risk of miscategorization at the edges.** As the Task 3 failure analysis showed, short
  or ambiguous tickets are the most likely to be misrouted. In a real deployment, priority
  categories tied to safety or account security (e.g., "Account" tickets involving a locked
  account) should have a lower auto-review threshold than lower-stakes categories, since the
  cost of a missed urgent ticket is higher than the cost of an unnecessary human review.
- **No demographic or personal data is used or required** — classification is based solely
  on the ticket's text content.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Flask backend — trains the model at startup, serves the UI and API |
| `ticket_intelligence.py` | Training data, model pipeline, cross-validation, and the `classify_ticket()` intelligent feature with error handling |
| `static/` | The single-page frontend (HTML/CSS/JS, no build step) |
| `requirements.txt` | Dependencies: Flask, scikit-learn, numpy |
