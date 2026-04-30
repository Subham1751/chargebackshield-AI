# ChargebackShield AI

An AI-powered dispute resolution system for fintech merchants. ChargebackShield AI combines multimodal evidence analysis with a retrieval-augmented generation pipeline to evaluate chargeback disputes and generate bank-ready response drafts.

Built for the real operational pain point behind merchant chargebacks: teams need to understand evidence quickly, map it to policy, estimate case strength, and respond to banks with clear documentation.

## Live Demo

- **Frontend:** [https://chargebackshield-ai.vercel.app](https://chargebackshield-ai.vercel.app)
- **Backend:** [https://chargebackshield-ai.onrender.com](https://chargebackshield-ai.onrender.com)
- **API Health:** [https://chargebackshield-ai.onrender.com/health](https://chargebackshield-ai.onrender.com/health)

## Screenshots

> Screenshots can be added here after capturing the deployed UI.

| Dispute Upload                    | AI Verdict                       |
| --------------------------------- | -------------------------------- |
| `frontend screenshot placeholder` | `results screenshot placeholder` |

## How It Works

ChargebackShield AI follows a clear fintech dispute workflow:

```text
User
  ↓
Upload evidence image + dispute context
  ↓
Gemini Vision extracts structured evidence from the image
  ↓
ChromaDB retrieves the most relevant dispute policies
  ↓
AI evaluates the evidence against policy criteria
  ↓
System returns win probability, confidence, reasoning, and bank response draft
```

### Pipeline Breakdown

1. **Evidence Upload**
   The merchant uploads an image such as a delivery screenshot, WhatsApp confirmation, invoice, receipt, or transaction proof.

2. **Context Input**
   The merchant adds a short explanation of the dispute, including the customer claim and available proof.

3. **Multimodal Evidence Extraction**
   Gemini Vision analyzes the uploaded image and extracts relevant fields such as document type, visible text, recipient information, timestamps, delivery proof, and key evidence.

4. **RAG Policy Retrieval**
   ChromaDB performs semantic search across dispute policy documents and retrieves the most relevant policy for the case.

5. **AI Verdict Generation**
   Gemini uses the extracted evidence, merchant context, and retrieved policy to generate a structured dispute assessment.

6. **Bank-Ready Output**
   The API returns a practical response package that includes win probability, confidence level, reasoning, missing criteria, recommended actions, and an auto-generated response draft for the bank.

## Tech Stack

| Layer               | Technology                            |
| ------------------- | ------------------------------------- |
| Backend             | Python, FastAPI, Uvicorn              |
| AI                  | Google Gemini API, Gemini 1.5 Flash   |
| Multimodal Input    | Image + text analysis                 |
| Vector Database     | ChromaDB                              |
| Frontend            | HTML, TailwindCSS, Vanilla JavaScript |
| Backend Deployment  | Render                                |
| Frontend Deployment | Vercel                                |

## Core Features

- Upload dispute evidence as an image
- Add merchant-side dispute context
- Extract visual and textual evidence using Gemini Vision
- Retrieve relevant dispute policies using ChromaDB
- Generate AI-assisted dispute verdicts
- Return win probability and confidence score
- Explain reasoning in a clear, professional format
- Draft a bank-ready dispute response
- Expose health, policy, and metrics endpoints for demos and monitoring

## Project Structure

```text
chargeback-ai/
├── backend/
│   ├── main.py              # FastAPI app, Gemini pipeline, ChromaDB RAG logic
│   ├── requirements.txt     # Python dependencies
│   ├── Procfile             # Render start command
│   ├── runtime.txt          # Python runtime configuration
│   └── policies.txt         # Policy reference content
├── frontend/
│   └── index.html           # Tailwind + Vanilla JS user interface
├── render.yaml              # Render deployment configuration
└── README.md
```

The project is intentionally split into a lightweight static frontend and an API-driven backend. This keeps the UI simple while allowing the backend to own AI orchestration, policy retrieval, validation, and response generation.

## API Endpoints

### `GET /`

Returns basic project metadata and available endpoints.

```bash
curl https://chargebackshield-ai.onrender.com/
```

### `GET /health`

Checks API availability, model readiness, and RAG policy loading status.

```bash
curl https://chargebackshield-ai.onrender.com/health
```

### `POST /analyze-dispute`

Runs the full dispute analysis pipeline.

**Request type:** `multipart/form-data`

| Field     | Type   | Required | Description                                    |
| --------- | ------ | -------- | ---------------------------------------------- |
| `context` | string | Yes      | Merchant explanation of the dispute            |
| `file`    | image  | Yes      | Evidence image, such as JPG, PNG, WEBP, or GIF |

Example:

```bash
curl -X POST "https://chargebackshield-ai.onrender.com/analyze-dispute" \
  -F "context=Customer claims the order was not delivered, but we have delivery proof and chat confirmation." \
  -F "file=@evidence.png"
```

Example response shape:

```json
{
  "status": "Analysis Complete",
  "request_id": "CS-1234567890",
  "total_time_sec": 3.42,
  "verdict": {
    "win_probability": "85%",
    "confidence_level": "HIGH",
    "dispute_category": "Item Not Received",
    "reasoning": "The uploaded evidence supports delivery confirmation and matches key policy criteria.",
    "auto_response_draft": "Dear Bank Team, please find attached evidence confirming successful delivery..."
  },
  "evidence_summary": {
    "document_type": "Delivery Screenshot",
    "delivery_proof": "YES",
    "timestamp_found": "Available",
    "confidence": "HIGH"
  },
  "rag_metadata": {
    "primary_policy": "Item Not Received (INR Claim)",
    "vector_db": "ChromaDB (cosine similarity)"
  }
}
```

### `GET /rag-policies`

Lists the dispute policies currently loaded into ChromaDB.

```bash
curl https://chargebackshield-ai.onrender.com/rag-policies
```

### `GET /metrics`

Returns demo-friendly system metrics.

```bash
curl https://chargebackshield-ai.onrender.com/metrics
```

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd chargeback-ai
```

### 2. Create a virtual environment

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file inside the `backend/` directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

### 5. Run the FastAPI backend locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

### 6. Run the frontend locally

The frontend is a static HTML file. You can open it directly in a browser:

```text
frontend/index.html
```

For local API testing, update the `BACKEND_URL` constant in `frontend/index.html` to:

```js
const BACKEND_URL = "http://localhost:8000";
```

## Deployment

### Backend on Render

The backend is configured for Render deployment using `render.yaml`.

Render settings:

```yaml
rootDir: backend
buildCommand: pip install -r requirements.txt
startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Required environment variable:

```text
GEMINI_API_KEY
```

Production backend:

```text
https://chargebackshield-ai.onrender.com
```

### Frontend on Vercel

The frontend is deployed as a static site from the `frontend/` directory.

Production frontend:

```text
https://chargebackshield-ai.vercel.app
```

Before deploying, confirm the frontend points to the production backend:

```js
const BACKEND_URL = "https://chargebackshield-ai.onrender.com";
```

## Future Improvements

- Add merchant authentication and saved dispute history
- Store generated responses and evidence metadata in a database
- Add support for PDF evidence and multiple file uploads
- Expand the policy knowledge base by card network and payment provider
- Add confidence calibration using historical dispute outcomes
- Generate downloadable PDF response packets for bank submission
- Add audit logs for compliance and dispute operations teams
- Introduce role-based dashboards for support, risk, and operations teams

## Why This Project Matters

Chargebacks are expensive, time-sensitive, and operationally heavy for fintech merchants. A strong response often depends on quickly connecting messy evidence, payment policy, and bank-facing documentation.

ChargebackShield AI demonstrates how multimodal AI and RAG can support this workflow by turning unstructured evidence into a structured dispute package. It is relevant for fintech, payments, risk operations, merchant support, and applied AI teams working on high-volume financial workflows.

## Summary

ChargebackShield AI is a practical AI system that combines:

- Multimodal evidence understanding
- Retrieval-augmented policy matching
- Structured dispute reasoning
- Production-style FastAPI endpoints
- A clean frontend deployed on Vercel
- A backend deployed on Render

It is designed to show how AI can move beyond chat interfaces and support real decision workflows in fintech operations.
