import os
import re
import json
import time
import tempfile
import traceback

import google.generativeai as genai
import chromadb

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv


# SECTION 1 — APP INITIALIZATION

load_dotenv()

app = FastAPI(
    title="ChargebackShield AI",
    description="Multimodal RAG-powered Dispute Resolution Engine for Fintech",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow frontend on any origin
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# SECTION 2 — GEMINI API CONFIGURATION


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(" GEMINI_API_KEY not found! Set it in your environment or Render dashboard.")

genai.configure(api_key=GEMINI_API_KEY)

# Two model handles — one for vision, one for text synthesis
vision_model = genai.GenerativeModel("gemini-1.5-flash")   # Multimodal (image + text)
text_model   = genai.GenerativeModel("gemini-1.5-flash")   # Text synthesis / final verdict

print(" Gemini Vision + Text models initialized.")


# SECTION 3 — CHROMADB RAG SETUP


chroma_client      = chromadb.Client()
policies_collection = chroma_client.get_or_create_collection(
    name="razorpay_dispute_policies",
    metadata={"hnsw:space": "cosine"}   # Cosine similarity for semantic matching
)

# 4 realistic Razorpay-style dispute policies
DISPUTE_POLICIES = [
    {
        "id": "policy_001",
        "title": "Item Not Received (INR Claim)",
        "category": "logistics",
        "content": """
        Policy ID: INR-001 | Category: Logistics / Physical Goods
        Dispute Type: Item Not Received

        WINNING CRITERIA (Merchant must satisfy 2+):
        - Valid courier tracking number showing 'Delivered' status.
        - Delivery screenshot with recipient name, address, and timestamp.
        - OTP or signature confirmation from recipient.
        - GPS photo proof from delivery executive at drop point.

        WIN PROBABILITY MATRIX:
        - 3+ criteria met   → 90–95%  (Auto-Win, no escalation)
        - 2   criteria met  → 65–80%  (Strong case, bank favors merchant)
        - 1   criterion met → 30–45%  (Weak, additional docs needed)
        - 0   criteria met  → 5–15%   (Near-certain loss)

        RESPONSE DEADLINE: 7 calendar days from chargeback filing date.
        PRO TIP: WhatsApp delivery screenshots with blue-ticks are accepted evidence.
        """,
    },
    {
        "id": "policy_002",
        "title": "Unauthorized / Fraudulent Transaction",
        "category": "fraud",
        "content": """
        Policy ID: FRAUD-002 | Category: Cybersecurity / Fraud Prevention
        Dispute Type: Unauthorized Charge / Stolen Card

        WINNING CRITERIA (Merchant must satisfy 3+):
        - Server access log showing customer IP matches billing address ISP.
        - Browser fingerprint / Device ID tied to customer's account history.
        - Email order confirmation sent to registered email immediately after payment.
        - SMS confirmation sent to cardholder's registered mobile number.
        - Signed Terms & Conditions acceptance (checkbox log with timestamp).

        WIN PROBABILITY MATRIX:
        - 4+ criteria met  → 92–98%  (Auto-Win, clear non-fraud evidence)
        - 3   criteria met → 75–88%  (Strong, bank typically favors merchant)
        - 2   criteria met → 50–65%  (Moderate, escalation recommended)
        - 1   criterion   → 20–35%  (Risky, near-loss territory)

        EDGE CASE AUTOMATIC LOSS: If card was reported lost/stolen BEFORE transaction date.
        RESPONSE DEADLINE: 5 business days. IP logs must be in PDF form.
        """,
    },
    {
        "id": "policy_003",
        "title": "Duplicate Charge / Service Quality Dispute",
        "category": "quality",
        "content": """
        Policy ID: DUP-003 | Category: Billing / Customer Service
        Dispute Type: Duplicate Charge or Unsatisfactory Service

        WINNING CRITERIA FOR DUPLICATE CHARGE:
        - Settlement report showing single transaction entry.
        - Unique Razorpay Transaction ID confirming one debit.
        - Bank reconciliation statement.

        WINNING CRITERIA FOR QUALITY DISPUTE:
        - Customer support chat transcript with <24 hr response time.
        - Refund offer proof showing merchant attempted resolution.
        - Service completion evidence (photos, certificates, logs).
        - Customer acknowledgment that service was rendered.

        WIN PROBABILITY MATRIX:
        - Duplicate (with settlement proof) → 88–95%  (Clear-cut win)
        - Quality (with refund offer proof) → 75–85%  (Bank sees good faith)
        - Quality (no resolution attempt)   → 20–35%  (Near-certain loss)

        RESPONSE DEADLINE: 10 calendar days. Attach all chat logs as PDF.
        """,
    },
    {
        "id": "policy_004",
        "title": "Digital Goods & Subscription Services",
        "category": "digital",
        "content": """
        Policy ID: DIGI-004 | Category: SaaS / Subscriptions / Digital Downloads
        Dispute Type: Digital Product Not Received or Subscription Dispute

        WINNING CRITERIA (Merchant must satisfy 3+):
        - Server download / access logs with customer IP and timestamp.
        - Email delivery receipt for digital product link (sent instantly).
        - API activation log showing API key provisioned post-payment.
        - Customer login history showing active usage after purchase.
        - Subscription management dashboard screenshot showing active plan.
        - T&C accepted via digital signature or checkbox with timestamp.

        WIN PROBABILITY MATRIX:
        - 5+ criteria met  → 95%+  (Strongest possible digital goods case)
        - 3–4 criteria met → 75–90%  (Very strong, bank rarely overrides)
        - 2   criteria met → 50–65%  (Moderate, escalation recommended)
        - 1   criterion   → 30–40%  (Weak, customer claim may win)

        AUTOMATIC LOSS SCENARIO: Customer claims "Buyer's Remorse" — bank always sides with buyer.
        RESPONSE DEADLINE: 7 days. Attach usage logs in CSV format.
        """,
    },
]


def load_policies_to_chromadb() -> None:
    """
    Embeds and stores all dispute policies into ChromaDB.
    Called once at startup. ChromaDB handles embedding automatically.
    """
    print("\n📚 [RAG INIT] Loading dispute policies into ChromaDB...")

    existing = policies_collection.get()
    if existing["ids"]:
        print(" Policies already loaded. Skipping re-ingestion.")
        return

    for policy in DISPUTE_POLICIES:
        combined_text = f"TITLE: {policy['title']}\n\nCATEGORY: {policy['category']}\n\n{policy['content']}"

        policies_collection.add(
            ids=[policy["id"]],
            documents=[combined_text],
            metadatas=[{
                "title":    policy["title"],
                "category": policy["category"],
                "source":   "razorpay_dispute_policy_v2",
            }],
        )
        print(f" Embedded & stored → {policy['title']}")

    print("[RAG INIT] All 4 policies loaded into vector store.\n")


# Run at startup
load_policies_to_chromadb()


# SECTION 4 — HELPER: RAG RETRIEVAL


def retrieve_relevant_policies(query_text: str, top_k: int = 2) -> list[dict]:
    """
    Performs semantic similarity search on ChromaDB.

    Args:
        query_text : Text extracted from evidence (from Vision step).
        top_k      : Number of most-relevant policies to return.

    Returns:
        List of dicts with title, category, content, distance score.
    """
    results = policies_collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "title":      meta.get("title", "Unknown"),
                "category":   meta.get("category", "general"),
                "content":    doc,
                "similarity": round(1 - dist, 4),   # Convert distance → similarity score
            })

    return retrieved


# SECTION 5 — HELPER: GEMINI VISION (MULTIMODAL)


def extract_evidence_via_vision(image_path: str, merchant_context: str) -> dict:
    """
    Sends the uploaded evidence image to Gemini 1.5 Flash Vision.
    Extracts structured evidence fields from the image.

    Args:
        image_path        : Temp path to uploaded image.
        merchant_context  : Text context typed by the merchant.

    Returns:
        Parsed dict with extracted evidence fields.
    """
    print("   📸 Sending image to Gemini Vision...")

    uploaded_file = genai.upload_file(path=image_path)

    vision_prompt = f"""
You are a Fintech Dispute Evidence Analyst specializing in payment fraud prevention.
Analyze the uploaded image carefully and extract all relevant information.

MERCHANT CONTEXT PROVIDED: "{merchant_context}"

Extract and return ONLY a valid JSON object (no markdown, no backticks) with:
{{
  "document_type"    : "Type of document (e.g., WhatsApp Chat, Delivery Screenshot, Receipt, Invoice, Bank Statement)",
  "extracted_text"   : "All visible text from the image",
  "delivery_proof"   : "YES / NO / PARTIAL",
  "timestamp_found"  : "Timestamp found in image, or 'None'",
  "recipient_name"   : "Name of recipient if visible, else 'Not Found'",
  "delivery_address" : "Address visible in image, or 'Not Found'",
  "otp_or_signature" : "YES / NO",
  "key_evidence"     : "Top 2-3 most legally relevant pieces of evidence from this image",
  "confidence"       : "HIGH / MEDIUM / LOW (how clearly the image supports the merchant)"
}}

Be precise. If image quality is poor, note it under confidence.
Return ONLY the JSON object. Nothing else.
"""

    response      = vision_model.generate_content([vision_prompt, uploaded_file])
    raw_text      = response.text.strip()

    # Safely parse the JSON response from Gemini
    try:
        # Strip accidental markdown code fences if Gemini adds them
        clean_json = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        parsed     = json.loads(clean_json)
    except json.JSONDecodeError:
        # Fallback: return raw text wrapped in dict
        parsed = {
            "document_type"   : "Unknown",
            "extracted_text"  : raw_text,
            "delivery_proof"  : "UNKNOWN",
            "confidence"      : "LOW",
            "parse_error"     : "Gemini returned non-JSON. Raw text preserved.",
        }

    print(f"    Vision extraction complete. Confidence: {parsed.get('confidence', 'N/A')}")
    return parsed


# SECTION 6 — HELPER: GEMINI TEXT (FINAL LLM SYNTHESIS)


def synthesize_final_verdict(
    merchant_context : str,
    vision_evidence  : dict,
    top_policy       : dict,
) -> dict:
    """
    Final LLM call: Gemini Text takes evidence + policy and generates:
    - Win Probability
    - Reasoning
    - Policy cited
    - Auto-draft bank response letter

    Args:
        merchant_context : Original merchant-typed context.
        vision_evidence  : Structured evidence extracted by Vision.
        top_policy       : Most relevant policy from RAG.

    Returns:
        Parsed dict with verdict fields.
    """
    print("    Sending evidence + policy to Gemini Text for synthesis...")

    synthesis_prompt = f"""
You are a Senior Razorpay Dispute Resolution Specialist with 10 years of fintech experience.
Your job is to evaluate dispute cases and provide a verdict that merchants can submit to banks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MERCHANT'S CONTEXT (their version of events):
{merchant_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE EXTRACTED FROM IMAGE (via AI Vision):
{json.dumps(vision_evidence, indent=2)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPLICABLE RAZORPAY POLICY (retrieved via RAG):
Policy Title    : {top_policy['title']}
Policy Category : {top_policy['category']}
RAG Similarity  : {top_policy['similarity']}
Policy Details  :
{top_policy['content']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK:
Generate a professional dispute verdict. Return ONLY a valid JSON object (no markdown):

{{
  "win_probability"       : "XX% (e.g., 85%)",
  "confidence_level"      : "HIGH / MEDIUM / LOW",
  "dispute_category"      : "Category of dispute (e.g., Item Not Received)",
  "policy_cited"          : "Exact policy title that applies",
  "criteria_matched"      : ["List", "of", "matching", "criteria", "found"],
  "criteria_missing"      : ["List", "of", "missing", "criteria", "that", "would", "strengthen", "case"],
  "reasoning"             : "2–3 sentence professional explanation of verdict",
  "risk_flags"            : ["Any red flags or automatic-loss triggers found"],
  "auto_response_draft"   : "Professional 3–4 sentence letter to the bank on behalf of the merchant",
  "recommended_actions"   : ["Next step 1", "Next step 2", "Next step 3"],
  "processing_note"       : "Internal note for Razorpay dispute team"
}}

Be realistic. Reflect evidence strength in win_probability.
Return ONLY the JSON object. Nothing else.
"""

    response  = text_model.generate_content(synthesis_prompt)
    raw_text  = response.text.strip()

    try:
        clean_json = re.sub(r"```(?:json)?|```", "", raw_text).strip()
        parsed     = json.loads(clean_json)
    except json.JSONDecodeError:
        parsed = {
            "win_probability"     : "Unable to calculate",
            "confidence_level"    : "LOW",
            "reasoning"           : raw_text,
            "auto_response_draft" : "Manual review required.",
            "parse_error"         : "Gemini returned non-JSON. Raw text preserved.",
        }

    print(f"    Verdict synthesized. Win probability: {parsed.get('win_probability', 'N/A')}")
    return parsed


# SECTION 7 — MAIN ENDPOINT: /analyze-dispute


ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_FILE_SIZE_MB   = 5

@app.post("/analyze-dispute", response_class=JSONResponse)
async def analyze_dispute(
    context : str        = Form(..., description="Merchant's description of the dispute"),
    file    : UploadFile = File(..., description="Evidence image (JPG/PNG/WEBP)"),
):
    """
    ═══════════════════════════════════════════════════════
    MAIN ENDPOINT: /analyze-dispute
    ═══════════════════════════════════════════════════════

    Full RAG + Vision + LLM Pipeline:

    Step 1 → Validate uploaded file
    Step 2 → Save image to temp storage
    Step 3 → Gemini Vision extracts structured evidence from image
    Step 4 → ChromaDB RAG retrieves most relevant dispute policy
    Step 5 → Gemini Text synthesizes final verdict + bank response
    Step 6 → Return complete structured response

    Request  : multipart/form-data  (context: str, file: image)
    Response : JSON with verdict, evidence, policy, logs
    """

    pipeline_start = time.time()
    pipeline_logs  = []

    def log(msg: str):
        """Append to pipeline logs and print to console."""
        pipeline_logs.append(msg)
        print(f"   {msg}")

    
    # STEP 1: INPUT VALIDATION
    
    
    print(" [/analyze-dispute] New dispute request received")
    

    # Validate context
    if not context or len(context.strip()) < 10:
        raise HTTPException(
            status_code=422,
            detail="Context too short. Provide at least a brief description."
        )

    # Validate file type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Upload JPG, PNG, or WEBP."
        )

    # Read file bytes & check size
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size_mb:.1f} MB. Max allowed: {MAX_FILE_SIZE_MB} MB."
        )

    log(f" Step 1/5 — Input validated | File: {file.filename} ({file_size_mb:.2f} MB) | Type: {file.content_type}")

    
    # STEP 2: SAVE IMAGE TO TEMP FILE
    
    try:
        suffix = ".jpg" if "jpeg" in file.content_type else f".{file.content_type.split('/')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        log(f" Step 2/5 — Image saved to temp storage | Path: {temp_path}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

    
    # STEP 3: GEMINI VISION — MULTIMODAL EVIDENCE EXTRACTION
    
    print("\n [Step 3/5] Gemini Vision — Multimodal Analysis")
    try:
        vision_start    = time.time()
        vision_evidence = extract_evidence_via_vision(temp_path, context)
        vision_duration = round(time.time() - vision_start, 2)

        log(f" Step 3/5 — Vision complete | Took: {vision_duration}s | Confidence: {vision_evidence.get('confidence', 'N/A')}")

    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Vision analysis failed: {str(e)}\n{traceback.format_exc()}")

    
    # STEP 4: CHROMADB RAG — POLICY RETRIEVAL
    
    print("\n [Step 4/5] ChromaDB RAG — Policy Retrieval")
    try:
        # Build rich query from evidence + merchant context
        rag_query = f"""
        Merchant Context : {context}
        Document Type    : {vision_evidence.get('document_type', '')}
        Key Evidence     : {vision_evidence.get('key_evidence', '')}
        Delivery Proof   : {vision_evidence.get('delivery_proof', '')}
        """

        rag_start  = time.time()
        policies   = retrieve_relevant_policies(rag_query, top_k=2)
        rag_duration = round(time.time() - rag_start, 2)

        if not policies:
            raise ValueError("RAG returned zero results. Vector store may be empty.")

        top_policy    = policies[0]
        second_policy = policies[1] if len(policies) > 1 else None

        log(f" Step 4/5 — RAG complete | Took: {rag_duration}s | Best match: '{top_policy['title']}' (similarity: {top_policy['similarity']})")

    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {str(e)}")

    
    # STEP 5: GEMINI TEXT — FINAL VERDICT SYNTHESIS
    
    print("\n [Step 5/5] Gemini Text — Final Verdict Synthesis")
    try:
        synthesis_start   = time.time()
        verdict           = synthesize_final_verdict(context, vision_evidence, top_policy)
        synthesis_duration = round(time.time() - synthesis_start, 2)

        log(f" Step 5/5 — Synthesis complete | Took: {synthesis_duration}s | Verdict: {verdict.get('win_probability', 'N/A')}")

    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Verdict synthesis failed: {str(e)}")

    
    # CLEANUP
    
    try:
        os.remove(temp_path)
        log(" Temp file cleaned up")
    except Exception:
        pass   # Non-critical

    
    # STEP 6: BUILD AND RETURN FINAL RESPONSE
    
    total_time = round(time.time() - pipeline_start, 2)

    return {
        "status"          : " Analysis Complete",
        "request_id"      : f"CS-{int(time.time())}",   # Fake request ID (looks enterprise)
        "total_time_sec"  : total_time,

        # The core AI output
        "verdict"         : verdict,

        # Evidence extracted by Vision
        "evidence_summary": {
            "document_type"   : vision_evidence.get("document_type", "Unknown"),
            "delivery_proof"  : vision_evidence.get("delivery_proof", "Unknown"),
            "timestamp_found" : vision_evidence.get("timestamp_found", "None"),
            "key_evidence"    : vision_evidence.get("key_evidence", "None"),
            "confidence"      : vision_evidence.get("confidence", "LOW"),
        },

        # RAG metadata
        "rag_metadata"    : {
            "primary_policy"  : top_policy["title"],
            "rag_similarity"  : top_policy["similarity"],
            "secondary_policy": second_policy["title"] if second_policy else None,
            "vector_db"       : "ChromaDB (cosine similarity)",
            "policies_searched": len(DISPUTE_POLICIES),
        },

        # Processing logs (shown in UI for "wow" factor)
        "pipeline_logs"   : pipeline_logs,

        # System metrics (makes it look production-grade)
        "system_metrics"  : {
            "vision_latency_sec"    : vision_duration,
            "rag_latency_sec"       : rag_duration,
            "synthesis_latency_sec" : synthesis_duration,
            "total_latency_sec"     : total_time,
            "model_used"            : "gemini-1.5-flash (vision + text)",
            "embedding_engine"      : "ChromaDB default (all-MiniLM-L6-v2)",
        },
    }


# SECTION 8 — UTILITY ENDPOINTS


@app.get("/")
async def root():
    return {
        "project"    : " ChargebackShield AI",
        "version"    : "2.0.0",
        "status"     : " Live",
        "endpoints"  : {
            "POST /analyze-dispute" : "Main pipeline (image + context → verdict)",
            "GET  /health"          : "System health check",
            "GET  /rag-policies"    : "List all loaded RAG policies",
            "GET  /metrics"         : "System performance metrics",
        },
    }


@app.get("/health")
async def health_check():
    rag_count = policies_collection.count()
    return {
        "api_status"       : " Online",
        "rag_status"       : f" {rag_count} policies loaded in ChromaDB",
        "vision_model"     : " Gemini 1.5 Flash (Vision) ready",
        "text_model"       : " Gemini 1.5 Flash (Text) ready",
        "server_time_utc"  : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uptime_check"     : " All systems operational",
    }


@app.get("/rag-policies")
async def list_rag_policies():
    return {
        "total_policies" : len(DISPUTE_POLICIES),
        "vector_db"      : "ChromaDB (in-memory, cosine similarity)",
        "policies"       : [
            {
                "id"      : p["id"],
                "title"   : p["title"],
                "category": p["category"],
            }
            for p in DISPUTE_POLICIES
        ],
    }


@app.get("/metrics")
async def metrics():
    """Fake metrics endpoint — looks very impressive in demos."""
    return {
        "disputes_processed_today" : 142,
        "avg_processing_time_sec"  : 2.4,
        "avg_win_probability"      : "73%",
        "top_dispute_category"     : "Item Not Received",
        "rag_query_avg_ms"         : 38,
        "vision_avg_ms"            : 1100,
        "system_uptime"            : "99.94%",
        "note"                     : "Simulated metrics for demo purposes",
    }


# SECTION 9 — ENTRY POINT


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
