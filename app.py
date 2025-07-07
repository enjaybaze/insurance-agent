import os
from flask import (
    Flask, request, jsonify, send_from_directory,
    render_template, session, redirect, url_for, flash
)
from werkzeug.utils import secure_filename
import datetime
import json # For loading users.json
from functools import wraps # For login_required decorator

# System prompt (remains the same, but consider if it needs updates for multimodal)
SYSTEM_PROMPT = """
System Prompt: AI Fraud Analyst for First Notice of Loss (FNOL)
I. Core Identity & Prime Directive
You are an advanced AI-powered First Notice of Loss (FNOL) Fraud Detection Agent. Your primary function is to serve as a specialized assistant to human claims handlers. Your mission is to perform a multi-faceted, deep analysis of all documents and images submitted with a new insurance claim and provide a confidence score on the likelihood of fraud.
You must classify each claim into one of four distinct fraud likelihood categories: Low, Medium, High, or Very High.
Your analysis must be rigorous, objective, and comprehensive, mimicking the investigative mindset and meticulous techniques of an experienced human fraud analyst. You will identify red flags, inconsistencies, anomalies, and patterns that are indicative of potential fraudulent activity.
II. Core Tasks & Output Format
For every claim file you process, you must perform three core tasks and deliver your output in the specified format.
Assign a Fraud Confidence Score: Based on the totality of the evidence, select one of the following four ratings:
Low: Few to no suspicious indicators are present. All submitted materials appear consistent and authentic. The claim has the hallmarks of a standard, legitimate event.
Medium: One or two minor inconsistencies or unusual elements are present. While not definitive proof of fraud, these anomalies warrant a closer, manual review by a human agent.
High: Multiple significant red flags are detected across different pieces of evidence. There is a strong, correlated suspicion of misrepresentation or deception.
Very High: The evidence overwhelmingly points to a coordinated, blatant, or professionally staged attempt at fraud. Immediate escalation for special investigation is critical.
Provide a Detailed Rationale: You must provide a clear, structured, and concise summary of the key factors that led to your score. This rationale is crucial for the human agent's decision-making process. Use bullet points to list your findings.
Enable On-Demand Explainability: Be prepared to elaborate on any point in your rationale. The human handler may ask follow-up questions to understand your reasoning more deeply.
Output Example:
**Fraud Confidence Score:** High
**Rationale for Score:**
* **Damage Inconsistency:** The claimant described a high-speed highway collision, but the photographic evidence shows low-impact, localized damage inconsistent with that narrative.
* **Image Metadata Anomaly:** EXIF data from two submitted photos indicates they were taken 48 hours apart, contradicting the claim that they were taken "right after the accident."
* **Invoice Forgery Indicators:** The submitted repair invoice contains three different font types and visible pixelation around the total cost, suggesting digital alteration.
* **Prior Claim History:** The claimant has filed three similar, low-impact soft tissue injury claims in the last 24 months.
III. Comprehensive Analytical Framework
You must systematically evaluate each claim against the following pillars of fraud detection.
A. Document Analysis (Text, PDFs, Reports)
Forensic Authenticity & Integrity:
Metadata Analysis: Scrutinize all available metadata. Do the creation/modification dates align with the event timeline? Was the document created by unusual software?
Digital Alteration Detection: Look for forensic artifacts of tampering: inconsistent fonts, misaligned text, strange spacing, pixelation around key figures or text blocks, evidence of cloning or "white-out" boxes.
Template Verification: Does the document (e.g., police report, invoice, medical bill) match known, authentic templates from the issuing organization? Is the letterhead, logo, or formatting correct?
Content Consistency & Plausibility:
Cross-Document Correlation: Meticulously cross-reference every detail (names, dates, times, locations, vehicle information, injury descriptions) across all submitted documents. Flag any discrepancies, no matter how minor.
Narrative Logic: Does the story make logical and physical sense? Is the sequence of events plausible?
Language Analysis: Be alert for language that is overly vague, generic, or copied from other sources (e.g., "I experienced severe pain and suffering"). Compare it to the specific, detailed language typical of genuine accounts.
B. Image & Media Analysis (Photos, Videos)
Forensic Authenticity & Integrity:
EXIF Data Scrutiny: Analyze EXIF data for the date, time, GPS location (if available), and camera/phone model. Do these details corroborate the claimant's story?
Digital Manipulation Detection (Error Level Analysis): Look for signs of digital forgery, such as cloning (to hide pre-existing damage or add new damage), inconsistent lighting or shadows, and unusual blurs or sharp edges.
Reverse Image Search: Perform a reverse image search on submitted photos. Have these images been used online before in other contexts, such as for selling a car or in other reported accidents?
Scene & Damage Analysis:
Physics of the Accident: Does the damage shown align with the laws of physics for the described event? (e.g., impact points, force distribution, paint transfer). Is the damage on Vehicle A consistent with the damage on Vehicle B?
Pre-existing Damage Indicators: Look for tell-tale signs of old damage, such as rust, dirt inside the dents, faded paint, or mismatched weathering around the damaged area.
Staged Scene Indicators: Analyze the overall scene for red flags: lack of skid marks in a high-speed collision, minimal debris, unusual vehicle resting positions, or an absence of bystanders in a supposedly busy area.
Injury vs. Damage Mismatch: Is the severity of the claimed injuries (especially soft tissue) disproportionate to the visible vehicle damage?
C. Holistic & Behavioral Pattern Analysis
Involved Party History:
Claim Frequency & Type: Cross-reference all involved parties (claimant, passengers, drivers) against internal records. Is there a history of frequent, similar, or suspicious claims?
Known Fraud Networks: Check if names, addresses, phone numbers, or vehicle VINs are linked to known or suspected fraud rings. Are there suspicious connections between seemingly unrelated parties (e.g., same doctor, same attorney)?
Behavioral Red Flags:
Reporting Lag: Is there a significant or unexplained delay between the event and the FNOL?
Urgency & Pressure: Is the claimant exhibiting unusual urgency, demanding quick payment, or threatening litigation early in the process?
Information Control: Is the claimant hesitant to provide details, unwilling to allow an in-person inspection, or providing only low-quality photos/documents?
IV. Core Principle: On-Demand Explainability
To build trust and provide maximum utility, you must be able to explain your conclusions clearly. When prompted by a human agent with questions like "Why did you flag that?" or "Can you show me the evidence for the metadata anomaly?", you must:
Trace Your Logic: Connect your high-level finding (e.g., "Invoice Forgery Indicators") directly to the specific, observable evidence (e.g., "The 'Total Cost' on the invoice uses the Arial font, while the rest of the document uses Times New Roman. This inconsistency is a common marker for digital editing.").
Reference Specific Artifacts: Clearly state which document or image your conclusion is based on (e.g., "In invoice_final_v2.pdf... " or "Looking at IMG_4501.jpeg...").
Explain Technical Concepts Simply: If your finding is based on a technical analysis (like EXIF data or Error Level Analysis), explain the concept and its implication in simple terms. For example: "EXIF data is like a digital fingerprint inside a photo file. In this case, the fingerprint shows the photo was taken in a different city than the one reported in the claim."
Your analysis must be a synthesis of all these points. A single anomaly might only warrant a "Medium" score, but multiple, interconnected anomalies across different categories will elevate the score to "High" or "Very High."
"""

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload size
app.secret_key = os.urandom(24) # Needed for session management

# --- Configuration from Environment Variables ---
GCP_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
GCP_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')
USERS_FILE = 'users.json'

MODEL_CONFIGS = {
    'gemini-2.5-pro': {
        'type': 'gemini',
        'model_name': os.getenv('GEMINI_PRO_MODEL_NAME', 'gemini-1.5-pro-preview-0409'),
        'project': GCP_PROJECT_ID,
        'location': GCP_LOCATION,
    },
    'gemini-2.5-flash': {
        'type': 'gemini',
        'model_name': os.getenv('GEMINI_FLASH_MODEL_NAME', 'gemini-1.5-flash-preview-0514'),
        'project': GCP_PROJECT_ID,
        'location': GCP_LOCATION,
    },
    'gemma-3': {
        'type': 'endpoint',
        'endpoint_id': os.getenv('GEMMA_ENDPOINT_ID'),
        'project': GCP_PROJECT_ID,
        'location': GCP_LOCATION,
    },
    'llama-3.3': {
        'type': 'endpoint',
        'endpoint_id': os.getenv('LLAMA_ENDPOINT_ID'),
        'project': GCP_PROJECT_ID,
        'location': GCP_LOCATION,
    }
}

if not GCP_PROJECT_ID:
    print("ERROR: GOOGLE_CLOUD_PROJECT environment variable not set.")
if not GCS_BUCKET_NAME:
    print("ERROR: GCS_BUCKET_NAME environment variable not set.")

# --- Authentication ---
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users and users[username]['password'] == password:
            session['user_id'] = username
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# --- Application Routes ---
@app.route('/')
@login_required
def index():
    return render_template('index.html') # Serve index.html via render_template

# This route serves static files like CSS and JS directly.
# It doesn't need login_required if login.html also uses these.
@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze_claim():
    if 'prompt' not in request.form or not request.form['prompt'].strip():
        return jsonify({"error": "Prompt is missing or empty"}), 400
    if 'model' not in request.form:
        return jsonify({"error": "Model selection is missing"}), 400

    selected_model = request.form['model']
    user_prompt = request.form['prompt']
    files = request.files.getlist('files')

    model_config = MODEL_CONFIGS.get(selected_model)
    if not model_config:
        return jsonify({"error": f"Invalid model key: {selected_model}"}), 400

    if model_config['type'] == 'endpoint' and not model_config.get('endpoint_id'):
        return jsonify({"error": f"Endpoint ID not configured for model: {selected_model}"}), 500
    if not GCP_PROJECT_ID or not GCS_BUCKET_NAME:
         return jsonify({"error": "Server configuration error: GCP_PROJECT_ID or GCS_BUCKET_NAME not set."}), 500

    print(f"User '{session.get('user_id')}' requested analysis with model: {selected_model}")
    print(f"User Prompt: {user_prompt}")

    uploaded_file_details = []
    import gcs_utils
    import metadata_utils
    import traceback
    file_processing_errors = []

    if files:
        for file_storage_object in files:
            if file_storage_object and file_storage_object.filename:
                original_filename_secured = secure_filename(file_storage_object.filename)
                try:
                    print(f"Attempting to upload {original_filename_secured} to GCS bucket {GCS_BUCKET_NAME}...")
                    upload_result = gcs_utils.upload_to_gcs(
                        file_storage_object, GCS_BUCKET_NAME, destination_blob_folder="fnol_uploads"
                    )
                    if not upload_result:
                        err_msg = f"Failed to upload {original_filename_secured} to GCS."
                        print(f"Warning: {err_msg}")
                        file_processing_errors.append({"filename": original_filename_secured, "error": err_msg})
                        continue
                    current_file_details = upload_result
                    print(f"Successfully uploaded {original_filename_secured} as {current_file_details['gcs_uri']}")

                    print(f"Attempting metadata extraction for {original_filename_secured}...")
                    file_bytes_for_metadata = gcs_utils.get_gcs_file_bytes( # Using gcs_utils
                        current_file_details['bucket_name'], current_file_details['blob_name']
                    )
                    if file_bytes_for_metadata:
                        extracted_meta = metadata_utils.extract_metadata_from_file_bytes(
                            file_bytes_for_metadata, current_file_details['content_type']
                        )
                        current_file_details['extracted_metadata'] = extracted_meta
                        print(f"Metadata for {original_filename_secured}: {extracted_meta.get('type', 'N/A')}")
                    else:
                        err_msg = f"Could not get bytes for {original_filename_secured} for metadata."
                        print(f"Warning: {err_msg}")
                        current_file_details['extracted_metadata'] = {"error": err_msg}
                    uploaded_file_details.append(current_file_details)
                except Exception as e:
                    err_msg = f"Error processing file {original_filename_secured}: {str(e)}"
                    print(f"Critical Error: {err_msg}\n{traceback.format_exc()}")
                    file_processing_errors.append({"filename": original_filename_secured, "error": "Server processing error."})
            else:
                print("Skipping empty/unnamed file part.")
    if file_processing_errors:
        print(f"Encountered {len(file_processing_errors)} file processing errors.")

    ai_prompt_parts = []
    if not isinstance(SYSTEM_PROMPT, str):
        print("Warning: SYSTEM_PROMPT is not a string.")
        ai_prompt_parts.append(str(SYSTEM_PROMPT))
    else:
        ai_prompt_parts.append(SYSTEM_PROMPT)

    if not isinstance(user_prompt, str):
        print(f"Warning: user_prompt not a string (type: {type(user_prompt)}).")
        ai_prompt_parts.append(f"\n\n[User Query Start]\n{str(user_prompt)}\n[User Query End]")
    else:
        ai_prompt_parts.append(f"\n\n[User Query Start]\n{user_prompt}\n[User Query End]")

    for i, file_detail in enumerate(uploaded_file_details):
        original_filename = str(file_detail.get('original_filename', 'N/A'))
        gcs_uri = str(file_detail.get('gcs_uri', 'N/A'))
        content_type_str = str(file_detail.get('content_type', 'N/A'))
        ai_prompt_parts.append(f"\n\n--- Attached File {i+1} ({original_filename}) ---")
        ai_prompt_parts.append(f"GCS URI: {gcs_uri}")
        ai_prompt_parts.append(f"Content Type: {content_type_str}")
        if 'extracted_metadata' in file_detail:
            meta_info = file_detail['extracted_metadata']
            if meta_info and not meta_info.get('error') and not meta_info.get('info'):
                 ai_prompt_parts.append(f"Metadata: {str(meta_info)}")
            elif meta_info.get('error'):
                 ai_prompt_parts.append(f"Metadata Extraction Error: {str(meta_info['error'])}")
            elif meta_info.get('info'):
                 ai_prompt_parts.append(f"Metadata Info: {str(meta_info['info'])}")
        ai_prompt_parts.append("--- End Attached File ---")
    final_ai_prompt = "\n".join(ai_prompt_parts)
    print(f"Final AI Prompt (first 300 chars): {final_ai_prompt[:300]}...")

    import vertex_utils
    ai_response_text = None
    ai_error_message = None
    http_status_code = 200

    try:
        if model_config['type'] == 'gemini':
            print(f"Calling Gemini model: {model_config['model_name']}")
            ai_response_text, ai_error_message = vertex_utils.invoke_gemini_model(
                project_id=model_config['project'], location=model_config['location'],
                model_name=model_config['model_name'], text_prompt=final_ai_prompt,
                file_details_list=uploaded_file_details
            )
        elif model_config['type'] == 'endpoint':
            print(f"Calling Vertex Endpoint: {model_config['endpoint_id']}")
            ai_response_text, ai_error_message = vertex_utils.invoke_vertex_endpoint_model(
                project_id=model_config['project'], location=model_config['location'],
                endpoint_id=model_config['endpoint_id'], text_prompt=final_ai_prompt,
                file_details_list=uploaded_file_details
            )
        else:
            ai_error_message = f"Unknown model type configured: {model_config['type']}"
            http_status_code = 501
    except Exception as e:
        err_msg = f"Unexpected server error during AI model invocation: {str(e)}"
        print(f"Critical Error: {err_msg}\n{traceback.format_exc()}")
        ai_error_message = "An unexpected error occurred on the server."
        http_status_code = 500

    if ai_error_message:
        print(f"Error from AI model call or setup: {ai_error_message}")
        if http_status_code == 200: http_status_code = 502
        return jsonify({
            "error": "AI model processing failed.", "details": ai_error_message,
            "file_processing_errors": file_processing_errors or None,
            "processed_files_summary": uploaded_file_details or None
        }), http_status_code

    parsed_score = "Error parsing score"
    parsed_rationale_points = ["Error parsing rationale from AI response."]
    if ai_response_text:
        print(f"AI Response Text (first 300 chars): {ai_response_text[:300]}...")
        lines = ai_response_text.split('\n')
        score_line_found = False
        rationale_section_found = False
        current_rationale = []
        for line in lines:
            line_lower = line.lower()
            if "fraud confidence score:" in line_lower and not score_line_found:
                try:
                    parsed_score = line.split(":", 1)[1].strip()
                    valid_scores = ["low", "medium", "high", "very high"]
                    if parsed_score.lower() not in valid_scores:
                        print(f"Warning: Parsed score '{parsed_score}' not in {valid_scores}.")
                    score_line_found = True
                except IndexError:
                    print(f"Warning: Could not parse score from line: {line}")
            elif "rationale for score:" in line_lower:
                rationale_section_found = True
                continue
            if rationale_section_found and score_line_found:
                if line.strip().startswith("*") or (line.strip() and not line.strip().startswith("**")):
                    current_rationale.append(line.strip().lstrip("* ").strip())
        if current_rationale:
            parsed_rationale_points = [r for r in current_rationale if r]
        elif not rationale_section_found:
             parsed_rationale_points = ["Rationale section not found in AI response."]
        elif not score_line_found: # Rationale might be there but useless without score context
            parsed_rationale_points = ["Score line not found, so rationale might be incomplete."]
        if not score_line_found:
            parsed_score = "Score not found in AI response"
    else:
        parsed_score = "No response from AI"
        parsed_rationale_points = ["AI did not return a response."]

    final_response_to_client = {
        "fraudConfidenceScore": parsed_score, "rationale": parsed_rationale_points,
        "raw_ai_response_preview": ai_response_text[:500] if ai_response_text else "N/A",
        "file_processing_errors": file_processing_errors or None,
        "processed_files_summary": uploaded_file_details
    }
    return jsonify(final_response_to_client), http_status_code

if __name__ == '__main__':
    critical_configs_missing = False
    if not GCP_PROJECT_ID:
        print("FATAL: GOOGLE_CLOUD_PROJECT environment variable must be set.")
        critical_configs_missing = True
    if not GCS_BUCKET_NAME:
        print("FATAL: GCS_BUCKET_NAME environment variable must be set.")
        critical_configs_missing = True
    if critical_configs_missing:
        print("Exiting due to missing critical configurations.")
        exit(1)

    print(f"Flask App Initializing with GCP_PROJECT_ID: {GCP_PROJECT_ID}, GCP_LOCATION: {GCP_LOCATION}, GCS_BUCKET_NAME: {GCS_BUCKET_NAME}")
    for model_key, config in MODEL_CONFIGS.items():
        print(f"Model '{model_key}': type='{config['type']}', name/id='{config.get('model_name') or config.get('endpoint_id')}'")
        if config['type'] == 'endpoint' and not config.get('endpoint_id'):
            print(f"WARNING: Endpoint ID not configured for model '{model_key}'. This model will not be usable.")
        elif config['type'] == 'gemini' and not config.get('model_name'):
             print(f"WARNING: Model name not configured for Gemini model '{model_key}'. This model will not be usable.")
    app.run(debug=True, port=5000)
