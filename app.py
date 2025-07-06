import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import datetime

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

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload size

# --- Configuration from Environment Variables ---
GCP_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
GCP_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1') # Default location
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')

# Model Configurations
# The keys here (e.g., 'gemini-2.5-pro') should match the 'value' attributes in index.html's model select options.
MODEL_CONFIGS = {
    'gemini-2.5-pro': {
        'type': 'gemini', # Native Vertex AI Gemini model
        'model_name': os.getenv('GEMINI_PRO_MODEL_NAME', 'gemini-1.5-pro-preview-0409'), # Default if not set
        'project': GCP_PROJECT_ID,
        'location': GCP_LOCATION,
    },
    'gemini-2.5-flash': {
        'type': 'gemini',
        'model_name': os.getenv('GEMINI_FLASH_MODEL_NAME', 'gemini-1.5-flash-preview-0514'),
        'project': GCP_PROJECT_ID,
        'location': GCP_LOCATION,
    },
    'gemma-3': { # Assuming Gemma 3 refers to a custom deployed Gemma model
        'type': 'endpoint', # Vertex AI Endpoint
        'endpoint_id': os.getenv('GEMMA_ENDPOINT_ID'), # Full endpoint ID: projects/.../locations/.../endpoints/...
        'project': GCP_PROJECT_ID, # Extracted if endpoint_id is full name, or use this
        'location': GCP_LOCATION, # Extracted if endpoint_id is full name, or use this
    },
    'llama-3.3': { # Assuming Llama 3.3 refers to a custom deployed Llama model
        'type': 'endpoint',
        'endpoint_id': os.getenv('LLAMA_ENDPOINT_ID'),
        'project': GCP_PROJECT_ID,
        'location': GCP_LOCATION,
    }
}

# Validate essential configurations
if not GCP_PROJECT_ID:
    print("ERROR: GOOGLE_CLOUD_PROJECT environment variable not set.")
    # Potentially exit or raise an error for critical missing config
if not GCS_BUCKET_NAME:
    print("ERROR: GCS_BUCKET_NAME environment variable not set.")

# (Upload folder is no longer needed for persistent user file storage, GCS will be used)
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Remove if not used for temp processing

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # This will serve style.css and script.js from the static folder
    # if they are requested from the root path.
    if filename in ['style.css', 'script.js']:
        return send_from_directory(app.static_folder, filename)
    return send_from_directory('.', filename)


@app.route('/api/analyze', methods=['POST'])
def analyze_claim():
    if 'prompt' not in request.form or not request.form['prompt'].strip():
        return jsonify({"error": "Prompt is missing or empty"}), 400
    if 'model' not in request.form:
        return jsonify({"error": "Model selection is missing"}), 400

    selected_model = request.form['model'] # Correctly get the model key from the form
    user_prompt = request.form['prompt']
    files = request.files.getlist('files') # Get list of files

    model_config = MODEL_CONFIGS.get(selected_model) # Use the correct variable 'selected_model'
    if not model_config:
        return jsonify({"error": f"Invalid model key: {selected_model}"}), 400 # Use selected_model in error msg too

    # Further validation for endpoint models
    if model_config['type'] == 'endpoint' and not model_config.get('endpoint_id'):
        return jsonify({"error": f"Endpoint ID not configured for model: {selected_model}"}), 500 # Changed selected_model_key to selected_model
    if not GCP_PROJECT_ID or not GCS_BUCKET_NAME:
         return jsonify({"error": "Server configuration error: GCP_PROJECT_ID or GCS_BUCKET_NAME not set."}), 500


    print(f"Received request for model: {selected_model} (Config: {model_config})") # Use selected_model for logging
    print(f"User Prompt: {user_prompt}")

    uploaded_file_details = [] # Will store dicts with GCS URI, original_filename, content_type, etc.

    if not GCS_BUCKET_NAME: # Already checked at startup, but good for request context too
        return jsonify({"error": "GCS_BUCKET_NAME is not configured on the server."}), 500

    # Import GCS and Metadata utilities
    import gcs_utils
    import metadata_utils
    import traceback # For logging detailed tracebacks

    file_processing_errors = []

    if files:
        for file_storage_object in files:
            if file_storage_object and file_storage_object.filename:
                original_filename_secured = secure_filename(file_storage_object.filename)
                try:
                    # 1. Upload to GCS
                    print(f"Attempting to upload {original_filename_secured} to GCS bucket {GCS_BUCKET_NAME}...")
                    upload_result = gcs_utils.upload_to_gcs(
                        file_storage_object,
                        GCS_BUCKET_NAME,
                        destination_blob_folder="fnol_uploads"
                    )
                    if not upload_result:
                        err_msg = f"Failed to upload {original_filename_secured} to GCS."
                        print(f"Warning: {err_msg}")
                        file_processing_errors.append({"filename": original_filename_secured, "error": err_msg})
                        continue

                    current_file_details = upload_result
                    print(f"Successfully uploaded {original_filename_secured} as {current_file_details['gcs_uri']}")

                    # 2. Extract Metadata
                    print(f"Attempting metadata extraction for {original_filename_secured} from GCS blob {current_file_details['blob_name']}...")
                    file_bytes_for_metadata = metadata_utils.get_gcs_file_bytes(
                        current_file_details['bucket_name'],
                        current_file_details['blob_name']
                    )

                    if file_bytes_for_metadata:
                        extracted_meta = metadata_utils.extract_metadata_from_file_bytes(
                            file_bytes_for_metadata,
                            current_file_details['content_type']
                        )
                        current_file_details['extracted_metadata'] = extracted_meta
                        print(f"Metadata extracted for {original_filename_secured}: {extracted_meta.get('type', 'unknown type')}")
                    else:
                        err_msg = f"Could not retrieve bytes for {original_filename_secured} from GCS for metadata extraction."
                        print(f"Warning: {err_msg}")
                        current_file_details['extracted_metadata'] = {"error": err_msg}

                    uploaded_file_details.append(current_file_details)

                except Exception as e:
                    err_msg = f"An unexpected error occurred while processing file {original_filename_secured}: {str(e)}"
                    print(f"Critical Error: {err_msg}\n{traceback.format_exc()}")
                    file_processing_errors.append({"filename": original_filename_secured, "error": "Server-side processing error."})
                    # Depending on policy, you might choose to fail the whole request here:
                    # return jsonify({"error": "A critical error occurred during file processing.", "filename": original_filename_secured, "details": str(e)}), 500
            else:
                print("Skipping an empty or unnamed file part in the request.")

    if file_processing_errors:
        # If there were non-critical file errors, inform the client but proceed if some files were successful
        # Or, if policy is to fail on any file error, return 400/500 here.
        # For now, we'll let it proceed to AI call if any files were successful,
        # but the errors will be in the final response.
        print(f"Encountered {len(file_processing_errors)} errors during file processing.")

    # Clean up the temporary local upload folder if it was created and no longer needed
    # temp_upload_folder = "temp_uploads_for_processing"
    # if os.path.exists(temp_upload_folder):
    #     import shutil
    #     shutil.rmtree(temp_upload_folder)
    #     print(f"Cleaned up temporary folder: {temp_upload_folder}")


    combined_prompt = SYSTEM_PROMPT + "\n\n[User Query Start]\n" + user_prompt + "\n[User Query End]"
    # Actual AI call will use combined_prompt and uploaded_file_details (which now contains GCS URIs)

    print(f"Combined Prompt for AI (first 200 chars): {combined_prompt[:200]}...")
    if uploaded_file_details:
        print(f"Files processed (GCS & Metadata): {uploaded_file_details}")

    # Construct prompt for AI, including metadata if available
    # This is a conceptual representation; actual formatting will depend on the model
    ai_prompt_parts = [SYSTEM_PROMPT, f"\n\n[User Query Start]\n{user_prompt}\n[User Query End]"]

    for i, file_detail in enumerate(uploaded_file_details):
        ai_prompt_parts.append(f"\n\n--- Attached File {i+1} ({file_detail.get('original_filename', 'N/A')}) ---")
        ai_prompt_parts.append(f"GCS URI: {file_detail.get('gcs_uri', 'N/A')}")
        ai_prompt_parts.append(f"Content Type: {file_detail.get('content_type', 'N/A')}")
        if 'extracted_metadata' in file_detail:
            meta_info = file_detail['extracted_metadata']
            if meta_info and not meta_info.get('error') and not meta_info.get('info'):
                 ai_prompt_parts.append(f"Metadata: {str(meta_info)}") # Convert dict to string for prompt
            elif meta_info.get('error'):
                 ai_prompt_parts.append(f"Metadata Extraction Error: {meta_info['error']}")
            elif meta_info.get('info'):
                 ai_prompt_parts.append(f"Metadata Info: {meta_info['info']}")
        ai_prompt_parts.append("--- End Attached File ---")

    final_ai_prompt = "\n".join(ai_prompt_parts)

    print(f"Final AI Prompt (first 300 chars): {final_ai_prompt[:300]}...")

    # --- Actual AI Call ---
    import vertex_utils

    ai_response_text = None
    ai_error_message = None
    http_status_code = 200 # Default OK

    try:
        if model_config['type'] == 'gemini':
            print(f"Calling Gemini model: {model_config['model_name']}")
            ai_response_text, ai_error_message = vertex_utils.invoke_gemini_model(
                project_id=model_config['project'],
                location=model_config['location'],
                model_name=model_config['model_name'],
                text_prompt=final_ai_prompt,
                file_details_list=uploaded_file_details
            )
        elif model_config['type'] == 'endpoint':
            print(f"Calling Vertex Endpoint: {model_config['endpoint_id']}")
            ai_response_text, ai_error_message = vertex_utils.invoke_vertex_endpoint_model(
                project_id=model_config['project'],
                location=model_config['location'],
                endpoint_id=model_config['endpoint_id'],
                text_prompt=final_ai_prompt,
                file_details_list=uploaded_file_details
            )
        else:
            ai_error_message = f"Unknown model type configured: {model_config['type']}"
            http_status_code = 501 # Not Implemented for unknown type

    except Exception as e:
        # Catch-all for unexpected errors during the setup or call to vertex_utils
        err_msg = f"Unexpected server error during AI model invocation setup: {str(e)}"
        print(f"Critical Error: {err_msg}\n{traceback.format_exc()}")
        ai_error_message = "An unexpected error occurred on the server."
        http_status_code = 500


    if ai_error_message:
        print(f"Error from AI model call or setup: {ai_error_message}")
        # Determine appropriate status code if not already set
        if http_status_code == 200: # If error came from within vertex_utils, might be 502/503 like
             # Heuristic: if "timeout" or "unavailable" or "quota" in error, could be 503.
             # If "permission denied" or "authentication", could be 500 or 401/403 (though ADC usually handles this).
             # For now, a generic 502 (Bad Gateway) for upstream AI issues.
            http_status_code = 502

        return jsonify({
            "error": "AI model processing failed.",
            "details": ai_error_message,
            "file_processing_errors": file_processing_errors if file_processing_errors else None,
            "processed_files_summary": uploaded_file_details if uploaded_file_details else None
        }), http_status_code

    # --- Parse AI Response ---
    # The AI is expected to return text that includes "Fraud Confidence Score: [Score]"
    # and then a "Rationale for Score:" followed by bullet points.
    # This is a basic parsing attempt. A more robust solution might involve
    # asking the model to return JSON, or more structured output.

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
                    # Basic validation of score
                    valid_scores = ["low", "medium", "high", "very high"]
                    if parsed_score.lower() not in valid_scores:
                        print(f"Warning: Parsed score '{parsed_score}' is not in {valid_scores}. Using as is.")
                    score_line_found = True
                except IndexError:
                    print(f"Warning: Could not parse score from line: {line}")

            elif "rationale for score:" in line_lower:
                rationale_section_found = True
                continue # Skip this line itself

            if rationale_section_found and score_line_found: # Only collect rationale after score is found
                if line.strip().startswith("*") or (line.strip() and not line.strip().startswith("**")): # Basic bullet point or continued line
                    current_rationale.append(line.strip().lstrip("* ").strip())

        if current_rationale:
            parsed_rationale_points = [r for r in current_rationale if r] # Filter out empty strings
        elif not rationale_section_found:
             parsed_rationale_points = ["Rationale section not found in AI response."]
        elif not score_line_found:
            parsed_rationale_points = ["Score line not found, so rationale might be incomplete or misattributed."]

        if not score_line_found:
            parsed_score = "Score not found in AI response"

    else: # ai_response_text is None or empty
        parsed_score = "No response from AI"
        parsed_rationale_points = ["AI did not return a response."]


    final_response_to_client = {
        "fraudConfidenceScore": parsed_score,
        "rationale": parsed_rationale_points,
        "raw_ai_response_preview": ai_response_text[:500] if ai_response_text else "N/A",
        "file_processing_errors": file_processing_errors if file_processing_errors else None, # Include file errors
        "processed_files_summary": uploaded_file_details
    }

    return jsonify(final_response_to_client), http_status_code # Return with status

if __name__ == '__main__':
    # Initial configuration checks
    critical_configs_missing = False
    if not GCP_PROJECT_ID:
        print("FATAL: GOOGLE_CLOUD_PROJECT environment variable must be set.")
        critical_configs_missing = True
    if not GCS_BUCKET_NAME:
        print("FATAL: GCS_BUCKET_NAME environment variable must be set.")
        critical_configs_missing = True

    if critical_configs_missing:
        print("Exiting due to missing critical configurations.")
        exit(1) # Exit if essential cloud configs are missing

    print(f"Flask App Initializing with GCP_PROJECT_ID: {GCP_PROJECT_ID}, GCP_LOCATION: {GCP_LOCATION}, GCS_BUCKET_NAME: {GCS_BUCKET_NAME}")
    for model_key, config in MODEL_CONFIGS.items():
        print(f"Model '{model_key}': type='{config['type']}', name/id='{config.get('model_name') or config.get('endpoint_id')}'")
        if config['type'] == 'endpoint' and not config.get('endpoint_id'):
            print(f"WARNING: Endpoint ID not configured for model '{model_key}'. This model will not be usable.")
        elif config['type'] == 'gemini' and not config.get('model_name'):
             print(f"WARNING: Model name not configured for Gemini model '{model_key}'. This model will not be usable.")


    app.run(debug=True, port=5000)
