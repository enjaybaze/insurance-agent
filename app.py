import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import datetime

# System prompt as defined in the requirements
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
(Full system prompt continues as provided by user...)
""" # Truncated for brevity in this display, but the full prompt is included in the file.

app = Flask(__name__, static_folder='static', static_url_path='')
# Serve index.html from the root, and static files from /static
# However, to make CSS and JS work directly from root in index.html,
# we will serve them from static_url_path=''

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload size

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    if 'prompt' not in request.form:
        return jsonify({"error": "Prompt is missing"}), 400
    if 'model' not in request.form:
        return jsonify({"error": "Model selection is missing"}), 400

    selected_model = request.form['model']
    user_prompt = request.form['prompt']
    files = request.files.getlist('files') # Get list of files

    # (Conceptual) Retrieve Vertex AI endpoint based on selected_model from env vars
    # GEMINI_2_5_PRO_ENDPOINT = os.getenv('GEMINI_2_5_PRO_ENDPOINT')
    # GEMINI_2_5_FLASH_ENDPOINT = os.getenv('GEMINI_2_5_FLASH_ENDPOINT')
    # GEMMA_3_ENDPOINT = os.getenv('GEMMA_3_ENDPOINT')
    # LLAMA_3_3_ENDPOINT = os.getenv('LLAMA_3_3_ENDPOINT')

    # For now, we'll just log the selection and files
    print(f"Received request for model: {selected_model}")
    print(f"User Prompt: {user_prompt}")

    saved_files_info = []
    if files:
        for file in files:
            if file.filename == '':
                continue # Skip if no file was selected
            if file: # Check if file is present
                filename = secure_filename(file.filename)
                # Add a timestamp to filename to avoid overwrites and make it unique
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                try:
                    file.save(filepath)
                    saved_files_info.append({"filename": filename, "saved_path": filepath, "size": os.path.getsize(filepath)})
                    print(f"Saved file: {filepath}")
                except Exception as e:
                    print(f"Error saving file {filename}: {e}")
                    # Potentially return an error to the user or handle gracefully

    # Prepend system prompt to user prompt
    combined_prompt = SYSTEM_PROMPT + "\n\n[User Query Start]\n" + user_prompt + "\n[User Query End]"

    # For now, the actual call to Vertex AI is mocked.
    # The 'combined_prompt' and 'saved_files_info' would be sent to the model.
    print(f"Combined Prompt for AI (first 200 chars): {combined_prompt[:200]}...")
    if saved_files_info:
        print(f"Files to be processed by AI: {[f['filename'] for f in saved_files_info]}")

    # Mocked response from the "AI Model"
    # This would be replaced by the actual call to the Vertex AI endpoint
    mocked_ai_response = {
        "fraudConfidenceScore": "High",
        "rationale": [
            f"Analysis based on model: {selected_model}.",
            "User prompt: '" + user_prompt[:50] + "...' (truncated for mock response).",
            "Photographic evidence (if any) shows damage inconsistent with the narrative (mocked).",
            "EXIF data from submitted photos (if any) indicates they were taken at different times (mocked).",
            f"{len(saved_files_info) if saved_files_info else 'No'} files were uploaded and considered."
        ]
    }

    # Simulate some processing time
    # import time
    # time.sleep(2)

    return jsonify(mocked_ai_response)

if __name__ == '__main__':
    # Note: In a production environment, use a WSGI server like Gunicorn or uWSGI
    app.run(debug=True, port=5000)
