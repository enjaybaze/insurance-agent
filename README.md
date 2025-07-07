# Insurance FNOL Agent

Baze Insurance FNOL Agent is a web application designed to assist insurance claims handlers by providing an AI-powered analysis of First Notice of Loss (FNOL) submissions. The tool aims to identify potential fraud by analyzing user-provided claim details and uploaded documents/images, then returning a fraud confidence score and rationale.

## Features

*   **AI Model Selection**: Choose from a list of AI models:
    *   Gemini 2.5 Pro (Vertex AI native)
    *   Gemini 2.5 Flash (Vertex AI native)
    *   Gemma 3 (Custom deployed on Vertex AI Endpoint)
    *   Llama 3.3 (Custom deployed on Vertex AI Endpoint)
*   **Prompt Input**: Enter detailed information about the insurance claim.
*   **Document/Image Upload**: Upload multiple files (images, PDFs, etc.) relevant to the claim. Files are stored in Google Cloud Storage.
*   **Metadata Extraction**: Basic metadata (EXIF for images, PDF info) is extracted from uploaded files and can be used to enrich the AI prompt.
*   **Fraud Analysis**: The backend processes the input:
    *   Uploads files to GCS.
    *   Extracts metadata.
    *   Constructs a detailed prompt including system instructions, user query, GCS URIs of files, and extracted metadata.
    *   Calls the selected Vertex AI model.
    *   Parses the AI's response to provide a **Fraud Confidence Score** (Low, Medium, High, or Very High) and a **Detailed Rationale**.
*   **User-Friendly Interface**: Clean UI with improved error feedback.

## Technical Stack

*   **Frontend**: HTML, CSS, JavaScript
*   **Backend**: Python (Flask)
*   **AI Models**: Google Cloud Vertex AI (Gemini native models and custom deployed models via Endpoints).
*   **File Storage**: Google Cloud Storage (GCS).
*   **Libraries**: `google-cloud-aiplatform`, `google-cloud-storage`, `Pillow`, `PyPDF2`.

## Project Structure

```
.
├── app.py                # Flask backend application
├── gcs_utils.py          # Utilities for Google Cloud Storage interaction
├── metadata_utils.py     # Utilities for extracting file metadata
├── vertex_utils.py       # Utilities for Vertex AI model invocation
├── index.html            # Main HTML file for the UI
├── static/
│   ├── script.js         # Frontend JavaScript
│   └── style.css         # CSS styles
├── requirements.txt      # Python dependencies
├── AGENTS.md             # Instructions for AI agents working on this codebase
└── README.md             # This file
```
The `uploads/` directory is no longer used for persistent user file storage.

## Setup and Running

### Prerequisites

*   Python 3.7+
*   Google Cloud SDK installed and configured (for Application Default Credentials).
*   A Google Cloud Project with the Vertex AI API and Cloud Storage API enabled.
*   A Google Cloud Storage bucket created for this application.
*   If using custom models (Gemma, Llama), they must be deployed to Vertex AI Endpoints.

### `requirements.txt`
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```
Key libraries include `Flask`, `google-cloud-aiplatform`, `google-cloud-storage`, `Pillow`, `PyPDF2`.

### Environment Variables

The application requires the following environment variables:
*   `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID.
*   `GOOGLE_CLOUD_LOCATION`: The region for your Vertex AI resources (e.g., `us-central1`).
*   `GCS_BUCKET_NAME`: The name of the Google Cloud Storage bucket for uploading files. This bucket must exist.
*   **Model-Specific Variables:**
    *   `GEMINI_PRO_MODEL_NAME`: Name of the Gemini Pro model in Vertex AI (e.g., `gemini-1.5-pro-preview-0409`).
    *   `GEMINI_FLASH_MODEL_NAME`: Name of the Gemini Flash model in Vertex AI (e.g., `gemini-1.5-flash-preview-0514`).
    *   `GEMMA_ENDPOINT_ID`: The full Vertex AI Endpoint resource name for your deployed Gemma model (e.g., `projects/YOUR_PROJECT_ID/locations/YOUR_LOCATION/endpoints/YOUR_GEMMA_ENDPOINT_ID`).
    *   `LLAMA_ENDPOINT_ID`: The full Vertex AI Endpoint resource name for your deployed Llama 3.3 model.
    *   *(Note for custom models: Ensure your deployed model container expects a JSON payload with a `prompt` field (string) and an optional `gcs_files` field (list of objects, each with `gcs_uri`, `mime_type`, `filename`). The `vertex_utils.py` module sends this format. Adjust `vertex_utils.invoke_vertex_endpoint_model` if your model expects a different structure.)*

**Important: Google Cloud Authentication**

The application uses Application Default Credentials (ADC) by default.
*   **Local Development:** Authenticate via the gcloud CLI:
    ```bash
    gcloud auth application-default login
    ```
*   **Deployment (e.g., Cloud Run, GKE):** Ensure the runtime service account has these IAM permissions:
    *   Vertex AI User (or `aiplatform.endpoints.predict` and `aiplatform.models.predict`).
    *   Storage Object Creator (`roles/storage.objectCreator`) on the GCS bucket.
    *   Storage Object Viewer (`roles/storage.objectViewer`) on the GCS bucket.
    Alternatively, set `GOOGLE_APPLICATION_CREDENTIALS` to the path of a service account key JSON file.

### Installation & Running

1.  **Clone/Download files.**
2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```
3.  **Install dependencies:** `pip install -r requirements.txt`
4.  **Set Environment Variables** (as described above). Ensure the GCS bucket exists.
5.  **Authenticate with Google Cloud** (if local and not done: `gcloud auth application-default login`).
6.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    Access at `http://127.0.0.1:5000/`.

### How to Use

1.  Open the application in your browser.
2.  Select an AI model.
3.  Enter claim details in the prompt area.
4.  Upload relevant documents/images.
5.  Click "Submit".
6.  View the AI's fraud confidence score and rationale. Error messages will be displayed if issues occur.

## Development Status

*   The application now makes actual calls to Vertex AI models and uses Google Cloud Storage.
*   Metadata is extracted from uploaded files.
*   Error handling has been significantly improved.

## Future Development Ideas

*   User authentication.
*   UI display of extracted metadata.
*   Streaming AI responses.
*   More granular GCS file management (lifecycle rules, folders per claim).
*   Unit and integration tests.

Refer to `AGENTS.md` for detailed developer/agent instructions.
```
