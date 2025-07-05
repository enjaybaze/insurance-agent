## Baze Insurance FNOL Agent - Agent Instructions

This document provides guidance for AI agents working on the Baze Insurance FNOL Agent codebase.

### Project Overview

The project is a web application designed to assist human claims handlers in detecting potential fraud in First Notice of Loss (FNOL) submissions. It allows users to select an AI model, submit a text prompt, and upload supporting documents. The backend, built with Flask, processes this information: uploads files to Google Cloud Storage (GCS), extracts metadata, constructs an enriched prompt, calls a configured Vertex AI model (either native Gemini or a custom model on an Endpoint), parses the response, and returns a fraud confidence score and rationale.

### Key Components & Architecture

*   **`app.py`**: The main Flask application.
    *   Serves the frontend (`index.html` and static files from `static/`).
    *   Handles API requests to `/api/analyze`.
    *   Orchestrates file uploads (via `gcs_utils.py`), metadata extraction (via `metadata_utils.py`), AI model calls (via `vertex_utils.py`), and response parsing.
    *   Manages model configurations loaded from environment variables.
*   **`gcs_utils.py`**: Module for all interactions with Google Cloud Storage. Includes functions for uploading files.
*   **`metadata_utils.py`**: Module for extracting metadata from files (e.g., EXIF from images, info from PDFs). Downloads files from GCS as needed for extraction.
*   **`vertex_utils.py`**: Module for invoking Vertex AI models.
    *   `invoke_gemini_model()`: For calling native Vertex AI Gemini multimodal models.
    *   `invoke_vertex_endpoint_model()`: For calling custom models deployed on Vertex AI Endpoints (e.g., Gemma, Llama).
*   **`index.html`**: Main HTML page for the user interface.
*   **`static/style.css`**: CSS for styling.
*   **`static/script.js`**: Frontend JavaScript for UI interactions, API calls to the backend, and display of results/errors.
*   **`requirements.txt`**: Lists all Python dependencies.
*   **`uploads/`**: This directory is NO LONGER used for persistent user file storage.

### System Prompt & AI Interaction

The base `SYSTEM_PROMPT` is defined in `app.py`. The actual prompt sent to the AI model is dynamically constructed in `app.py`'s `/api/analyze` route by:
1.  Starting with the `SYSTEM_PROMPT`.
2.  Appending the user's textual query.
3.  For each uploaded file, appending its GCS URI, content type, and a summary of its extracted metadata.

The AI model is expected to return a textual response containing a "Fraud Confidence Score:" line and a "Rationale for Score:" section with bullet points. `app.py` includes logic to parse this structure.

### Environment Variables & Configuration

The application relies heavily on environment variables. Refer to `README.md` for a full list and setup instructions. Key categories:
*   **GCP Configuration**: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GCS_BUCKET_NAME`.
*   **Model Identifiers**: `GEMINI_PRO_MODEL_NAME`, `GEMINI_FLASH_MODEL_NAME`, `GEMMA_ENDPOINT_ID`, `LLAMA_ENDPOINT_ID`.
These are used in `app.py` to configure `MODEL_CONFIGS` and initialize utility modules.

### Google Cloud Authentication

*   Primarily uses Application Default Credentials (ADC). Local development requires `gcloud auth application-default login`.
*   Deployed environments need appropriate service account IAM permissions (Vertex AI User, Storage Object Creator/Viewer).
*   `GOOGLE_APPLICATION_CREDENTIALS` can be used as an alternative.
*   Details are in `README.md`.

### Dependencies
Install using `pip install -r requirements.txt`. Includes `Flask`, `google-cloud-aiplatform`, `google-cloud-storage`, `Pillow`, `PyPDF2`.

### Development Guidelines

1.  **Model Integration & `vertex_utils.py`**:
    *   **Endpoint Model Payload**: The `invoke_vertex_endpoint_model` function sends a JSON payload like `{"prompt": "...", "gcs_files": [{"gcs_uri": "...", "mime_type": "...", "filename": "..."}]}`. If your custom deployed model (Gemma, Llama) expects a different input structure, you **must** modify this function.
    *   **Gemini Model Calls**: Uses `Part.from_uri` for GCS files. Ensure MIME types are correct.
    *   Error handling for API calls is included but can be further refined (e.g., more specific retry logic if appropriate).
2.  **File Handling (`gcs_utils.py`, `metadata_utils.py`)**:
    *   Files are uploaded to GCS. Metadata is extracted by temporarily downloading the file bytes from GCS.
    *   Consider efficiency for very large files if metadata extraction becomes a bottleneck (though current models have input size limits anyway).
3.  **Adding New Models**:
    *   Update `index.html` dropdown.
    *   Define new environment variables for the model.
    *   Add to `MODEL_CONFIGS` in `app.py`.
    *   If interaction logic is new, add/modify functions in `vertex_utils.py`.
    *   Update dispatch logic in `app.py`.
4.  **Error Handling**:
    *   **Backend**: `app.py` now catches errors from utility modules and AI calls, returning structured JSON errors (e.g., `{"error": "...", "details": "..."}`) with appropriate HTTP status codes (400, 500, 502, 501). File processing errors are also reported in the response. Logging uses `print` and `traceback`; consider a more formal logging library for production.
    *   **Frontend**: `static/script.js` parses these JSON error responses and displays them. It also handles network errors and provides user-friendly messages. HTML escaping is used for displaying error details.
5.  **Security**:
    *   GCS bucket permissions must be correctly configured (least privilege).
    *   `secure_filename` is used.
    *   Input validation on text prompt is basic; consider if more is needed.
    *   HTML escaping in `script.js` helps prevent XSS from error messages or AI output.
6.  **Testing**:
    *   **Crucial**: Full end-to-end testing requires a configured GCP environment (project, bucket, enabled APIs, deployed models/endpoints, and ADC/service account).
    *   Test various file types, including unsupported ones for metadata.
    *   Test error conditions (e.g., invalid bucket/model names, API errors, quota limits).
    *   Consider `pytest` and `unittest.mock` for unit tests of utility functions to avoid constant cloud calls during isolated development.
7.  **Code Style & Structure**:
    *   Maintain modularity. Adhere to PEP 8 (Python) and general JS best practices.

### Future Enhancements (Considerations)

*   **User Authentication/Authorization**.
*   **Streaming AI Responses**.
*   **UI for Displaying Extracted Metadata**.
*   **Configuration for Endpoint Payloads**: If many custom models with different payload structures are used, make this more configurable than hardcoding in `vertex_utils.py`.
*   **Advanced GCS Management**: Lifecycle policies, more specific error handling for GCS uploads (retries).
*   **Formal Logging**: Implement Python's `logging` module.

This document should be kept up-to-date with significant architectural changes.
```
