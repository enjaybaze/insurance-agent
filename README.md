# Baze Insurance FNOL Agent

Baze Insurance FNOL Agent is a web application designed to assist insurance claims handlers by providing an AI-powered analysis of First Notice of Loss (FNOL) submissions. The tool aims to identify potential fraud by analyzing user-provided claim details and uploaded documents/images, then returning a fraud confidence score and rationale.

## Features

*   **AI Model Selection**: Choose from a list of AI models (currently mocked, intended for Vertex AI):
    *   Gemini 2.5 Pro
    *   Gemini 2.5 Flash
    *   Gemma 3
    *   Llama 3.3
*   **Prompt Input**: Enter detailed information about the insurance claim.
*   **Document/Image Upload**: Upload multiple files (images, PDFs, etc.) relevant to the claim.
*   **Fraud Analysis**: The backend processes the input and (conceptually) uses the selected AI model with a specialized system prompt to:
    *   Assign a **Fraud Confidence Score**: Low, Medium, High, or Very High.
    *   Provide a **Detailed Rationale** for the score.
*   **User-Friendly Interface**: Clean UI inspired by modern web applications.

## Technical Stack

*   **Frontend**: HTML, CSS, JavaScript
*   **Backend**: Python (Flask)
*   **AI Models**: (Conceptual) Google Cloud Vertex AI - integration pending.

## Project Structure

```
.
├── app.py                # Flask backend application
├── index.html            # Main HTML file for the UI
├── static/
│   ├── script.js         # Frontend JavaScript
│   └── style.css         # CSS styles
├── uploads/              # Directory for storing uploaded files (created automatically)
├── AGENTS.md             # Instructions for AI agents working on this codebase
└── README.md             # This file
```

## Setup and Running

### Prerequisites

*   Python 3.7+
*   Flask (`pip install Flask`)

### Environment Variables

For actual AI model integration (currently mocked), the application expects the following environment variables to be set with the appropriate Vertex AI model endpoint URLs:

*   `GEMINI_2_5_PRO_ENDPOINT="your_gemini_2_5_pro_endpoint_url"`
*   `GEMINI_2_5_FLASH_ENDPOINT="your_gemini_2_5_flash_endpoint_url"`
*   `GEMMA_3_ENDPOINT="your_gemma_3_endpoint_url"`
*   `LLAMA_3_3_ENDPOINT="your_llama_3_3_endpoint_url"`

Set these in your shell environment before running the application if you are connecting to live models. For the current mocked version, these are not strictly required but are good to be aware of.

### Installation & Running

1.  **Clone the repository (if applicable) or download the files.**
2.  **Navigate to the project directory.**
3.  **Install dependencies:**
    ```bash
    pip install Flask
    ```
4.  **(Optional) Set Environment Variables:**
    As mentioned above, if you have Vertex AI endpoints, set them in your terminal:
    ```bash
    export GEMINI_2_5_PRO_ENDPOINT="your_endpoint_here"
    # etc. for other models
    ```
5.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    The application will typically start on `http://127.0.0.1:5000/`.

### How to Use

1.  Open your web browser and navigate to `http://127.0.0.1:5000/`.
2.  Select an AI model from the dropdown menu.
3.  Enter the claim details or your query into the prompt text area.
4.  Click "Upload Documents/Images" to select any relevant files. You will see a list of selected files.
5.  Click the "Submit" button.
6.  The application will process the request (currently with a mocked backend response) and display the Fraud Confidence Score and Rationale.

## Development

*   The core AI logic and system prompt are located in `app.py`.
*   Frontend interactions are managed in `static/script.js`.
*   UI styling is in `static/style.css`.
*   The AI response is currently mocked in `app.py`. To connect to actual Vertex AI models, you would need to:
    *   Install the `google-cloud-aiplatform` library.
    *   Modify the `/api/analyze` route in `app.py` to use the Vertex AI SDK to call the selected model with the combined prompt and uploaded files.
    *   Ensure your environment is authenticated with Google Cloud and has the necessary permissions.

## Future Development Ideas

*   Implement actual calls to Vertex AI models.
*   Add more robust error handling and user feedback.
*   Develop a more sophisticated file management system (e.g., using cloud storage for uploads).
*   Incorporate user authentication if needed.
*   Extract and display metadata from uploaded files.

Refer to `AGENTS.md` for more detailed instructions if you are an AI agent contributing to this project.
```
