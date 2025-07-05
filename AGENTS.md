## Baze Insurance FNOL Agent - Agent Instructions

This document provides guidance for AI agents working on the Baze Insurance FNOL Agent codebase.

### Project Overview

The project is a web application designed to assist human claims handlers in detecting potential fraud in First Notice of Loss (FNOL) submissions. It allows users to select an AI model, submit a text prompt (claim details), and upload supporting documents (images, PDFs, etc.). The backend, built with Flask, processes this information, (conceptually) calls a Vertex AI model, and returns a fraud confidence score (Low, Medium, High, Very High) along with a rationale.

### Key Components

*   **`index.html`**: The main HTML file for the user interface.
*   **`static/style.css`**: CSS for styling the application. Aims for a clean, modern UI similar to Google's Gemini.
*   **`static/script.js`**: Frontend JavaScript for handling user interactions, model selection, file uploads, and communication with the backend API.
*   **`app.py`**: The Flask backend.
    *   Serves the frontend (`index.html` and static files).
    *   Provides the `/api/analyze` endpoint for processing claims.
    *   Manages file uploads.
    *   Contains the core system prompt for the AI model.
    *   (Conceptually) interacts with Vertex AI models based on environment variable configurations.
*   **`uploads/`**: Directory where uploaded files are temporarily stored. This directory is created automatically if it doesn't exist.

### System Prompt

The core system prompt is defined as a string variable `SYSTEM_PROMPT` in `app.py`. This prompt guides the AI model's behavior and output format. When modifying or extending the AI's capabilities, consider if updates to this system prompt are necessary. The prompt emphasizes:
*   Assigning one of four fraud confidence scores.
*   Providing a detailed, bullet-pointed rationale.
*   Analyzing documents and images for authenticity, consistency, and forensic indicators.
*   Holistic pattern analysis.

### Environment Variables

The application is designed to use the following environment variables to specify Vertex AI model endpoints:

*   `GEMINI_2_5_PRO_ENDPOINT`
*   `GEMINI_2_5_FLASH_ENDPOINT`
*   `GEMMA_3_ENDPOINT`
*   `LLAMA_3_3_ENDPOINT`

These are accessed in `app.py` using `os.getenv()`. When deploying or testing with actual models, ensure these are correctly set.

### Development Guidelines

1.  **Model Integration**:
    *   The current implementation uses a mocked AI response in `app.py`. To integrate actual Vertex AI models, you will need to:
        *   Install the necessary Google Cloud client libraries (e.g., `google-cloud-aiplatform`).
        *   Implement the logic to call the selected model's endpoint using the `combined_prompt` (system prompt + user prompt) and any uploaded file data.
        *   Handle API responses and errors gracefully.
        *   Ensure that file data (especially images and PDFs) is correctly formatted and passed to the Vertex AI API. This might involve reading file content as bytes, base64 encoding, or using specific SDK methods for multimodal inputs.
2.  **Adding New Models**:
    *   Update the `<select>` dropdown in `index.html`.
    *   Add a corresponding environment variable for the new model's endpoint.
    *   Extend the logic in `app.py` (in the `/api/analyze` route) to handle the new model selection and call its specific endpoint.
3.  **Error Handling**:
    *   Improve error handling on both frontend (`script.js`) and backend (`app.py`). Provide clear feedback to the user.
4.  **Security**:
    *   File uploads are handled by `werkzeug.utils.secure_filename`. Review and enhance security measures as needed, especially if the application is exposed externally (e.g., input validation, further sanitization, rate limiting).
    *   Uploaded files are currently stored in the `uploads/` directory. For a production system, consider a more robust file storage solution (e.g., Google Cloud Storage) and a strategy for managing these files (e.g., lifecycle policies).
5.  **Testing**:
    *   Implement unit tests for backend logic (e.g., using `pytest`).
    *   Implement frontend tests if possible.
    *   Conduct thorough end-to-end testing, especially when integrating with live AI models.
6.  **Code Style**:
    *   Maintain clean, readable code with appropriate comments.
    *   Follow Python (PEP 8) and JavaScript best practices.

### Future Enhancements (Considerations)

*   **Real-time Progress**: For long-running AI analyses, implement real-time progress updates to the user.
*   **Advanced File Analysis**: Integrate tools for more in-depth client-side or server-side pre-processing of files (e.g., EXIF data extraction from images directly in the backend, PDF text extraction).
*   **User Authentication**: If needed, add user authentication and authorization.
*   **Session Management**: For more complex interactions or user history.
*   **Streaming Responses**: If models support streaming, update the UI to display responses as they arrive.

Remember to keep this `AGENTS.md` file updated as the project evolves.Tool output for `create_file_with_block`:
