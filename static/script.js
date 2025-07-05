document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('model-select');
    const promptInput = document.getElementById('prompt-input');
    const fileUpload = document.getElementById('file-upload');
    const submitButton = document.getElementById('submit-button');
    const responseArea = document.getElementById('response-area');
    const selectedFilesList = document.getElementById('selected-files-list');

    fileUpload.addEventListener('change', () => {
        selectedFilesList.innerHTML = ''; // Clear previous list
        if (fileUpload.files.length > 0) {
            const filesArray = Array.from(fileUpload.files);
            filesArray.forEach(file => {
                const listItem = document.createElement('li');
                listItem.textContent = `${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
                selectedFilesList.appendChild(listItem);
            });
        } else {
            selectedFilesList.innerHTML = '<li>No files selected.</li>'; // Show message if selection cleared
        }
    });

    submitButton.addEventListener('click', async () => {
        const selectedModel = modelSelect.value;
        const promptText = promptInput.value;
        const files = fileUpload.files;

        if (!promptText.trim()) {
            responseArea.innerHTML = '<p style="color: red; font-weight: bold;">Please enter a prompt.</p>';
            return;
        }

        submitButton.classList.add('loading');
        submitButton.disabled = true;
        responseArea.innerHTML = '<p>Processing your request... this may take a moment.</p>';

        const formData = new FormData();
        formData.append('model', selectedModel);
        formData.append('prompt', promptText);
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            let resultJson = {};
            try {
                // Try to parse JSON. If it fails, it means the response body wasn't valid JSON.
                resultJson = await response.json();
            } catch (jsonError) {
                console.error('Response was not valid JSON:', jsonError);
                // If response.ok is false AND parsing failed, it's likely an HTML error page from a proxy or server misconfiguration.
                if (!response.ok) {
                    const responseText = await response.text().catch(() => "Could not read response text.");
                    console.error('Non-JSON error response text:', responseText);
                    responseArea.innerHTML = `<p style="color: red; font-weight: bold;">Server Error: ${response.status} ${response.statusText}.<br>The server returned an unexpected response. Check console for more details.</p>`;
                    return; // Early exit from the try...finally block
                }
                // If response.ok is true but JSON parsing failed (highly unlikely for this app's design but possible)
                resultJson = {
                    fraudConfidenceScore: "Error",
                    rationale: ["Response from server was not valid JSON, though the request seemed to be successful initially."]
                };
            }

            if (!response.ok) { // Handle HTTP errors (4xx, 5xx) where JSON might contain error details
                let errorMsg = `<strong>Request Failed (Status: ${response.status} ${response.statusText || ''})</strong>`;
                if (resultJson.error) {
                    errorMsg = `<strong>Error: ${resultJson.error}</strong>`;
                    if (resultJson.details) {
                        // Basic sanitization for details to prevent HTML injection if details are user-influenced (though less likely here)
                        errorMsg += `<br>Details: ${escapeHTML(resultJson.details.toString())}`;
                    }
                } else if (response.status === 502) {
                    errorMsg = `<strong>Error: Upstream service (AI Model) failed.</strong><br>This might be a temporary issue with the AI provider or model configuration. Please try again later or check server logs.`;
                } else if (response.status === 500) {
                     errorMsg = `<strong>Error: An unexpected error occurred on the server.</strong><br>Please report this issue if it persists.`;
                } else if (response.status === 400 && resultJson.error) {
                     errorMsg = `<strong>Input Error: ${escapeHTML(resultJson.error)}</strong>`;
                     if (resultJson.details) errorMsg += `<br>Details: ${escapeHTML(resultJson.details)}`;
                }


                if (resultJson.file_processing_errors && resultJson.file_processing_errors.length > 0) {
                    errorMsg += `<br><br><strong>File Processing Issues:</strong><ul style="list-style-type: disc; margin-left: 20px;">`;
                    resultJson.file_processing_errors.forEach(fpError => {
                        errorMsg += `<li>${escapeHTML(fpError.filename)}: ${escapeHTML(fpError.error)}</li>`;
                    });
                    errorMsg += `</ul>`;
                }
                responseArea.innerHTML = `<div style="color: red; border: 1px solid red; padding: 10px; background-color: #ffeeee;">${errorMsg}</div>`;
            } else { // response.ok and JSON parsed successfully
                let responseHTML = `<h3>Fraud Confidence Score: ${escapeHTML(resultJson.fraudConfidenceScore)}</h3>`;
                responseHTML += `<h4>Rationale for Score:</h4>`;
                if (resultJson.rationale && resultJson.rationale.length > 0) {
                    responseHTML += `<ul style="list-style-type: disc; margin-left: 20px;">`;
                    resultJson.rationale.forEach(point => {
                        responseHTML += `<li>${escapeHTML(point)}</li>`;
                    });
                    responseHTML += `</ul>`;
                } else {
                    responseHTML += `<p>No rationale provided or rationale parsing failed.</p>`;
                }

                if (resultJson.file_processing_errors && resultJson.file_processing_errors.length > 0) {
                    responseHTML += `<hr><h4>Partial Success - File Processing Issues:</h4><ul style="list-style-type: disc; margin-left: 20px; color: #cc8400;">`; // Darker orange
                    resultJson.file_processing_errors.forEach(fpError => {
                        responseHTML += `<li>${escapeHTML(fpError.filename)}: ${escapeHTML(fpError.error)}</li>`;
                    });
                    responseHTML += `</ul>`;
                }

                if (resultJson.raw_ai_response_preview && resultJson.raw_ai_response_preview !== "N/A") {
                     responseHTML += `<hr><h4>AI Response Preview (Debug):</h4><pre style="white-space: pre-wrap; font-size: 0.8em; background-color: #f0f0f0; border: 1px solid #ccc; padding: 10px; max-height: 200px; overflow-y: auto; word-break: break-all;">${escapeHTML(resultJson.raw_ai_response_preview)}</pre>`;
                }
                responseArea.innerHTML = responseHTML;
            }

        } catch (error) { // Catches network errors or if fetch() itself throws (e.g., CORS, network down)
            console.error('Fetch call failed or other JS error:', error);
            responseArea.innerHTML = `<p style="color: red; font-weight: bold;">Application Error: ${error.message}. <br>Please check your internet connection or contact support if the issue persists.</p>`;
        } finally {
            submitButton.classList.remove('loading');
            submitButton.disabled = false;
        }
    });

    // Helper to escape HTML to prevent XSS if displaying user-influenced error messages or AI output directly
    function escapeHTML(str) {
        if (str === null || typeof str === 'undefined') return '';
        return str.toString()
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
