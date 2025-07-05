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
        }
    });

    submitButton.addEventListener('click', async () => {
        const selectedModel = modelSelect.value;
        const promptText = promptInput.value;
        const files = fileUpload.files;

        if (!promptText.trim()) {
            responseArea.innerHTML = '<p style="color: red;">Please enter a prompt.</p>';
            return;
        }

        submitButton.classList.add('loading');
        submitButton.disabled = true;
        responseArea.innerHTML = '<p>Processing...</p>';

        const formData = new FormData();
        formData.append('model', selectedModel);
        formData.append('prompt', promptText);
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData, // FormData handles the Content-Type header automatically
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error occurred' }));
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.error}`);
            }

            const result = await response.json();

            let responseHTML = `<h3>Fraud Confidence Score: ${result.fraudConfidenceScore}</h3>`;
            responseHTML += `<h4>Rationale for Score:</h4>`;
            if (result.rationale && result.rationale.length > 0) {
                responseHTML += `<ul>`;
                result.rationale.forEach(point => {
                    responseHTML += `<li>${point}</li>`;
                });
                responseHTML += `</ul>`;
            } else {
                responseHTML += `<p>No rationale provided.</p>`;
            }
            responseArea.innerHTML = responseHTML;

        } catch (error) {
            console.error('Error submitting form:', error);
            responseArea.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        } finally {
            submitButton.classList.remove('loading');
            submitButton.disabled = false;
        }
    });
});
