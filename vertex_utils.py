import os
import google.auth
import google.auth.transport.requests
import google.api_core.exceptions
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
# from google.cloud import aiplatform as old_aiplatform # No longer needed by revamped function
from google.cloud import storage # For GCS operations, though mostly via gcs_utils

# --- OpenAI SDK related imports ---
import openai
import PyPDF2 # For PDF processing
from io import BytesIO # For PDF processing
import gcs_utils # For fetching PDF from GCS
import re # For parsing endpoint ID


# --- Gemini Model Invocation (using new vertexai SDK) ---
# This function remains unchanged as per user request.
def invoke_gemini_model(project_id, location, model_name, text_prompt, file_details_list):
    """
    Invokes a Gemini multimodal model on Vertex AI using the new vertexai SDK.
    """
    print(f"Invoking Gemini model (vertexai SDK): {model_name} in {project_id}/{location}")
    print(f"Text prompt (first 100 chars): {text_prompt[:100]}...")
    print(f"File details for Gemini: {file_details_list}")

    try:
        vertexai.init(project=project_id, location=location)
        if "/" in model_name:
            model_name_short = model_name.split("/")[-1]
        else:
            model_name_short = model_name
        model = GenerativeModel(model_name_short)
        prompt_parts = [Part.from_text(text_prompt)]
        for file_detail in file_details_list:
            if file_detail.get("gcs_uri") and file_detail.get("content_type"):
                try:
                    prompt_parts.append(
                        Part.from_uri(uri=file_detail["gcs_uri"], mime_type=file_detail["content_type"])
                    )
                except Exception as e:
                    print(f"Warning: Could not create Part from URI for {file_detail['gcs_uri']}. Error: {e}")

        if not any(hasattr(part, 'text') and part.text for part in prompt_parts):
            if not any(hasattr(part, 'file_data') for part in prompt_parts):
                 return None, "Cannot send an empty prompt (no text and no files) to Gemini."

        generation_config_dict = {
            "max_output_tokens": 8192, "temperature": 0.4, "top_p": 1.0,
        }
        gen_config_instance = GenerationConfig(**generation_config_dict)
        response = model.generate_content(
            contents=prompt_parts, generation_config=gen_config_instance
        )

        if not response.candidates:
            block_reason_msg = "Unknown"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = response.prompt_feedback.block_reason_message or str(response.prompt_feedback.block_reason)
            safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in response.prompt_feedback.safety_ratings]) if response.prompt_feedback else "N/A"
            error_detail = f"Gemini (vertexai SDK) returned no candidates. Block reason: {block_reason_msg}. Safety: [{safety_ratings_str}]"
            print(error_detail)
            return None, error_detail

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            finish_reason_str = str(candidate.finish_reason) if candidate.finish_reason else "Unknown"
            safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in candidate.safety_ratings]) if candidate.safety_ratings else "N/A"
            error_detail = f"Gemini (vertexai SDK) response candidate was empty. Finish reason: {finish_reason_str}. Safety: [{safety_ratings_str}]"
            print(error_detail)
            return None, error_detail
        response_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text is not None])
        return response_text, None
    except google.api_core.exceptions.GoogleAPIError as e:
        return None, f"Google API Error invoking Gemini (vertexai SDK) {model_name}: {e}"
    except Exception as e:
        import traceback
        print(f"Unexpected error invoking Gemini (vertexai SDK) {model_name}: {str(e)}\n{traceback.format_exc()}")
        return None, f"Unexpected error invoking Gemini (vertexai SDK) {model_name}: {str(e)}"


# --- Vertex AI Endpoint Model Invocation (Revamped) ---
def invoke_vertex_endpoint_model(project_id, location, endpoint_id, text_prompt, file_details_list):
    """
    Invokes a model deployed on a Vertex AI Endpoint using the OpenAI SDK chat completions method.
    For Gemma or Llama models, if PDFs are provided, their text is extracted and appended to the prompt.
    """
    print(f"Revamped Invoking Vertex Endpoint: {endpoint_id} using OpenAI SDK method.")

    current_text_prompt = str(text_prompt)

    model_type_for_pdf_extraction = "Gemma"
    endpoint_id_lower = endpoint_id.lower()
    if "gemma" in endpoint_id_lower:
        model_type_for_pdf_extraction = "Gemma"
    elif "llama" in endpoint_id_lower:
        model_type_for_pdf_extraction = "Llama"

    if model_type_for_pdf_extraction:
        print(f"{model_type_for_pdf_extraction} model detected. Checking for PDFs to extract text...")
        pdf_texts_appended_count = 0
        for file_detail in file_details_list:
            if file_detail.get("content_type") == "application/pdf":
                original_fn = file_detail.get('original_filename', 'unknown_pdf')
                print(f"Processing PDF: {original_fn} from GCS bucket: {file_detail.get('bucket_name')} blob: {file_detail.get('blob_name')}")
                try:
                    if not file_detail.get('bucket_name') or not file_detail.get('blob_name'):
                        print(f"Warning: Missing bucket_name or blob_name for PDF: {original_fn}. Skipping text extraction.")
                        current_text_prompt += f"\n\n--- Attached PDF ({original_fn}) could not be processed (missing GCS details). ---"
                        continue

                    pdf_bytes = gcs_utils.get_gcs_file_bytes(file_detail['bucket_name'], file_detail['blob_name'])

                    if pdf_bytes:
                        extracted_text = ""
                        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
                        for page_num in range(len(reader.pages)):
                            page = reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text += page_text + "\n"

                        if extracted_text.strip():
                            current_text_prompt += f"\n\n--- Extracted PDF Content ({original_fn}) ---\n{extracted_text.strip()}\n--- End PDF Content ---"
                            pdf_texts_appended_count += 1
                            print(f"Successfully extracted and appended text from PDF: {original_fn}")
                        else:
                            print(f"Warning: No text extracted from PDF: {original_fn}")
                            current_text_prompt += f"\n\n--- Attached PDF ({original_fn}) contained no extractable text. ---"
                    else:
                        print(f"Warning: Could not retrieve bytes for PDF: {original_fn} from GCS.")
                        current_text_prompt += f"\n\n--- Could not retrieve PDF: {original_fn} from GCS. ---"
                except Exception as e:
                    print(f"Error processing PDF {original_fn}: {type(e).__name__} - {str(e)}")
                    current_text_prompt += f"\n\n--- Error processing PDF: {original_fn}: {str(e)} ---"
        if pdf_texts_appended_count > 0:
             print(f"Finished appending text from {pdf_texts_appended_count} PDF(s) to prompt for {model_type_for_pdf_extraction} model.")
    else:
        print(f"Model {endpoint_id} is not Gemma or Llama. PDF text extraction step skipped. File details (if any) are assumed to be part of text_prompt from app.py.")

    print(f"Final Text prompt for OpenAI SDK (first 200 chars): {current_text_prompt}...")

    match = re.match(
        r"projects/(?P<project_id_from_ep>[^/]+)/locations/(?P<region_from_ep>[^/]+)/endpoints/(?P<endpoint_id_num>[^/]+)",
        endpoint_id
    )
    if not match:
        return None, f"Invalid Vertex AI Endpoint ID format: {endpoint_id}. Expected: projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID_NUM"

    parsed_region = match.group("region_from_ep")
    parsed_project_num = match.group("project_id_from_ep")
    parsed_endpoint_id = match.group("endpoint_id_num")

    base_url = f"https://{parsed_endpoint_id}.{parsed_region}-{parsed_project_num}.prediction.vertexai.goog/v1/{endpoint_id}"
    print(f"Using OpenAI SDK with base_url: {base_url}")

    try:
        creds, auth_project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not creds:
             return None, "Failed to obtain default Google credentials. Ensure ADC is configured."

        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)

        client = openai.OpenAI(base_url=base_url, api_key=creds.token)

        max_tokens_for_openai = 64000
        temperature_for_openai = 0.5

        messages = [{"role": "user", "content": current_text_prompt}]

        print(f"Sending request to OpenAI compatible endpoint. User message (first 100 chars): {messages[0]['content']}...")

        model_response = client.chat.completions.create(
            model="",
            messages=messages,
            temperature=temperature_for_openai,
            max_tokens=max_tokens_for_openai,
            stream=False
        )

        if model_response.choices and len(model_response.choices) > 0 and \
           model_response.choices[0].message and \
           model_response.choices[0].message.content is not None:
            response_text = model_response.choices[0].message.content
            print(f"Received response from OpenAI compatible endpoint (first 100 chars): {response_text}...")
            return response_text, None
        else:
            error_message = "OpenAI SDK call succeeded but response format was unexpected or content was empty."
            if hasattr(model_response, 'model_dump_json'):
                 error_message += f" Full response: {model_response.model_dump_json(indent=2)}"
            else:
                 error_message += f" Raw response object: {str(model_response)}"
            print(f"Error: {error_message}")
            return None, error_message

    except openai.APIError as e:
        error_message_detail = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                err_body = e.response.json()
                error_message_detail = f"{str(e)} - Status: {e.status_code} - Body: {err_body}"
            except:
                error_message_detail = f"{str(e)} - Status: {e.status_code} - Body: {e.response.text[:500]}"
        elif hasattr(e, 'body') and e.body is not None:
            error_message_detail = f"{str(e)} - Body: {e.body}"

        error_message = f"OpenAI API Error invoking Vertex Endpoint {endpoint_id}: {type(e).__name__} - {error_message_detail}"
        print(f"Error: {error_message}")
        return None, error_message
    except google.auth.exceptions.RefreshError as e:
        error_message = f"Google Auth RefreshError: {e}. Check GCloud authentication/ADC setup or service account permissions."
        print(f"Error: {error_message}")
        return None, error_message
    except Exception as e:
        error_message = f"Unexpected error invoking Vertex Endpoint {endpoint_id} via OpenAI SDK: {type(e).__name__} - {str(e)}"
        import traceback
        print(f"Error: {error_message}\n{traceback.format_exc()}")
        return None, error_message


if __name__ == '__main__':
    print("vertex_utils.py loaded. Contains utilities for Vertex AI model and endpoint invocation.")
    # Example Test (Conceptual - uncomment and fill with your details if testing)
    # Ensure GCS_BUCKET_NAME and relevant endpoint IDs are set as environment variables.
    # Also, ensure a test PDF (e.g., test_pdfs/dummy.pdf) exists in your GCS_BUCKET_NAME.

    # TEST_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    # TEST_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1") # Often where endpoints are
    # TEST_GEMMA_ENDPOINT_ID = os.getenv("GEMMA_ENDPOINT_ID")
    # TEST_LLAMA_ENDPOINT_ID = os.getenv("LLAMA_ENDPOINT_ID")
    # GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")
    # TEST_PDF_BLOB_NAME = "test_pdfs/dummy.pdf" # Example path

    # def run_endpoint_test(model_name, endpoint_id_to_test, prompt, files):
    #     if not all([TEST_PROJECT_ID, TEST_LOCATION, endpoint_id_to_test, GCS_BUCKET]):
    #         print(f"\nSkipping {model_name} Endpoint direct test: Ensure GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, {model_name.upper()}_ENDPOINT_ID, and GCS_BUCKET_NAME are set.")
    #         return

    #     print(f"\n--- Testing {model_name} Endpoint ({endpoint_id_to_test}) with OpenAI SDK ---")

    #     # Verify test PDF exists for relevant models
    #     if "gemma" in endpoint_id_to_test.lower() or "llama" in endpoint_id_to_test.lower():
    #         if not files or not any(f.get("content_type") == "application/pdf" for f in files):
    #             print(f"Info: No PDF provided in files list for {model_name} test, though PDF extraction is enabled.")
    #         else: # PDF is in files list, check if it exists
    #             try:
    #                 storage_client_test = storage.Client()
    #                 bucket_obj_test = storage_client_test.bucket(GCS_BUCKET)
    #                 # Assuming the first PDF is the one to check, or adapt if multiple
    #                 pdf_file_to_check = next((f for f in files if f.get("content_type") == "application/pdf"), None)
    #                 if pdf_file_to_check and pdf_file_to_check.get("blob_name"):
    #                     blob_obj_test = bucket_obj_test.blob(pdf_file_to_check["blob_name"])
    #                     if not blob_obj_test.exists():
    #                         print(f"ERROR: Test PDF {pdf_file_to_check['blob_name']} does not exist in bucket {GCS_BUCKET}. Skipping {model_name} endpoint test.")
    #                         return
    #                 elif pdf_file_to_check:
    #                     print(f"Warning: PDF file detail for {model_name} test is missing 'blob_name'. Cannot verify existence.")
    #             except Exception as e_test_setup:
    #                 print(f"Error during test setup for {model_name} endpoint (checking PDF): {e_test_setup}")
    #                 return

    #     response, error = invoke_vertex_endpoint_model(
    #         TEST_PROJECT_ID, TEST_LOCATION, endpoint_id_to_test, prompt, files
    #     )
    #     if error:
    #         print(f"{model_name} Endpoint Test Error: {error}")
    #     else:
    #         print(f"{model_name} Endpoint Test Response:\n{response}")

    # # Common test data
    # test_fnol_prompt = "Analyze the attached FNOL document for potential fraud. What are the key findings?"
    # test_pdf_file_details = []
    # if GCS_BUCKET and TEST_PDF_BLOB_NAME: # Only prepare if GCS_BUCKET is set
    #     test_pdf_file_details = [{
    #         "gcs_uri": f"gs://{GCS_BUCKET}/{TEST_PDF_BLOB_NAME}",
    #         "original_filename": "dummy.pdf",
    #         "content_type": "application/pdf",
    #         "bucket_name": GCS_BUCKET,
    #         "blob_name": TEST_PDF_BLOB_NAME
    #     }]
    # empty_file_details = []

    # # Test Gemma
    # if TEST_GEMMA_ENDPOINT_ID:
    #    run_endpoint_test("Gemma", TEST_GEMMA_ENDPOINT_ID, test_fnol_prompt, test_pdf_file_details)

    # # Test Llama
    # if TEST_LLAMA_ENDPOINT_ID:
    #    run_endpoint_test("Llama with PDF", TEST_LLAMA_ENDPOINT_ID, test_fnol_prompt, test_pdf_file_details)
    #    run_endpoint_test("Llama (no PDF)", TEST_LLAMA_ENDPOINT_ID, "What is the capital of France?", empty_file_details)
    pass
