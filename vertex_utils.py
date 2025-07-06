import os
import google.auth # Keep for general auth, though vertexai.init might handle it
import google.auth.transport.requests # Keep for general auth
import google.api_core.exceptions # For specific API errors
import re # For parsing endpoint string

# New imports for Gemini using vertexai namespace
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
# from vertexai.generative_models import HarmCategory, SafetySetting # Uncomment if using safety settings

# Keep aiplatform for Endpoint interactions if they are separate
from google.cloud import aiplatform as old_aiplatform # Alias to avoid confusion if still needed for Endpoint
from google.protobuf import struct_pb2 # For endpoint model instances

# --- Gemini Model Invocation (using new vertexai SDK) ---
def invoke_gemini_model(project_id, location, model_name, text_prompt, file_details_list):
    """
    Invokes a Gemini multimodal model on Vertex AI using the new vertexai SDK.

    Args:
        project_id: GCP Project ID.
        location: GCP Location for Vertex AI.
        model_name: The name of the Gemini model (e.g., "gemini-1.5-pro-preview-0409").
        text_prompt: The textual part of the prompt.
        file_details_list: A list of dictionaries, where each dict contains
                           'gcs_uri' and 'content_type' for a file.
    Returns:
        A tuple (response_text, error_message). response_text is None if an error occurs.
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
            if not any(hasattr(part, 'file_data') for part in prompt_parts): # Check if any part is a file part
                 return None, "Cannot send an empty prompt (no text and no files) to Gemini."

        generation_config_dict = {
            "max_output_tokens": 8192,
            "temperature": 0.4,
            "top_p": 1.0,
        }
        gen_config_instance = GenerationConfig(**generation_config_dict)

        response = model.generate_content(
            contents=prompt_parts,
            generation_config=gen_config_instance,
        )

        if not response.candidates:
            block_reason_msg = "Unknown"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = response.prompt_feedback.block_reason_message or str(response.prompt_feedback.block_reason)
            safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in response.prompt_feedback.safety_ratings]) if response.prompt_feedback else "N/A"
            error_detail = f"Gemini (vertexai SDK) returned no candidates. Possible block reason: {block_reason_msg}. Safety ratings: [{safety_ratings_str}]"
            print(error_detail)
            return None, error_detail

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            finish_reason_str = str(candidate.finish_reason) if candidate.finish_reason else "Unknown"
            safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability}" for rating in candidate.safety_ratings])
            error_detail = f"Gemini (vertexai SDK) response candidate was empty or malformed. Finish reason: {finish_reason_str}. Safety ratings: [{safety_ratings_str}]"
            print(error_detail)
            return None, error_detail

        response_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text is not None])
        return response_text, None

    except google.api_core.exceptions.GoogleAPIError as e:
        error_message = f"Google API Error invoking Gemini model (vertexai SDK) {model_name_short}: {e}"
        print(error_message)
        return None, error_message
    except Exception as e:
        error_message = f"Unexpected error invoking Gemini model (vertexai SDK) {model_name_short}: {str(e)}"
        print(error_message)
        # import traceback; print(traceback.format_exc()) # For more detailed debugging
        return None, error_message


# --- Vertex AI Endpoint Model Invocation (adapted from Gemma notebook) ---
def invoke_vertex_endpoint_model(project_id, location, endpoint_id, text_prompt, file_details_list):
    """
    Invokes a model deployed on a Vertex AI Endpoint.
    Uses aiplatform.Endpoint client, which is common for Model Garden deployments like Gemma.
    The `text_prompt` here is assumed to be the fully constructed prompt including any
    file context (like GCS URIs and metadata) as a single string.
    `file_details_list` is logged but not directly used in the instances payload here,
    as standard text model endpoints expect the prompt to contain all context.
    """
    print(f"Invoking Vertex Endpoint (using aiplatform.Endpoint): {endpoint_id}")
    print(f"Text prompt for endpoint (first 100 chars): {text_prompt[:100]}...")
    if file_details_list:
        print(f"Note: File details are part of text_prompt for this endpoint type. Details: {file_details_list}")

    # The endpoint_id parameter is expected to be the full resource name.
    match = re.match(
        r"projects/(?P<project_id_num>[^/]+)/locations/(?P<region>[^/]+)/endpoints/(?P<endpoint_id_num>[^/]+)",
        endpoint_id
    )
    if not match:
        return None, f"Invalid Vertex AI Endpoint ID format: {endpoint_id}. Expected projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID_NUM"

    parsed_project_id_num = match.group("project_id_num")
    parsed_region = match.group("region")

    # Initialize aiplatform with project and location derived from the endpoint string
    old_aiplatform.init(project=parsed_project_id_num, location=parsed_region)

    endpoint_client = old_aiplatform.Endpoint(endpoint_name=endpoint_id) # Pass full resource name

    # Construct the payload (instances) based on Gemma notebook examples.
    # Gemma text models often expect a list of prompt strings as instances.
    # Our `text_prompt` is already the fully constructed prompt string.
    instances_payload = [text_prompt]

    # Parameters for generation, common for text models
    # These parameters are from the linked Gemma notebook.
    parameters_payload = {
        "temperature": 0.2,
        "max_output_tokens": 1024, # Gemma notebook uses 1024
        "top_p": 0.8,
        "top_k": 40
    }

    print(f"Sending payload to aiplatform.Endpoint {endpoint_id}: instances='{str(instances_payload)[:200]}...', parameters={parameters_payload}")

    try:
        response = endpoint_client.predict(
            instances=instances_payload,
            parameters=parameters_payload
        )

        # print(f"Raw Endpoint Response (aiplatform.Endpoint): {response}")

        if not response.predictions:
             error_detail = f"Endpoint ({endpoint_id}) returned no predictions object or an empty list. Deployed Model ID: {response.deployed_model_id if hasattr(response, 'deployed_model_id') else 'N/A'}"
             print(error_detail)
             return None, error_detail

        if len(response.predictions) == 0:
            error_detail = f"Endpoint ({endpoint_id}) returned an empty predictions list. Deployed Model ID: {response.deployed_model_id if hasattr(response, 'deployed_model_id') else 'N/A'}"
            print(error_detail)
            return None, error_detail

        prediction_content = response.predictions[0]

        response_text = None
        if isinstance(prediction_content, str):
            response_text = prediction_content
        elif isinstance(prediction_content, dict):
            # Standardized Model Garden containers often return a dict with a "generated_text" or similar key.
            # The Gemma notebook example shows it might be directly in prediction_content if it's a string,
            # or in prediction_content['generated_text']
            keys_to_try = ["generated_text", "text", "output", "outputs", "prediction"]
            for key in keys_to_try:
                if key in prediction_content:
                    if isinstance(prediction_content[key], str):
                        response_text = prediction_content[key]
                        break
                    elif isinstance(prediction_content[key], list) and len(prediction_content[key]) > 0 and isinstance(prediction_content[key][0], str):
                        response_text = prediction_content[key][0]
                        break
            if response_text is None:
                print(f"Warning: Could not find a standard text key in endpoint prediction dict: {prediction_content}. Using string representation.")
                response_text = str(prediction_content) # Fallback
        elif isinstance(prediction_content, list) and len(prediction_content) > 0 and isinstance(prediction_content[0], str):
            # Some models might return a list of strings (e.g. if multiple candidates were requested, though not typical for predict())
            response_text = prediction_content[0]
        else:
            error_detail = f"Unexpected prediction format from endpoint. Expected str, dict, or list of str. Got {type(prediction_content)}. Content: {str(prediction_content)[:200]}..."
            print(error_detail)
            return None, error_detail

        return response_text, None

    except google.api_core.exceptions.GoogleAPIError as e:
        error_message = f"Google API Error invoking Vertex Endpoint {endpoint_id}: {e}"
        print(error_message)
        # This will now include the 501/404 if it still occurs, providing context.
        return None, error_message
    except Exception as e:
        error_message = f"Unexpected error invoking Vertex Endpoint {endpoint_id}: {str(e)}"
        print(error_message)
        # import traceback; print(traceback.format_exc()) # For more detailed debugging
        return None, error_message


if __name__ == '__main__':
    print("vertex_utils.py loaded. Contains utilities for Vertex AI model and endpoint invocation.")
    # To test these functions directly, you would need:
    # 1. GOOGLE_APPLICATION_CREDENTIALS set up or gcloud auth application-default login.
    # 2. Environment variables for GCP_PROJECT_ID, GCP_LOCATION.
    # 3. For Gemini: A valid GEMINI_PRO_MODEL_NAME (e.g., "gemini-1.5-pro-preview-0409").
    #    Some GCS files uploaded and their gs:// URIs.
    # 4. For Endpoint: A deployed Vertex AI Endpoint ID (LLAMA_ENDPOINT_ID or GEMMA_ENDPOINT_ID).
    #    The endpoint must be configured to accept the payload format this script sends.

    # Example Test (Conceptual - uncomment and fill with your details if testing)
    # TEST_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    # TEST_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    # TEST_GEMINI_MODEL = os.getenv("GEMINI_PRO_MODEL_NAME") # e.g. gemini-1.5-pro-preview-0409
    # TEST_LLAMA_ENDPOINT = os.getenv("LLAMA_ENDPOINT_ID")

    # test_text_prompt = "System: You are a helpful assistant.\nUser: What is the capital of France? Also, look at this image."
    # test_file_details = [
    #     {"gcs_uri": "gs://your-gcs-bucket-name/path/to/your/image.jpg", "content_type": "image/jpeg", "original_filename": "image.jpg"}
    # ]

    # if TEST_PROJECT_ID and TEST_LOCATION and TEST_GEMINI_MODEL and test_file_details[0]["gcs_uri"] != "gs://your-gcs-bucket-name/path/to/your/image.jpg":
    #     print(f"\n--- Testing Gemini ({TEST_GEMINI_MODEL}) ---")
    #     gemini_response, gemini_error = invoke_gemini_model(
    #         TEST_PROJECT_ID, TEST_LOCATION, TEST_GEMINI_MODEL, test_text_prompt, test_file_details
    #     )
    #     if gemini_error:
    #         print(f"Gemini Test Error: {gemini_error}")
    #     else:
    #         print(f"Gemini Test Response:\n{gemini_response}")
    # else:
    #     print("\nSkipping Gemini direct test: Ensure GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GEMINI_PRO_MODEL_NAME are set and test_file_details GCS URI is updated.")

    # if TEST_PROJECT_ID and TEST_LOCATION and TEST_LLAMA_ENDPOINT:
    #     print(f"\n--- Testing Llama Endpoint ({TEST_LLAMA_ENDPOINT}) ---")
    #     # Note: Llama endpoint might not support GCS URIs in the same way as Gemini.
    #     # The payload for Llama might only take the text prompt, or require a different format for files.
    #     # This test assumes it can handle the "gcs_files" key in the instance.
    #     llama_response, llama_error = invoke_vertex_endpoint_model(
    #         TEST_PROJECT_ID, TEST_LOCATION, TEST_LLAMA_ENDPOINT, test_text_prompt, test_file_details
    #     )
    #     if llama_error:
    #         print(f"Llama Endpoint Test Error: {llama_error}")
    #     else:
    #         print(f"Llama Endpoint Test Response:\n{llama_response}")
    # else:
    #     print("\nSkipping Llama Endpoint direct test: Ensure GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, LLAMA_ENDPOINT_ID are set.")

    pass
