import os
import google.auth
import google.auth.transport.requests
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from google.genai.types import HttpOptions, Part
import google.api_core.exceptions # For specific API errors
# from google.cloud.aiplatform.generative_models import GenerativeModel, GenerationConfig # Import GenerativeModel and GenerationConfig directly
# from google.cloud.aiplatform_v1.types.content import Part # Content, Tool, FunctionDeclaration not used directly in current Gemini impl.
from google.protobuf import struct_pb2 # For endpoint model instances

# --- Gemini Model Invocation ---
def invoke_gemini_model(project_id, location, model_name, text_prompt, file_details_list):
    """
    Invokes a Gemini multimodal model on Vertex AI.

    Args:
        project_id: GCP Project ID.
        location: GCP Location for Vertex AI.
        model_name: The name of the Gemini model (e.g., "gemini-1.5-pro-preview-0409").
        text_prompt: The textual part of the prompt.
        file_details_list: A list of dictionaries, where each dict contains
                           'gcs_uri' and 'content_type' for a file.
                           Example: [{"gcs_uri": "gs://bucket/file.jpg", "content_type": "image/jpeg"}, ...]
    Returns:
        A tuple (response_text, error_message). response_text is None if an error occurs.
    """
    print(f"Invoking Gemini model: {model_name} in {project_id}/{location}") # Corrected 'project' to 'project_id'
    print(f"Text prompt (first 100 chars): {text_prompt[:100]}...")
    print(f"File details for Gemini: {file_details_list}")

    aiplatform.init(project=project_id, location=location)

    # Ensure model_name does not include "projects/.../locations/.../models/" prefix for GenerativeModel
    if "/" in model_name: # It might be a full resource name if not careful with env var
        model_name_short = model_name.split("/")[-1]
    else:
        model_name_short = model_name

    model = aiplatform.gapic.PredictionServiceClient( # Using gapic directly for more control
         client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    # Or use the higher-level aiplatform.GenerativeModel - let's try that first for simplicity
    # model = aiplatform.GenerativeModel(model_name_short) # This is simpler

    # Construct content parts
    prompt_parts = [Part(text=text_prompt)]
    for file_detail in file_details_list:
        if file_detail.get("gcs_uri") and file_detail.get("content_type"):
            try:
                # For Gemini API, use Part.from_uri for GCS files
                prompt_parts.append(
                    Part.from_uri(uri=file_detail["gcs_uri"], mime_type=file_detail["content_type"])
                )
            except Exception as e:
                print(f"Warning: Could not create Part from URI for {file_detail['gcs_uri']}. Error: {e}")
                # Optionally, append an error message to the prompt or handle differently
                # prompt_parts.append(Part(text=f"[Error processing file: {file_detail.get('original_filename', 'unknown')}]"))
        else:
            print(f"Warning: Missing GCS URI or content type for file: {file_detail.get('original_filename', 'unknown')}")


    # Ensure there's at least one text part; Gemini requires non-empty contents.
    if not any(part.text for part in prompt_parts if hasattr(part, 'text') and part.text):
        if not file_details_list: # No text and no files
             return None, "Cannot send an empty prompt (no text and no files) to Gemini."
        # If there are files but no initial text prompt, some models might still need a placeholder text part.
        # Let's assume the `text_prompt` (which includes SYSTEM_PROMPT) is always present.

    generation_config = {
        "max_output_tokens": 8192, # Max for Gemini 1.5 Pro
        "temperature": 0.4,       # Adjust as needed
        "top_p": 1.0,
        # "top_k": 32 # Not typically used with top_p
    }

    # Safety settings (optional, configure as needed)
    # from google.cloud.aiplatform.generative_models import SafetySetting, HarmCategory # Correct import if used
    # safety_settings = [
    #    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    #    # ... other categories
    # ]

    try:
        # Using the higher-level SDK with direct import:
        vertex_model = GenerativeModel(model_name_short) # Use imported GenerativeModel
        response = vertex_model.generate_content(
            contents=prompt_parts, # Pass the list of Part objects
            generation_config=GenerationConfig(**generation_config), # Use imported GenerationConfig
            # safety_settings=safety_settings,
            # tools=[TOOL_CONFIG]
        )

        # print(f"Raw Gemini Response: {response}") # Can be very verbose

        if not response.candidates:
            block_reason_msg = "Unknown"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name
            safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in response.prompt_feedback.safety_ratings]) if response.prompt_feedback else "N/A"
            error_detail = f"Gemini returned no candidates. Possible block reason: {block_reason_msg}. Safety ratings: [{safety_ratings_str}]"
            print(error_detail)
            return None, error_detail

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            finish_reason_str = candidate.finish_reason.name if candidate.finish_reason else "Unknown"
            safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in candidate.safety_ratings])
            error_detail = f"Gemini response candidate was empty or malformed. Finish reason: {finish_reason_str}. Safety ratings: [{safety_ratings_str}]"
            print(error_detail)
            return None, error_detail

        response_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
        return response_text, None

    except google.api_core.exceptions.GoogleAPIError as e: # More specific
        error_message = f"Google API Error invoking Gemini model {model_name_short}: {e}"
        print(error_message)
        # import traceback; print(traceback.format_exc())
        return None, error_message
    except Exception as e:
        error_message = f"Unexpected error invoking Gemini model {model_name_short}: {str(e)}"
        print(error_message)
        # import traceback; print(traceback.format_exc())
        return None, error_message


# --- Vertex AI Endpoint Model Invocation ---
def invoke_vertex_endpoint_model(project_id, location, endpoint_id, text_prompt, file_details_list):
    """
    Invokes a model deployed on a Vertex AI Endpoint.

    Args:
        project_id: GCP Project ID.
        location: GCP Location of the endpoint.
        endpoint_id: The full ID or the number of the Vertex AI Endpoint.
        text_prompt: The textual part of the prompt.
        file_details_list: A list of dictionaries, where each dict contains
                           'gcs_uri', 'content_type', 'original_filename' for a file.
                           (This function will need to decide how to pass these to the model)
    Returns:
        A tuple (response_text, error_message). response_text is None if an error occurs.
    """
    print(f"Invoking Vertex Endpoint: {endpoint_id} in {project_id}/{location}")
    print(f"Text prompt (first 100 chars): {text_prompt[:100]}...")
    print(f"File details for Endpoint: {file_details_list}")

    aiplatform.init(project=project_id, location=location)

    # Construct the endpoint name if only ID is given
    if not endpoint_id.startswith("projects/"):
        endpoint_name = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
    else:
        endpoint_name = endpoint_id

    endpoint = aiplatform.Endpoint(endpoint_name)

    # The instance format depends heavily on how the custom model was deployed
    # and what input format it expects.
    # Common pattern: {"prompt": "...", "max_tokens": ..., "temperature": ..., "images": ["gs://...", ...]}
    # We'll construct a generic one here. This is the MOST LIKELY part to need adjustment.

    instance_dict = {
        "prompt": text_prompt, # Assuming the endpoint expects a 'prompt' field
        "max_tokens": 4096,    # Example, adjust as needed
        "temperature": 0.5,    # Example, adjust as needed
    }

    # Add file URIs if the model supports it (e.g. under a key like "documents" or "images")
    # This is highly model-dependent. For now, let's assume a "gcs_uris" key.
    if file_details_list:
        instance_dict["gcs_files"] = [
            {"gcs_uri": detail["gcs_uri"], "mime_type": detail["content_type"], "filename": detail.get("original_filename")}
            for detail in file_details_list
        ]

    # Convert dict to google.protobuf.Value for the predict call
    # For some models, just sending the dict might work if the underlying framework handles it.
    # However, to be more explicit or if issues arise, use struct_pb2.Value()
    # instance_payload = struct_pb2.Value()
    # instance_payload.struct_value.update(instance_dict)
    # instances = [instance_payload]

    # Simpler approach: many endpoints accept a list of dicts directly
    instances = [instance_dict]

    try:
        response = endpoint.predict(instances=instances)
        # response.predictions is a list of dicts (or protobufs if not parsed automatically)

        # print(f"Raw Endpoint Response: {response}") # Can be verbose

        if not response.predictions: # Check if predictions list is empty or None
             error_detail = f"Endpoint returned no predictions object or an empty list. Deployed ID: {response.deployed_model_id if hasattr(response, 'deployed_model_id') else 'N/A'}"
             print(error_detail)
             return None, error_detail

        if len(response.predictions) == 0:
            error_detail = f"Endpoint returned an empty predictions list. Deployed ID: {response.deployed_model_id if hasattr(response, 'deployed_model_id') else 'N/A'}"
            print(error_detail)
            return None, error_detail

        # Assuming one instance was sent, so one prediction is expected.
        prediction_content = response.predictions[0]

        response_text = None
        if isinstance(prediction_content, dict):
            # Try common keys for text output
            keys_to_try = ["generated_text", "text", "output_text", "output", "prediction"]
            for key in keys_to_try:
                if key in prediction_content:
                    # Ensure the value is a string or can be meaningfully converted
                    if isinstance(prediction_content[key], str):
                        response_text = prediction_content[key]
                        break
                    elif isinstance(prediction_content[key], list) and len(prediction_content[key]) > 0 and isinstance(prediction_content[key][0], str):
                        # Handle cases where prediction might be a list of strings
                        response_text = prediction_content[key][0]
                        break
            if response_text is None: # If no common key found, stringify the dict
                print(f"Warning: Could not find a standard text key in endpoint prediction dict: {prediction_content}. Using string representation.")
                response_text = str(prediction_content)
        elif isinstance(prediction_content, str):
            response_text = prediction_content
        else:
            error_detail = f"Unexpected prediction format from endpoint. Expected dict or str, got {type(prediction_content)}. Content: {prediction_content}"
            print(error_detail)
            return None, error_detail

        return response_text, None

    except google.api_core.exceptions.GoogleAPIError as e: # More specific
        error_message = f"Google API Error invoking Vertex Endpoint {endpoint_name}: {e}"
        print(error_message)
        # import traceback; print(traceback.format_exc())
        return None, error_message
    except Exception as e:
        error_message = f"Unexpected error invoking Vertex Endpoint {endpoint_name}: {str(e)}"
        print(error_message)
        # import traceback; print(traceback.format_exc())
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
