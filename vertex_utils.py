import os
import google.auth # Keep for general auth, though vertexai.init might handle it
import google.auth.transport.requests # Keep for general auth
import google.api_core.exceptions # For specific API errors

# New imports for Gemini using vertexai namespace
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
# from vertexai.generative_models import HarmCategory, SafetySetting # Uncomment if using safety settings

# Keep aiplatform for Endpoint interactions if they are separate
from google.cloud import aiplatform as old_aiplatform # Alias to avoid confusion if still needed for Endpoint
from google.protobuf import struct_pb2 # For endpoint model instances
import vertexai.preview

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

        # Ensure model_name does not include "projects/.../locations/.../models/" prefix
        # The new SDK's GenerativeModel.from_pretrained expects just the model ID string.
        if "/" in model_name:
            model_name_short = model_name.split("/")[-1]
        else:
            model_name_short = model_name

        model = GenerativeModel(model_name_short) # Use GenerativeModel from vertexai.generative_models

        # Construct content parts
        # Part is now from vertexai.generative_models
        prompt_parts = [Part.from_text(text_prompt)] # Use Part.from_text for text parts
        for file_detail in file_details_list:
            if file_detail.get("gcs_uri") and file_detail.get("content_type"):
                try:
                    prompt_parts.append(
                        Part.from_uri(uri=file_detail["gcs_uri"], mime_type=file_detail["content_type"])
                    )
                except Exception as e:
                    print(f"Warning: Could not create Part from URI for {file_detail['gcs_uri']}. Error: {e}")
            # Removed the 'else' block associated with the for-loop as it was potentially confusing/misplaced.
            # The warning for missing GCS URI or content type is now handled by the `if` condition inside the loop.

        # Ensure there's at least one text part; Gemini requires non-empty contents.
        # This validation should be inside the try block as it depends on prompt_parts.
        if not any(hasattr(part, 'text') and part.text for part in prompt_parts): # Check if any part has actual text
            if not any(hasattr(part, 'file_data') for part in prompt_parts): # Check if any part is a file part
                 return None, "Cannot send an empty prompt (no text and no files) to Gemini."
            # If there are files but no initial text prompt, it might be okay for some models/use-cases.
            # The current construction always adds a text_prompt part first.

        generation_config_dict = {
            "max_output_tokens": 8192,
            "temperature": 0.4,
            "top_p": 1.0,
        }

        gen_config_instance = GenerationConfig(**generation_config_dict)

        response = model.generate_content(
            contents=prompt_parts,
            generation_config=gen_config_instance,
            # safety_settings=safety_settings, # Define if needed
            # tools=tools # Define if needed
        )

        # print(f"Raw Gemini Response (vertexai SDK): {response}")

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

    # For custom endpoints, google.cloud.aiplatform (aliased as old_aiplatform) might still be the way
    # unless the new vertexai SDK also provides a unified Endpoint client.
    # Let's assume old_aiplatform.Endpoint is still valid for now.

    # Parse the full endpoint_id name to get project, location, and endpoint number
    import re
    match = re.match(
        r"projects/(?P<project_id_num>[^/]+)/locations/(?P<region>[^/]+)/endpoints/(?P<endpoint_id_num>[^/]+)",
        endpoint_id # This is the full resource name from env var
    )
    if not match:
        return None, f"Invalid Vertex AI Endpoint ID format: {endpoint_id}. Expected format: projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID_NUM"

    parsed_project_id_num = match.group("project_id_num")
    parsed_region = match.group("region")
    parsed_endpoint_id_num = match.group("endpoint_id_num")

    # Initialize with the parsed project and location from the endpoint string itself,
    # as these are authoritative for the endpoint.
    old_aiplatform.init(project=parsed_project_id_num, location=parsed_region)

    endpoint_name = endpoint_id # endpoint_name is the full resource path

    # The instance format depends heavily on how the custom model was deployed
    # and what input format it expects.
    # Common pattern: {"prompt": "...", "max_tokens": ..., "temperature": ..., "images": ["gs://...", ...]}
    # We'll construct a generic one here. This is the MOST LIKELY part to need adjustment.

    # Determine the correct input key based on the model type.
    # For Gemma models deployed with Saxml, the key is "text_batch".
    # For Llama models, this needs to be verified by the user/docs.
    if "gemma" in endpoint_id.lower():
        input_key = "text_batch"
        print(f"Using input key 'text_batch' for Gemma model {endpoint_id}.")
    elif "llama" in endpoint_id.lower():
        input_key = "prompt" # Placeholder - VERIFY THIS for Llama models
        print(f"Warning: Using placeholder input key 'prompt' for Llama model {endpoint_id}. If issues persist (e.g., prompt in response), this key needs to be verified and potentially changed to the model-specific key (e.g., 'inputs', 'instances', 'text_batch', etc.).")
    else:
        # Default for unknown model types, maintain original behavior before this change.
        input_key = "prompt"
        print(f"Warning: Unknown model type for endpoint {endpoint_id}. Defaulting to input key 'prompt'. This may need adjustment.")

    instance_dict = {
        input_key: text_prompt,
        "max_tokens": 30000,    # Example, adjust as needed
        "temperature": 0.5,    # Example, adjust as needed
    }

    # Add file URIs if the model supports it (e.g. under a key like "documents" or "images")
    # This is highly model-dependent. For now, let's assume a "gcs_uris" key.
    #if file_details_list:
    #    instance_dict["gcs_files"] = [
    #        {"gcs_uri": detail["gcs_uri"], "mime_type": detail["content_type"], "filename": detail.get("original_filename")}
    #        for detail in file_details_list
    #    ]
    #
    # Convert dict to google.protobuf.Value for the predict call
    # For some models, just sending the dict might work if the underlying framework handles it.
    # However, to be more explicit or if issues arise, use struct_pb2.Value()
    # instance_payload = struct_pb2.Value()
    # instance_payload.struct_value.update(instance_dict)
    # instances = [instance_payload]

    # Simpler approach: many endpoints accept a list of dicts directly
    instances = [instance_dict]

    # Dedicated domain construction
    sample_endpoint = f"{parsed_region}-aiplatform.googleapis.com"
    dedicated_domain = f"{parsed_endpoint_id_num}.{parsed_region}-{parsed_project_id_num}.prediction.vertexai.goog"
    print(f"Using dedicated domain for endpoint: {dedicated_domain}")
    aip_endpoint_name = f"projects/{parsed_project_id_num}/locations/{parsed_region}/endpoints/{parsed_endpoint_id_num}"
    endpointes = old_aiplatform.Endpoint(aip_endpoint_name)

    try:
        # Use PredictionServiceClient with custom endpoint
        #client_options = {"api_endpoint": sample_endpoint}
        #client = old_aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        response = endpointes.predict(instances=instances, use_dedicated_endpoint=True)

        # print(f"Raw Endpoint Response: {response}")

        if not response.predictions:
             error_detail = f"Endpoint returned no predictions object or an empty list. Deployed ID: {response.deployed_model_id if hasattr(response, 'deployed_model_id') else 'N/A'}"
             print(error_detail)
             return None, error_detail

        if len(response.predictions) == 0:
            error_detail = f"Endpoint returned an empty predictions list. Deployed ID: {response.deployed_model_id if hasattr(response, 'deployed_model_id') else 'N/A'}"
            print(error_detail)
            return None, error_detail

        prediction_content = response.predictions[0]
        ai_error_message = None
        response_text = None
        keys_to_try = ["generated_text", "text", "output_text", "output", "prediction", "completion", "outputs"] # Added "completion" and "outputs"
        input_like_key_substrings = ["prompt", "input", "query", "context", "text_batch", "echo", "instance"]

        if isinstance(prediction_content, dict):
            found_by_key = False
            # 1. Try common known keys
            for key in keys_to_try:
                if key in prediction_content:
                    value = prediction_content[key]
                    if isinstance(value, str):
                        response_text = value
                        found_by_key = True
                        break
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                        response_text = value[0] # Take the first string if it's a list of strings
                        found_by_key = True
                        break

            if not found_by_key:
                # 2. If not found, iterate through dict values, preferring non-input-like keys
                candidate_strings = {}
                other_strings = {}
                for k, v in prediction_content.items():
                    if isinstance(v, str):
                        is_input_like = any(substr in k.lower() for substr in input_like_key_substrings)
                        if not is_input_like:
                            candidate_strings[k] = v
                        else:
                            other_strings[k] = v

                if candidate_strings:
                    response_text = max(candidate_strings.values(), key=len) # Longest from non-input-like
                    print(f"Info: Used heuristic to extract text from dict. Key chosen by length from non-input-like keys: '{max(candidate_strings, key=lambda k_lambda: len(candidate_strings[k_lambda]))}'.")
                elif other_strings: # Only if no non-input-like strings were found
                    response_text = max(other_strings.values(), key=len) # Longest from input-like if no other choice
                    print(f"Warning: Extracted text from a key that might be an input echo ('{max(other_strings, key=lambda k_lambda: len(other_strings[k_lambda]))}'). This might include the prompt.")

            # 3. Check for known nested structures (e.g., HuggingFace Transformers format)
            if response_text is None and "generated_text" in prediction_content.get("predictions", [{}])[0]: # Common for HF
                 if isinstance(prediction_content["predictions"], list) and len(prediction_content["predictions"]) > 0:
                     if isinstance(prediction_content["predictions"][0], dict) and "generated_text" in prediction_content["predictions"][0]:
                        response_text = prediction_content["predictions"][0]["generated_text"]
                        print("Info: Extracted text from nested 'predictions[0].generated_text'.")

            if response_text is None: # If still no text found
                ai_error_message = f"Failed to parse AI response. No standard text key or parsable string value found in the prediction dictionary. Content (first 200 chars): {str(prediction_content)[:200]}"
                print(f"Error: {ai_error_message}")
                return None, ai_error_message

        elif isinstance(prediction_content, str):
            response_text = prediction_content

        elif isinstance(prediction_content, list): # Handles cases like a list of strings or list of dicts
            processed_parts = []
            if not prediction_content: # Empty list
                ai_error_message = f"Prediction list was empty. Content: {str(prediction_content)[:200]}"
                print(f"Error: {ai_error_message}")
                return None, ai_error_message

            for item_idx, item in enumerate(prediction_content):
                item_text_segment = None
                if isinstance(item, str):
                    item_text_segment = item
                elif isinstance(item, dict):
                    # Try common keys
                    for key in keys_to_try:
                        if key in item and isinstance(item[key], str):
                            item_text_segment = item[key]
                            break
                    # If not by common key, try non-input-like heuristic
                    if item_text_segment is None:
                        item_candidate_strings = {k:v for k,v in item.items() if isinstance(v,str) and not any(substr in k.lower() for substr in input_like_key_substrings)}
                        if item_candidate_strings:
                            item_text_segment = max(item_candidate_strings.values(), key=len)

                if item_text_segment is not None:
                    processed_parts.append(item_text_segment)
                else:
                    # If item is a dict and still no text, or other type, log and skip or convert
                    print(f"Warning: Could not extract specific text from item {item_idx} (type: {type(item)}) in prediction list. Item (first 100 chars): {str(item)[:100]}")

            if processed_parts:
                response_text = "\n".join(processed_parts) # Or handle as a list if appropriate for caller
            else:
                ai_error_message = f"Prediction list contained no parsable text segments. Content (first 200 chars): {str(prediction_content)[:200]}"
                print(f"Error: {ai_error_message}")
                return None, ai_error_message
        else:
            ai_error_message = f"Unexpected prediction format from endpoint. Expected dict, str, or list, got {type(prediction_content)}. Content (first 200 chars): {str(prediction_content)[:200]}"
            print(f"Error: {ai_error_message}")
            return None, ai_error_message

        return response_text, None

    except google.api_core.exceptions.GoogleAPIError as e:
        error_message = f"Google API Error invoking Vertex Endpoint {endpoint_name} via {dedicated_domain}: {e}"
        print(error_message)
        return None, error_message
    except Exception as e:
        error_message = f"Unexpected error invoking Vertex Endpoint {endpoint_name} via {dedicated_domain}: {str(e)}"
        print(error_message)
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
