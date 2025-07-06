import os
import datetime
from google.cloud import storage
from werkzeug.utils import secure_filename

# Initialize GCS client globally or within functions as preferred.
# Global initialization is fine if GOOGLE_APPLICATION_CREDENTIALS is set
# or running in an environment with ADC.
storage_client = storage.Client()

def upload_to_gcs(file_storage_object, bucket_name, destination_blob_folder="uploads"):
    """
    Uploads a file object to Google Cloud Storage.

    Args:
        file_storage_object: The FileStorage object from Flask request.files.
        bucket_name: The name of the GCS bucket.
        destination_blob_folder: The folder within the bucket to upload to.

    Returns:
        A dictionary containing the GCS URI, original filename, content type,
        and blob name, or None if upload fails.
    """
    if not file_storage_object or not file_storage_object.filename:
        return None

    original_filename = secure_filename(file_storage_object.filename)
    content_type = file_storage_object.mimetype

    # Create a unique blob name to avoid overwrites
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    unique_blob_name = f"{destination_blob_folder}/{timestamp}_{original_filename}"

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(unique_blob_name)

        # Rewind the stream to the beginning before uploading
        file_storage_object.seek(0)

        blob.upload_from_file(file_storage_object, content_type=content_type)

        gcs_uri = f"gs://{bucket_name}/{unique_blob_name}"

        print(f"Successfully uploaded {original_filename} to {gcs_uri}")
        return {
            "gcs_uri": gcs_uri,
            "original_filename": original_filename,
            "content_type": content_type,
            "blob_name": unique_blob_name,
            "bucket_name": bucket_name
        }
    except google.cloud.exceptions.GoogleCloudError as e:
        print(f"GCS API Error during upload of {original_filename}: {e}")
        # import traceback; print(traceback.format_exc()) # For detailed debugging
        return None
    except Exception as e:
        print(f"Unexpected error during upload of {original_filename} to GCS: {e}")
        # import traceback; print(traceback.format_exc())
        return None

def generate_signed_url(bucket_name, blob_name, expiration_minutes=15):
    """
    Generates a v4 signed URL for downloading a blob.
    The service account needs 'Service Account Token Creator' role if using service account.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=expiration_minutes),
            method="GET",
        )
        return url
    except google.cloud.exceptions.GoogleCloudError as e:
        print(f"GCS API Error generating signed URL for {blob_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error generating signed URL for {blob_name}: {e}")
        return None

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        print(f"Blob {blob_name} deleted.")
        return True
    except google.cloud.exceptions.GoogleCloudError as e:
        print(f"GCS API Error deleting blob {blob_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error deleting blob {blob_name}: {e}")
        return False

# Example of how to get a blob's metadata if needed directly
# def get_blob_metadata(bucket_name, blob_name):
#     try:
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.get_blob(blob_name)
#         if blob:
#             return blob # The Blob object itself contains metadata properties
#         return None
#     except google.cloud.exceptions.GoogleCloudError as e:
#         print(f"GCS API Error getting metadata for {blob_name}: {e}")
#         return None
#     except Exception as e:
#         print(f"Unexpected error getting metadata for {blob_name}: {e}")
#         return None

if __name__ == '__main__':
    # This section is for testing the module directly.
    # You'd need to have GOOGLE_APPLICATION_CREDENTIALS set and a bucket.
    # And a dummy file object.
    print("gcs_utils.py loaded. Contains utilities for GCS operations.")
    # Example usage (requires a running Flask app context or mock FileStorage):
    # from werkzeug.datastructures import FileStorage
    # import io
    #
    # test_bucket = os.getenv("GCS_BUCKET_NAME_FOR_TESTING") # Set this env var for testing
    # if test_bucket:
    #     # Create a dummy FileStorage object
    #     dummy_content = b"This is a test file content."
    #     dummy_file = FileStorage(
    #         stream=io.BytesIO(dummy_content),
    #         filename="test_upload.txt",
    #         content_type="text/plain"
    #     )
    #     upload_result = upload_to_gcs(dummy_file, test_bucket)
    #     if upload_result:
    #         print(f"Test upload successful: {upload_result}")
    #         signed_url = generate_signed_url(upload_result['bucket_name'], upload_result['blob_name'])
    #         if signed_url:
    #             print(f"Generated signed URL: {signed_url}")
    #
    #         # Test delete (optional, be careful)
    #         # if delete_blob(upload_result['bucket_name'], upload_result['blob_name']):
    #         #     print(f"Successfully deleted {upload_result['blob_name']} after test.")
    # else:
    #     print("GCS_BUCKET_NAME_FOR_TESTING environment variable not set. Skipping direct test.")
