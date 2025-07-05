import io
from PIL import Image, ExifTags
from PyPDF2 import PdfReader
from google.cloud import storage # To download from GCS if needed

# May need storage_client if downloading from GCS for metadata extraction
# For now, assumes file_content_or_path is accessible directly or as bytes.
# storage_client = storage.Client() # Uncomment if direct GCS download is added here

def extract_image_metadata(file_content_bytes):
    """
    Extracts EXIF metadata from image bytes.
    Args:
        file_content_bytes: Bytes of the image file.
    Returns:
        A dictionary of selected EXIF data or an error message.
    """
    metadata = {}
    try:
        img = Image.open(io.BytesIO(file_content_bytes))
        # Ensure img object is valid before trying to access _getexif
        if not hasattr(img, '_getexif'):
            return {"info": "Not a valid image format or no EXIF data supported by Pillow."}

        exif_data = img._getexif() # Returns a dictionary {tag_id: value}

        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                # Handle bytes values - decode if possible, otherwise store repr
                if isinstance(value, bytes):
                    try:
                        metadata[str(tag_name)] = value.decode('utf-8', errors='replace')
                    except UnicodeDecodeError:
                        metadata[str(tag_name)] = repr(value)
                else:
                    # Limit length of very long string values to avoid overly large metadata dicts
                    if isinstance(value, str) and len(value) > 256:
                         metadata[str(tag_name)] = value[:256] + "..."
                    else:
                        metadata[str(tag_name)] = value

            # Specific useful tags (examples)
            if 'DateTimeOriginal' in metadata:
                metadata['Timestamp (Original)'] = metadata['DateTimeOriginal']
            if 'Make' in metadata and 'Model' in metadata:
                metadata['Camera'] = f"{metadata['Make']} {metadata['Model']}"
            if 'GPSInfo' in metadata: # GPSInfo itself is a dict of GPS tags
                gps_info_dict = {}
                for gps_tag_id, gps_value in metadata['GPSInfo'].items():
                    gps_tag_name = ExifTags.GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info_dict[str(gps_tag_name)] = gps_value
                metadata['GPSInfoProcessed'] = gps_info_dict
        else:
            return {"info": "No EXIF data found."}

        # Add image format and size
        metadata['Format'] = img.format
        metadata['Size'] = img.size # (width, height)

        return {"type": "image", "details": metadata}
    except FileNotFoundError: # Should not happen with BytesIO but good practice
        return {"error": "Image file not found (should not occur with byte stream)."}
    except OSError as e: # Pillow often raises OSError for format issues
        return {"error": f"Could not process image (possibly corrupted or unsupported format): {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error extracting image metadata: {str(e)}"}

def extract_pdf_metadata(file_content_bytes):
    """
    Extracts metadata from PDF bytes.
    Args:
        file_content_bytes: Bytes of the PDF file.
    Returns:
        A dictionary of PDF metadata or an error message.
    """
    metadata = {}
    try:
        pdf_reader = PdfReader(io.BytesIO(file_content_bytes))
        doc_info = pdf_reader.metadata

        if doc_info:
            metadata['Title'] = doc_info.title
            metadata['Author'] = doc_info.author
            metadata['Subject'] = doc_info.subject
            metadata['Creator'] = doc_info.creator # Creating application
            metadata['Producer'] = doc_info.producer # Usually PDF generation library
            metadata['CreationDate'] = doc_info.creation_date.isoformat() if doc_info.creation_date else None
            metadata['ModificationDate'] = doc_info.modification_date.isoformat() if doc_info.modification_date else None

        metadata['Pages'] = len(pdf_reader.pages)
        # Remove None values for cleaner output
        metadata = {k: v for k, v in metadata.items() if v is not None}

        if not metadata:
            return {"info": "No standard metadata fields found in PDF."}

        return {"type": "pdf", "details": metadata}
    except PyPDF2.errors.PdfReadError as e: # Specific PyPDF2 error
        return {"error": f"Could not read PDF (possibly corrupted or encrypted): {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error extracting PDF metadata: {str(e)}"}


def extract_metadata_from_file_bytes(file_content_bytes, content_type):
    """
    Orchestrates metadata extraction based on content type.
    Args:
        file_content_bytes: Bytes of the file.
        content_type: MIME type of the file.
    Returns:
        A dictionary of extracted metadata or an error/info message.
    """
    if content_type in ['image/jpeg', 'image/png', 'image/tiff']:
        return extract_image_metadata(file_content_bytes)
    elif content_type == 'application/pdf':
        return extract_pdf_metadata(file_content_bytes)
    # Add more content types as needed
    # elif content_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
    #     return extract_doc_metadata(file_content_bytes) # Placeholder
    else:
        return {"info": f"Metadata extraction not supported for content type: {content_type}"}


# --- Helper to get file bytes from GCS ---
# This is needed if app.py doesn't want to handle GCS download directly
_storage_client_for_download = None

def get_gcs_file_bytes(bucket_name, blob_name):
    """Downloads a blob from GCS and returns its content as bytes."""
    global _storage_client_for_download
    if _storage_client_for_download is None:
        _storage_client_for_download = storage.Client()

    try:
        bucket = _storage_client_for_download.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # Add a check if blob exists before download attempt
        if not blob.exists():
            print(f"Blob gs://{bucket_name}/{blob_name} not found for download.")
            return None
        file_bytes = blob.download_as_bytes()
        return file_bytes
    except google.cloud.exceptions.NotFound:
        print(f"GCS Blob gs://{bucket_name}/{blob_name} not found during download attempt.")
        return None
    except google.cloud.exceptions.GoogleCloudError as e:
        print(f"GCS API Error downloading gs://{bucket_name}/{blob_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error downloading gs://{bucket_name}/{blob_name}: {e}")
        return None

if __name__ == '__main__':
    print("metadata_utils.py loaded. Contains utilities for file metadata extraction.")
    # Example usage would require actual file bytes.
    # For testing, you could read a local image/pdf file into bytes:
    #
    # try:
    #     with open("local_test_image.jpg", "rb") as f:
    #         image_bytes = f.read()
    #     meta = extract_metadata_from_file_bytes(image_bytes, "image/jpeg")
    #     print("Image Meta:", meta)
    # except FileNotFoundError:
    #     print("Skipping image metadata test: local_test_image.jpg not found.")
    #
    # try:
    #     with open("local_test_doc.pdf", "rb") as f:
    #         pdf_bytes = f.read()
    #     meta = extract_metadata_from_file_bytes(pdf_bytes, "application/pdf")
    #     print("PDF Meta:", meta)
    # except FileNotFoundError:
    #     print("Skipping PDF metadata test: local_test_doc.pdf not found.")

    # Test GCS download (requires bucket and blob to exist and ADC to be set up)
    # test_gcs_bucket = os.getenv("GCS_BUCKET_NAME_FOR_TESTING")
    # test_gcs_blob = "fnol_uploads/your_test_file.jpg" # Replace with an actual blob in your test bucket
    # if test_gcs_bucket:
    #     print(f"\nAttempting to download gs://{test_gcs_bucket}/{test_gcs_blob} for metadata test...")
    #     file_b = get_gcs_file_bytes(test_gcs_bucket, test_gcs_blob)
    #     if file_b:
    #         print(f"Downloaded {len(file_b)} bytes. Attempting metadata extraction...")
    #         # Assuming it's a JPEG for this test path
    #         meta = extract_metadata_from_file_bytes(file_b, "image/jpeg")
    #         print("GCS File Meta:", meta)
    #     else:
    #         print(f"Could not download gs://{test_gcs_bucket}/{test_gcs_blob}")
    # else:
    #     print("GCS_BUCKET_NAME_FOR_TESTING not set, skipping GCS download test.")
