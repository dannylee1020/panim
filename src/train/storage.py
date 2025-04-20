import json
from google.cloud import storage
from google.cloud.exceptions import NotFound
import fire
from typing import Dict, Any
import os
from pathlib import Path

class GCSJsonStorage:
    """
    A class to handle uploading and downloading JSON files to/from Google Cloud Storage.
    """

    def __init__(self, bucket_name: str, project: str | None = None):
        self.client = storage.Client(project=project)
        self.bucket_name = bucket_name
        try:
            self.bucket = self.client.get_bucket(bucket_name)
        except NotFound:
            print(f"Warning: Bucket '{bucket_name}' not found or access denied.")

    def upload(self, data: Dict[str, Any], blob_name: str) -> None:
        """
        Uploads a Python dictionary as a JSON file to GCS.

        Args:
            data: The dictionary data to upload.
            blob_name: The desired name for the robject (file path) in the GCS bucket.

        Raises:
            Exception: If the bucket was not found during initialization.
            google.cloud.exceptions.GoogleCloudError: For GCS API errors.
        """
        if not self.bucket:
             raise Exception(f"Bucket '{self.bucket_name}' not initialized or accessible.")

        blob = self.bucket.blob(blob_name)
        json_data = json.dumps(data, indent=2) # Use indent for readability in GCS console
        blob.upload_from_string(json_data, content_type='application/json')
        print(f"Successfully uploaded JSON data to gs://{self.bucket_name}/{blob_name}")

    def download(self, blob_name: str) -> Dict[str, Any] | None:
        """
        Downloads a JSON file from GCS and parses it into a Python dictionary.

        Args:
            blob_name: The name of the object (file path) in the GCS bucket.

        Returns:
            The parsed dictionary data, or None if the blob is not found.

        Raises:
            Exception: If the bucket was not found during initialization.
            json.JSONDecodeError: If the downloaded content is not valid JSON.
            google.cloud.exceptions.GoogleCloudError: For other GCS API errors.
        """
        if not self.bucket:
             raise Exception(f"Bucket '{self.bucket_name}' not initialized or accessible.")

        blob = self.bucket.blob(blob_name)
        try:
            json_data = blob.download_as_text()
            data = json.loads(json_data)
            print(f"Successfully downloaded JSON data fom gs://{self.bucket_name}/{blob_name}")
            return data
        except NotFound:
            print(f"Error: Blob '{blob_name}' not found in bucket '{self.bucket_name}'.")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON from gs://{self.bucket_name}/{blob_name}. Error: {e}")
            raise # Re-raise the error as it indicates corrupted data or wrong file type


def upload_directory_to_gcs(local_dir: str, bucket_name: str, project: str | None = None):
    """
    Scans a local directory for JSON files and uploads them to a GCS bucket,
    preserving the relative directory structure.

    Args:
        local_dir: The path to the local directory to scan (e.g., 'inst').
        bucket_name: The name of the target GCS bucket.
        project: The Google Cloud project ID (optional, inferred if None).
    """
    storage_client = GCSJsonStorage(bucket_name=bucket_name, project=project)
    base_path = Path(local_dir).resolve()
    files_uploaded = 0
    files_failed = 0

    print(f"Starting upload from '{local_dir}' to gs://{bucket_name}/ ...")

    for root, _, files in os.walk(local_dir):
        for filename in files:
            if filename.lower().endswith('.json'):
                local_filepath = (Path(root) / filename).resolve()
                try:
                    relative_path = local_filepath.relative_to(base_path)
                except ValueError as e:
                    print(f"  Error calculating relative path for {local_filepath} against base {base_path}: {e}")
                    files_failed += 1
                    continue

                # Ensure GCS blob name uses forward slashes and is a string
                blob_name = str(relative_path.as_posix())

                print(f"  Processing: {local_filepath} -> gs://{bucket_name}/{blob_name}")

                try:
                    with open(local_filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    storage_client.upload(data=data, blob_name=blob_name)
                    files_uploaded += 1
                except FileNotFoundError:
                    print(f"  Error: Local file not found (should not happen with os.walk): {local_filepath}")
                    files_failed += 1
                except json.JSONDecodeError as e:
                    print(f"  Error: Failed to decode JSON from {local_filepath}. Error: {e}")
                    files_failed += 1
                except Exception as e:
                    print(f"  Error: Failed to upload {local_filepath} to {blob_name}. Error: {e}")
                    files_failed += 1

    print(f"\nUpload complete.")
    print(f"  Successfully uploaded: {files_uploaded} files.")
    print(f"  Failed uploads: {files_failed} files.")

if __name__ == "__main__":
    fire.Fire(upload_directory_to_gcs)
