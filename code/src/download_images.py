import os
from google.cloud import storage
from pathlib import Path
from logging import Logger
from config import (
    logger,
    BLUE,
    YELLOW,
    GREEN,
    RED,
    COLOR_END,
    BUCKET_NAME,
    GCS_FOLDER,
    LOCAL_DESTINATION,
    BILLING_PROJECT_ID,
)


def download_folder_from_requester_pays_bucket(
    billing_project_id: str,
    bucket_name: str,
    gcs_folder: str,
    local_destination: str,
    logger: Logger,
):
    """Downloads all files from a 'folder' in a Requester Pays bucket."""

    print(f"{YELLOW}bucket name:{COLOR_END} {GREEN}{bucket_name}{COLOR_END}")
    logger.info(f"bucket name: '{bucket_name}'")

    # initialize the client with billing project.
    storage_client = storage.Client(project=billing_project_id)

    # get the bucket by specifying the billing project.
    bucket = storage_client.bucket(bucket_name, user_project=billing_project_id)

    # list all blobs that start with the folder's prefix
    blobs = bucket.list_blobs(prefix=gcs_folder)

    Path(local_destination).mkdir(parents=True, exist_ok=True)

    print(
        f"downloading files from {YELLOW}{gcs_folder}{COLOR_END} in bucket {BLUE}{bucket_name}{COLOR_END}..."
    )
    logger.info(f"downloading files from '{gcs_folder}' in bucket '{bucket_name}'")

    # create the local destination directory if it doesn't exist.
    # if not os.path.exists(destination_directory):
    #     os.makedirs(destination_directory)

    downloaded_count = 0

    for blob in blobs:
        # check if it's not just a "folder" placeholder.
        if blob.name.endswith("/"):
            continue

        relative_path = os.path.relpath(blob.name, gcs_folder)
        local_file_path = os.path.join(local_destination, relative_path)

        local_file_dir = os.path.dirname(local_file_path)
        Path(local_file_dir).mkdir(parents=True, exist_ok=True)

        try:
            print(
                f"Downloading {YELLOW}{blob.name}{COLOR_END} to {BLUE}{local_file_path}{COLOR_END}"
            )
            logger.info(f"Downloading '{blob.name}' to '{local_file_path}'")
            blob.download_to_filename(local_file_path)
            downloaded_count += 1
        except Exception as e:
            print(
                f"{RED}failed to download:{COLOR_END} {YELLOW}{blob.name}{COLOR_END} to {BLUE}{local_file_path}{COLOR_END}"
            )
            logger.critical(f"failed to download '{blob.name}' to '{local_file_path}'")

    print(
        f"{GREEN}All files from{COLOR_END} {YELLOW}{gcs_folder}{COLOR_END} {GREEN}downloaded to{COLOR_END}  {BLUE}{local_destination}{COLOR_END}"
    )
    print(
        f"{GREEN}Total downloaded files{COLOR_END}: {YELLOW}{downloaded_count}{COLOR_END}"
    )
    logger.success(f"Total downloaded files: {downloaded_count}")


if __name__ == "__main__":
    download_folder_from_requester_pays_bucket(
        billing_project_id=BILLING_PROJECT_ID,
        bucket_name=BUCKET_NAME,
        gcs_folder=GCS_FOLDER,
        local_destination=LOCAL_DESTINATION,
        logger=logger
    )
