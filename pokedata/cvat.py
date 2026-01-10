"""CVAT API client for downloading tasks and annotations."""

import zipfile
from pathlib import Path
from http import HTTPStatus

import requests
from loguru import logger


class CVATError(Exception):
    """Raised when CVAT API operations fail."""

    pass


class CVATClient:
    """Client for interacting with the CVAT API."""

    def __init__(self, api_url: str, auth: str):
        """
        Initialize CVAT client.

        Args:
            api_url: Base URL for CVAT API (e.g., "https://staging-grading.agscard.com/api/v1")
            auth: Bearer token string (should include "Bearer" prefix if needed)
        """
        self.api_url = api_url.rstrip("/")
        self.auth = auth

        self.session = requests.Session()
        self.session.headers.update({"Authorization": auth})

    def download_task(
        self,
        task_id: int,
        output_dir: Path,
        format: str = "COCO 1.0",
        save_images: bool = True,
        timeout: int = 300,
    ) -> Path:
        """
        Download a task's dataset (images and annotations) from CVAT.

        This method downloads the dataset directly using CVAT v1 API.

        Args:
            task_id: The task ID to download
            output_dir: Directory where the extracted dataset will be saved
            format: Annotation format (default: "COCO 1.0")
            save_images: Whether to include images in the export (default: True)
            timeout: Maximum time to wait for download in seconds (default: 300)

        Returns:
            Path to the extracted dataset directory

        Raises:
            CVATError: If the download or extraction fails
        """
        logger.info(f"Starting download for task {task_id} in format {format}")

        # CVAT v1 API: GET /api/v1/tasks/{id}/dataset with action=download
        dataset_url = f"{self.api_url}/tasks/{task_id}/dataset"
        params = {
            "action": "download",
            "format": format,
            "filename": f"task_{task_id}_dataset.zip",
        }

        try:
            response = self.session.get(
                dataset_url, params=params, stream=True, timeout=timeout
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status_code = (
                e.response.status_code
                if hasattr(e, "response") and e.response
                else None
            )
            if status_code == HTTPStatus.NOT_FOUND:
                raise CVATError(f"Task {task_id} not found") from e
            elif status_code == HTTPStatus.UNAUTHORIZED:
                raise CVATError("Authentication failed. Check your auth token.") from e
            else:
                raise CVATError(
                    f"Failed to download dataset for task {task_id}: {e}"
                ) from e
        except requests.exceptions.RequestException as e:
            raise CVATError(f"Network error while downloading dataset: {e}") from e

        # Save ZIP to temporary location
        output_dir.mkdir(parents=True, exist_ok=True)
        task_output_dir = output_dir / f"task_{task_id}"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = task_output_dir / "dataset.zip"

        logger.info(f"Downloading dataset for task {task_id}")

        try:
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except IOError as e:
            raise CVATError(f"Failed to save dataset ZIP file: {e}") from e

        logger.info(f"Dataset ZIP saved to {zip_path}")

        # Extract the ZIP file
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(task_output_dir)
            logger.info(f"Dataset extracted to {task_output_dir}")
        except zipfile.BadZipFile as e:
            raise CVATError(f"Invalid ZIP file downloaded: {e}") from e
        except zipfile.LargeZipFile as e:
            raise CVATError(f"ZIP file too large to extract: {e}") from e
        except Exception as e:
            raise CVATError(f"Failed to extract ZIP file: {e}") from e

        # Optionally remove the ZIP file after extraction
        try:
            zip_path.unlink()
            logger.debug(f"Removed ZIP file {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to remove ZIP file {zip_path}: {e}")

        logger.info(f"Task {task_id} downloaded successfully to {task_output_dir}")
        return task_output_dir
