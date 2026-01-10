"""Tests for pokedata.cvat module."""

import io
import zipfile
from pathlib import Path
from http import HTTPStatus
from unittest.mock import Mock, patch

import pytest
import requests

from pokedata.cvat import CVATClient, CVATError


class TestCVATClient:
    """Tests for CVATClient class."""

    def test_init(self):
        """Test CVATClient initialization."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )
        assert client.api_url == "https://example.com/api/v1"
        assert client.auth == "Bearer test_token"
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == "Bearer test_token"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slashes are removed from API URL."""
        client = CVATClient(
            api_url="https://example.com/api/v1/", auth="Bearer test_token"
        )
        assert client.api_url == "https://example.com/api/v1"

    def test_init_without_bearer_prefix(self):
        """Test that auth works with or without Bearer prefix."""
        client = CVATClient(api_url="https://example.com/api/v1", auth="test_token")
        assert client.auth == "test_token"
        assert client.session.headers["Authorization"] == "test_token"


class TestDownloadTask:
    """Tests for download_task method."""

    def _create_mock_zip(self) -> bytes:
        """Create a mock ZIP file in memory."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("annotations/instances_default.json", '{"images": []}')
            zip_file.writestr("images/test_image.png", b"fake image data")
        zip_buffer.seek(0)
        return zip_buffer.read()

    @patch("pokedata.cvat.logger")
    def test_download_task_success(self, mock_logger, tmp_path):
        """Test successful task download."""
        # Setup
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )
        task_id = 123
        output_dir = tmp_path / "output"
        mock_zip_data = self._create_mock_zip()

        # Mock the session.get response
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.headers = {"Content-Type": "application/zip"}
        mock_response.iter_content = Mock(return_value=[mock_zip_data])
        mock_response.raise_for_status = Mock()

        client.session.get = Mock(return_value=mock_response)

        # Execute
        result_path = client.download_task(
            task_id=task_id, output_dir=output_dir, format="COCO 1.0"
        )

        # Verify
        assert result_path == output_dir / f"task_{task_id}"
        assert result_path.exists()
        assert (result_path / "annotations" / "instances_default.json").exists()
        assert (result_path / "images" / "test_image.png").exists()
        # ZIP file should be removed after extraction
        assert not (result_path / "dataset.zip").exists()

        # Verify API call
        client.session.get.assert_called_once()
        call_args = client.session.get.call_args
        assert call_args[0][0] == f"https://example.com/api/v1/tasks/{task_id}/dataset"
        assert call_args[1]["params"]["action"] == "download"
        assert call_args[1]["params"]["format"] == "COCO 1.0"
        assert call_args[1]["stream"] is True

    @patch("pokedata.cvat.logger")
    def test_download_task_custom_format(self, mock_logger, tmp_path):
        """Test download with custom format."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )
        mock_zip_data = self._create_mock_zip()

        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.iter_content = Mock(return_value=[mock_zip_data])
        mock_response.raise_for_status = Mock()
        client.session.get = Mock(return_value=mock_response)

        result_path = client.download_task(
            task_id=456, output_dir=tmp_path, format="YOLO 1.0"
        )

        call_args = client.session.get.call_args
        assert call_args[1]["params"]["format"] == "YOLO 1.0"

    @patch("pokedata.cvat.logger")
    def test_download_task_not_found(self, mock_logger, tmp_path):
        """Test download when task is not found."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )

        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.NOT_FOUND
        mock_response.raise_for_status = Mock(
            side_effect=requests.exceptions.HTTPError(response=mock_response)
        )
        client.session.get = Mock(return_value=mock_response)

        # Execute and verify
        with pytest.raises(CVATError, match="Task 999 not found"):
            client.download_task(task_id=999, output_dir=tmp_path)

    @patch("pokedata.cvat.logger")
    def test_download_task_unauthorized(self, mock_logger, tmp_path):
        """Test download when authentication fails."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer invalid_token"
        )

        # Mock 401 response
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.UNAUTHORIZED
        mock_response.raise_for_status = Mock(
            side_effect=requests.exceptions.HTTPError(response=mock_response)
        )
        client.session.get = Mock(return_value=mock_response)

        # Execute and verify
        with pytest.raises(CVATError, match="Authentication failed"):
            client.download_task(task_id=123, output_dir=tmp_path)

    @patch("pokedata.cvat.logger")
    def test_download_task_other_http_error(self, mock_logger, tmp_path):
        """Test download with other HTTP errors."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )

        # Mock 500 response
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.INTERNAL_SERVER_ERROR
        mock_response.raise_for_status = Mock(
            side_effect=requests.exceptions.HTTPError(response=mock_response)
        )
        client.session.get = Mock(return_value=mock_response)

        # Execute and verify
        with pytest.raises(CVATError, match="Failed to download dataset"):
            client.download_task(task_id=123, output_dir=tmp_path)

    @patch("pokedata.cvat.logger")
    def test_download_task_network_error(self, mock_logger, tmp_path):
        """Test download when network error occurs."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )

        # Mock network error
        client.session.get = Mock(
            side_effect=requests.exceptions.ConnectionError("Connection failed")
        )

        # Execute and verify
        with pytest.raises(CVATError, match="Network error"):
            client.download_task(task_id=123, output_dir=tmp_path)

    @patch("pokedata.cvat.logger")
    def test_download_task_invalid_zip(self, mock_logger, tmp_path):
        """Test download when invalid ZIP file is received."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )

        # Mock response with invalid ZIP data
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.iter_content = Mock(return_value=[b"not a zip file"])
        mock_response.raise_for_status = Mock()
        client.session.get = Mock(return_value=mock_response)

        # Execute and verify
        with pytest.raises(CVATError, match="Invalid ZIP file"):
            client.download_task(task_id=123, output_dir=tmp_path)

    @patch("pokedata.cvat.logger")
    def test_download_task_io_error(self, mock_logger, tmp_path):
        """Test download when file I/O error occurs."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )
        mock_zip_data = self._create_mock_zip()

        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.iter_content = Mock(return_value=[mock_zip_data])
        mock_response.raise_for_status = Mock()
        client.session.get = Mock(return_value=mock_response)

        # Mock file write to raise IOError
        with patch("builtins.open", side_effect=IOError("Disk full")):
            with pytest.raises(CVATError, match="Failed to save dataset ZIP file"):
                client.download_task(task_id=123, output_dir=tmp_path)

    @patch("pokedata.cvat.logger")
    def test_download_task_creates_output_directory(self, mock_logger, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )
        mock_zip_data = self._create_mock_zip()

        # Use a non-existent subdirectory
        output_dir = tmp_path / "new" / "nested" / "dir"

        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.iter_content = Mock(return_value=[mock_zip_data])
        mock_response.raise_for_status = Mock()
        client.session.get = Mock(return_value=mock_response)

        result_path = client.download_task(task_id=123, output_dir=output_dir)

        assert output_dir.exists()
        assert result_path.exists()

    @patch("pokedata.cvat.logger")
    def test_download_task_chunked_download(self, mock_logger, tmp_path):
        """Test that download handles chunked content correctly."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )
        mock_zip_data = self._create_mock_zip()

        # Split zip data into chunks
        chunk_size = 100
        chunks = [
            mock_zip_data[i : i + chunk_size]
            for i in range(0, len(mock_zip_data), chunk_size)
        ]

        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.iter_content = Mock(return_value=chunks)
        mock_response.raise_for_status = Mock()
        client.session.get = Mock(return_value=mock_response)

        result_path = client.download_task(task_id=123, output_dir=tmp_path)

        # Verify file was reconstructed correctly
        assert (result_path / "annotations" / "instances_default.json").exists()

    @patch("pokedata.cvat.logger")
    def test_download_task_timeout(self, mock_logger, tmp_path):
        """Test that timeout parameter is passed to request."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )
        mock_zip_data = self._create_mock_zip()

        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.iter_content = Mock(return_value=[mock_zip_data])
        mock_response.raise_for_status = Mock()
        client.session.get = Mock(return_value=mock_response)

        client.download_task(task_id=123, output_dir=tmp_path, timeout=600)

        call_args = client.session.get.call_args
        assert call_args[1]["timeout"] == 600

    @patch("pokedata.cvat.logger")
    def test_download_task_filename_parameter(self, mock_logger, tmp_path):
        """Test that filename parameter is included in request."""
        client = CVATClient(
            api_url="https://example.com/api/v1", auth="Bearer test_token"
        )
        mock_zip_data = self._create_mock_zip()

        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.iter_content = Mock(return_value=[mock_zip_data])
        mock_response.raise_for_status = Mock()
        client.session.get = Mock(return_value=mock_response)

        task_id = 789
        client.download_task(task_id=task_id, output_dir=tmp_path)

        call_args = client.session.get.call_args
        assert call_args[1]["params"]["filename"] == f"task_{task_id}_dataset.zip"
