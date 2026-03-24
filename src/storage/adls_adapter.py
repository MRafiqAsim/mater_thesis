"""
ADLS Gen2 Storage Adapter

Thin I/O layer that replaces local file operations with ADLS Gen2.
Used by Synapse notebooks to read/write pipeline data from Azure Data Lake.

Usage:
    adapter = ADLSAdapter(storage_account="mystorageaccount", container="pipeline-data")
    adapter.write_json("bronze/emails/email_001.json", data)
    data = adapter.read_json("bronze/emails/email_001.json")
    adapter.move("input/source/file.pst", "input/processed/file.pst")
"""

import io
import json
import logging
import os
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ADLSAdapter:
    """
    Azure Data Lake Storage Gen2 adapter for pipeline I/O.

    Wraps azure-storage-file-datalake SDK for read/write/move/list operations.
    Falls back to local filesystem if no storage account is configured (dev mode).
    """

    def __init__(
        self,
        storage_account: Optional[str] = None,
        container: str = "pipeline-data",
        credential: Optional[str] = None,
    ):
        """
        Initialize ADLS adapter.

        Args:
            storage_account: Azure storage account name. If None, uses local filesystem.
            container: ADLS container name.
            credential: Account key or SAS token. If None, uses DefaultAzureCredential.
        """
        self.storage_account = storage_account or os.getenv("ADLS_STORAGE_ACCOUNT")
        self.container = container or os.getenv("ADLS_CONTAINER", "pipeline-data")
        self.credential = credential or os.getenv("ADLS_STORAGE_KEY")
        self.is_local = not self.storage_account

        if self.is_local:
            logger.info("ADLSAdapter: No storage account configured, using local filesystem")
            self._local_root = os.getenv("LOCAL_DATA_ROOT", "./data")
        else:
            self._init_client()

    def _init_client(self):
        """Initialize ADLS client."""
        from azure.storage.filedatalake import DataLakeServiceClient

        account_url = f"https://{self.storage_account}.dfs.core.windows.net"

        if self.credential:
            self.service_client = DataLakeServiceClient(
                account_url=account_url,
                credential=self.credential,
            )
        else:
            from azure.identity import DefaultAzureCredential
            self.service_client = DataLakeServiceClient(
                account_url=account_url,
                credential=DefaultAzureCredential(),
            )

        self.fs_client = self.service_client.get_file_system_client(self.container)
        logger.info(f"ADLSAdapter: Connected to {self.storage_account}/{self.container}")

    # -------------------------------------------------------------------------
    # Core I/O
    # -------------------------------------------------------------------------

    def read_bytes(self, path: str) -> bytes:
        """Read raw bytes from a file."""
        if self.is_local:
            full_path = os.path.join(self._local_root, path)
            with open(full_path, "rb") as f:
                return f.read()

        file_client = self.fs_client.get_file_client(path)
        download = file_client.download_file()
        return download.readall()

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from a file."""
        return self.read_bytes(path).decode(encoding)

    def read_json(self, path: str) -> Any:
        """Read and parse a JSON file."""
        return json.loads(self.read_text(path))

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write raw bytes to a file."""
        if self.is_local:
            full_path = os.path.join(self._local_root, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(data)
            return

        file_client = self.fs_client.get_file_client(path)
        # Ensure parent directory exists
        parent = str(PurePosixPath(path).parent)
        if parent and parent != ".":
            self._ensure_directory(parent)
        file_client.upload_data(data, overwrite=True)

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        """Write text to a file."""
        self.write_bytes(path, text.encode(encoding))

    def write_json(self, path: str, data: Any, indent: int = 2) -> None:
        """Write data as JSON."""
        text = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
        self.write_text(path, text)

    # -------------------------------------------------------------------------
    # File operations
    # -------------------------------------------------------------------------

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        if self.is_local:
            return os.path.exists(os.path.join(self._local_root, path))

        try:
            file_client = self.fs_client.get_file_client(path)
            file_client.get_file_properties()
            return True
        except Exception:
            # Check if it's a directory
            try:
                dir_client = self.fs_client.get_directory_client(path)
                dir_client.get_directory_properties()
                return True
            except Exception:
                return False

    def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """
        List files in a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern (e.g., "*.json", "*.pst")

        Returns:
            List of file paths relative to container root
        """
        if self.is_local:
            from pathlib import Path
            local_dir = Path(self._local_root) / directory
            if not local_dir.exists():
                return []
            matches = list(local_dir.glob(pattern))
            return [str(m.relative_to(self._local_root)) for m in matches if m.is_file()]

        paths = []
        try:
            for item in self.fs_client.get_paths(path=directory):
                if not item.is_directory:
                    name = item.name
                    if pattern == "*" or PurePosixPath(name).match(pattern):
                        paths.append(name)
        except Exception as e:
            logger.warning(f"Failed to list {directory}: {e}")
        return paths

    def list_dirs(self, directory: str) -> List[str]:
        """List subdirectories in a directory."""
        if self.is_local:
            from pathlib import Path
            local_dir = Path(self._local_root) / directory
            if not local_dir.exists():
                return []
            return [str(d.relative_to(self._local_root)) for d in local_dir.iterdir() if d.is_dir()]

        dirs = []
        try:
            for item in self.fs_client.get_paths(path=directory):
                if item.is_directory:
                    dirs.append(item.name)
        except Exception:
            pass
        return dirs

    def move(self, source: str, destination: str) -> None:
        """
        Move a file from source to destination.

        Args:
            source: Source file path
            destination: Destination file path
        """
        if self.is_local:
            src = os.path.join(self._local_root, source)
            dst = os.path.join(self._local_root, destination)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            logger.info(f"Moved {source} → {destination}")
            return

        # ADLS rename = move
        dest_parent = str(PurePosixPath(destination).parent)
        if dest_parent and dest_parent != ".":
            self._ensure_directory(dest_parent)

        file_client = self.fs_client.get_file_client(source)
        new_path = f"{self.container}/{destination}"
        file_client.rename_file(new_path)
        logger.info(f"Moved {source} → {destination}")

    def delete(self, path: str) -> None:
        """Delete a file."""
        if self.is_local:
            full_path = os.path.join(self._local_root, path)
            if os.path.exists(full_path):
                os.remove(full_path)
            return

        file_client = self.fs_client.get_file_client(path)
        file_client.delete_file()

    def copy(self, source: str, destination: str) -> None:
        """Copy a file."""
        data = self.read_bytes(source)
        self.write_bytes(destination, data)

    # -------------------------------------------------------------------------
    # Bulk operations
    # -------------------------------------------------------------------------

    def read_all_json(self, directory: str) -> Dict[str, Any]:
        """Read all JSON files in a directory. Returns {filename: data}."""
        results = {}
        for path in self.list_files(directory, "*.json"):
            try:
                filename = PurePosixPath(path).name
                results[filename] = self.read_json(path)
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
        return results

    def write_all_json(self, directory: str, data: Dict[str, Any]) -> int:
        """Write multiple JSON files. data = {filename: content}. Returns count."""
        count = 0
        for filename, content in data.items():
            path = f"{directory}/{filename}"
            self.write_json(path, content)
            count += 1
        return count

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _ensure_directory(self, path: str) -> None:
        """Ensure a directory exists in ADLS."""
        if self.is_local:
            os.makedirs(os.path.join(self._local_root, path), exist_ok=True)
            return

        dir_client = self.fs_client.get_directory_client(path)
        try:
            dir_client.create_directory()
        except Exception:
            pass  # Already exists

    def abfss_path(self, path: str) -> str:
        """Get the full abfss:// path for Spark/Synapse compatibility."""
        if self.is_local:
            return os.path.join(self._local_root, path)
        return f"abfss://{self.container}@{self.storage_account}.dfs.core.windows.net/{path}"

    def get_local_temp(self, path: str, temp_dir: str = "/tmp/pipeline") -> str:
        """
        Download a file to local temp for libraries that need local paths (e.g., pypff).

        Returns local file path.
        """
        if self.is_local:
            return os.path.join(self._local_root, path)

        local_path = os.path.join(temp_dir, PurePosixPath(path).name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data = self.read_bytes(path)
        with open(local_path, "wb") as f:
            f.write(data)
        logger.info(f"Downloaded {path} → {local_path} ({len(data)} bytes)")
        return local_path
