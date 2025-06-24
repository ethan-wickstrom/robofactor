import os
import re
import shutil
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def replace_list_with_sequence(file_path: Path) -> None:
    """Replace list/List with Sequence and add import if needed"""
    try:
        # Read the entire file
        content = file_path.read_text(encoding="utf-8")

        # Check if Sequence import exists
        has_sequence_import = any(
            re.search(r"from\s+collections\.abc\s+import\s+Sequence", line)
            for line in content.splitlines()
        )

        # Replace list/List with Sequence
        new_content = re.sub(r"\blist\b", "Sequence", content)
        new_content = re.sub(r"\bList\b", "Sequence", new_content)

        # Add import if needed
        if not has_sequence_import:
            new_content = f"from collections.abc import Sequence\n{new_content}"

        # Create backup
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

        # Write changes
        file_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Updated: {file_path}")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")


def process_directory(directory: Path) -> None:
    """Process all Python files in directory"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                replace_list_with_sequence(file_path)


if __name__ == "__main__":
    target_dir = Path("src/robofactor")
    if not target_dir.exists():
        logger.error(f"Target directory not found: {target_dir}")
        sys.exit(1)

    process_directory(target_dir)
    logger.info(f"Successfully updated files in {target_dir}")
