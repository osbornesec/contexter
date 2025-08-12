"""
Environment variable management for secure credential handling.
"""

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment variable integration."""

    def __init__(self) -> None:
        self.required_env_vars = ["BRIGHTDATA_CUSTOMER_ID", "BRIGHTDATA_PASSWORD"]

    def validate_required_environment_vars(self) -> Dict[str, str]:
        """Validate that all required environment variables are set."""

        missing_vars = []
        env_values = {}

        for var_name in self.required_env_vars:
            value = os.getenv(var_name)
            if not value:
                missing_vars.append(var_name)
            else:
                env_values[var_name] = value

        if missing_vars:
            raise OSError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                f"Please set these variables before running the application.\n\n"
                f"Example:\n"
                f"  export BRIGHTDATA_CUSTOMER_ID='your_customer_id'\n"
                f"  export BRIGHTDATA_PASSWORD='your_password'"
            )

        logger.debug(f"Validated {len(env_values)} required environment variables")
        return env_values

    def get_brightdata_credentials(self) -> Dict[str, str]:
        """Get BrightData credentials from environment variables."""

        customer_id = os.getenv("BRIGHTDATA_CUSTOMER_ID")
        password = os.getenv("BRIGHTDATA_PASSWORD")

        if not customer_id or not password:
            raise OSError(
                "BrightData credentials not found. Please set BRIGHTDATA_CUSTOMER_ID "
                "and BRIGHTDATA_PASSWORD environment variables."
            )

        # Basic validation of credentials format
        if len(customer_id.strip()) < 3:
            raise ValueError("BRIGHTDATA_CUSTOMER_ID appears to be too short")

        if len(password.strip()) < 8:
            raise ValueError("BRIGHTDATA_PASSWORD appears to be too short")

        logger.debug("Successfully retrieved BrightData credentials from environment")

        return {"customer_id": customer_id.strip(), "password": password.strip()}

    def get_optional_config_overrides(self) -> Dict[str, str]:
        """Get optional configuration overrides from environment variables."""

        optional_vars = {
            "CONTEXTER_CONFIG_PATH": os.getenv("CONTEXTER_CONFIG_PATH"),
            "CONTEXTER_STORAGE_PATH": os.getenv("CONTEXTER_STORAGE_PATH"),
            "CONTEXTER_LOG_LEVEL": os.getenv("CONTEXTER_LOG_LEVEL"),
            "CONTEXTER_DEBUG_MODE": os.getenv("CONTEXTER_DEBUG_MODE"),
        }

        # Filter out None values
        return {k: v for k, v in optional_vars.items() if v is not None}

    def validate_credential_security(self) -> None:
        """Validate that credentials are not exposed in common locations."""

        # Check if credentials might be exposed in shell history
        shell_history_files = ["~/.bash_history", "~/.zsh_history", "~/.history"]

        warnings = []

        for history_file in shell_history_files:
            history_path = os.path.expanduser(history_file)
            if os.path.exists(history_path):
                try:
                    with open(history_path, errors="ignore") as f:
                        content = f.read()
                        if (
                            "BRIGHTDATA_CUSTOMER_ID" in content
                            or "BRIGHTDATA_PASSWORD" in content
                        ):
                            warnings.append(
                                f"Credentials found in shell history: {history_file}"
                            )
                except Exception:
                    pass  # Ignore read errors

        if warnings:
            logger.warning(
                "Security warning: BrightData credentials may be exposed in shell history. "
                f"Consider clearing history files: {', '.join(warnings)}"
            )
