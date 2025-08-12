"""
YAML configuration parser with environment variable substitution.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class YAMLConfigParser:
    """YAML configuration parser with environment variable substitution."""

    def __init__(self) -> None:
        self.env_var_pattern = re.compile(r"\$\{([^}]+)\}")

    async def load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load and parse YAML configuration file with environment substitution."""

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            # Read YAML file content
            with open(config_path, encoding="utf-8") as f:
                yaml_content = f.read()

            # Perform environment variable substitution
            substituted_content = self._substitute_environment_variables(yaml_content)

            # Parse YAML
            config_data = yaml.safe_load(substituted_content)

            if not isinstance(config_data, dict):
                raise ValueError("Configuration file must contain a YAML dictionary")

            logger.info(f"Successfully loaded configuration from {config_path}")
            return config_data

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file: {e}") from e

    async def save_yaml_config(
        self, config_data: Dict[str, Any], config_path: Path
    ) -> None:
        """Save configuration data to YAML file with proper formatting."""

        try:
            # Ensure parent directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate YAML with comments and formatting
            yaml_content = self._generate_commented_yaml(config_data)

            # Write to temporary file first (atomic operation)
            temp_path = config_path.with_suffix(".tmp")

            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            # Atomic move to final location
            temp_path.replace(config_path)

            logger.info(f"Successfully saved configuration to {config_path}")

        except Exception as e:
            # Clean up temp file if it exists
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to save configuration file: {e}") from e

    def _substitute_environment_variables(self, content: str) -> str:
        """Substitute environment variables in YAML content."""

        # Process line by line to skip comments
        lines = content.split("\n")
        processed_lines = []

        for line in lines:
            # Skip processing environment variables in comment lines
            if line.strip().startswith("#"):
                processed_lines.append(line)
                continue

            # Process environment variables in non-comment lines
            def replace_env_var(match: Any) -> str:
                var_name = match.group(1)

                # Support default values: ${VAR:default_value}
                if ":" in var_name:
                    var_name, default_value = var_name.split(":", 1)
                    env_value = os.getenv(var_name, default_value)
                else:
                    env_value = os.getenv(var_name)
                    if env_value is None:
                        raise ValueError(
                            f"Environment variable '{var_name}' is not set"
                        )

                return env_value

            try:
                processed_line = self.env_var_pattern.sub(replace_env_var, line)
                processed_lines.append(processed_line)
            except Exception as e:
                raise ValueError(
                    f"Environment variable substitution failed on line '{line}': {e}"
                ) from e

        return "\n".join(processed_lines)

    def _generate_commented_yaml(self, config_data: Dict[str, Any]) -> str:
        """Generate YAML with helpful comments and documentation."""

        # Get field descriptions from Pydantic model
        config_comments = {
            "proxy": {
                "_section_comment": "BrightData proxy configuration",
                "customer_id": "Your BrightData customer ID (set via BRIGHTDATA_CUSTOMER_ID env var)",
                "zone_name": "BrightData zone name for residential proxies",
                "pool_size": "Number of concurrent proxy connections (1-50)",
                "health_check_interval": "Health check interval in seconds (60-3600)",
                "circuit_breaker_threshold": "Failures before circuit breaker opens (1-20)",
                "circuit_breaker_timeout": "Circuit breaker recovery timeout in seconds (10-300)",
            },
            "download": {
                "_section_comment": "Download engine configuration",
                "max_concurrent": "Maximum concurrent downloads (1-50)",
                "max_contexts": "Maximum contexts per library (1-10)",
                "jitter_min": "Minimum delay between requests in seconds",
                "jitter_max": "Maximum delay between requests in seconds",
                "max_retries": "Maximum retry attempts (1-10)",
                "request_timeout": "Request timeout in seconds",
                "token_limit": "Maximum tokens per API request (1000-200000)",
            },
            "storage": {
                "_section_comment": "Storage configuration",
                "base_path": "Base directory for downloaded documentation",
                "compression_level": "Gzip compression level (1=fast, 9=best)",
                "retention_versions": "Number of versions to keep per library",
                "verify_integrity": "Enable integrity verification with checksums",
                "cleanup_threshold_gb": "Trigger cleanup when storage exceeds this size in GB",
            },
            "logging": {
                "_section_comment": "Logging configuration",
                "level": "Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
                "file_path": "Log file path (leave empty for console only)",
                "max_file_size_mb": "Maximum log file size in MB before rotation",
                "backup_count": "Number of rotated log files to keep",
            },
        }

        lines = []
        lines.append("# C7DocDownloader Configuration")
        lines.append("# Generated automatically - modify as needed")
        lines.append(
            "# Environment variables can be substituted using $\u007bVAR_NAME\u007d syntax"
        )
        lines.append("")

        for section_name, section_data in config_data.items():
            section_comments = config_comments.get(section_name, {})
            section_comment = section_comments.get(
                "_section_comment", f"{section_name} configuration"
            )

            lines.append(f"# {section_comment}")
            lines.append(f"{section_name}:")

            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    comment = section_comments.get(key, "")
                    if comment:
                        lines.append(f"  # {comment}")

                    # Handle special formatting for certain values
                    if key == "customer_id" and str(value).startswith("${"):
                        # Handle environment variable placeholders
                        lines.append(f"  {key}: {value}")
                    else:
                        # Format the value properly for YAML
                        if isinstance(value, (int, float, bool)):
                            lines.append(f"  {key}: {value}")
                        elif isinstance(value, str):
                            if (
                                " " in value
                                or ":" in value
                                or value.startswith(("${", "#"))
                            ):
                                lines.append(f'  {key}: "{value}"')
                            else:
                                lines.append(f"  {key}: {value}")
                        else:
                            yaml_value = yaml.dump(
                                value, default_flow_style=True
                            ).strip()
                            # Remove the document separator that yaml.dump adds
                            if yaml_value.endswith("..."):
                                yaml_value = yaml_value[:-3].strip()
                            lines.append(f"  {key}: {yaml_value}")
            else:
                yaml_value = yaml.dump(section_data, default_flow_style=True).strip()
                # Remove the document separator that yaml.dump adds
                if yaml_value.endswith("..."):
                    yaml_value = yaml_value[:-3].strip()
                lines.append(f"  {yaml_value}")

            lines.append("")

        return "\n".join(lines)
