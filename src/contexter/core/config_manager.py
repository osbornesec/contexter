"""
Main configuration management with file watching and validation.
"""

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..models.config_models import C7DocConfig
from .environment_manager import EnvironmentManager
from .yaml_parser import YAMLConfigParser

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""

    pass


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration file changes."""

    def __init__(self, config_manager: "ConfigurationManager"):
        self.config_manager = config_manager
        self.debounce_timeout = 1.0  # 1 second debounce
        self.last_modified = 0

    def on_modified(self, event: Any) -> None:
        """Handle file modification events."""
        self._handle_event(event, "modified")

    def on_moved(self, event: Any) -> None:
        """Handle file move events (for atomic writes)."""
        self._handle_event(event, "moved")

    def on_created(self, event: Any) -> None:
        """Handle file creation events."""
        self._handle_event(event, "created")

    def _handle_event(self, event: Any, event_type: str) -> None:
        """Handle file system events."""
        if event.is_directory:
            return

        # Check if it's our configuration file or a related temporary file
        event_path = Path(event.src_path)
        config_path = self.config_manager.config_path

        # For move events, check the destination path
        if hasattr(event, "dest_path"):
            dest_path = Path(event.dest_path)
            if dest_path == config_path:
                self._process_config_change(f"File moved to {event.dest_path}")
                return

        # Check for direct match
        if event_path == config_path:
            self._process_config_change(f"File {event_type}: {event.src_path}")
            return

        # Check for temporary files that might be related (atomic writes)
        if (
            config_path
            and event_path.parent == config_path.parent
            and event_path.stem == config_path.stem
            and event_path.suffix == ".tmp"
        ):
            # Wait a bit for the atomic operation to complete, then check if config file exists
            def check_config_after_temp() -> None:
                time.sleep(0.1)  # Short wait for atomic operation
                if config_path.exists():
                    self._process_config_change(
                        f"Config updated via temp file: {event.src_path}"
                    )

            reload_thread = threading.Thread(target=check_config_after_temp)
            reload_thread.daemon = True
            reload_thread.start()

    def _process_config_change(self, description: str) -> None:
        """Process a configuration file change."""
        # Debounce rapid file changes
        current_time = time.time()
        if current_time - self.last_modified < self.debounce_timeout:
            return

        self.last_modified = int(current_time)
        logger.info(f"Config file changed: {description}")

        # Schedule configuration reload using threading approach
        def reload_config() -> None:
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self.config_manager._handle_config_file_change()
                )
                loop.close()
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")

        # Run in a separate thread to avoid event loop conflicts
        reload_thread = threading.Thread(target=reload_config)
        reload_thread.daemon = True
        reload_thread.start()


class ConfigurationManager:
    """Main configuration management with file watching and validation."""

    def __init__(self) -> None:
        self.yaml_parser = YAMLConfigParser()
        self.env_manager = EnvironmentManager()
        self.current_config: Optional[C7DocConfig] = None
        self.config_path: Optional[Path] = None
        self.file_watcher: Optional[Any] = None
        self.change_callback: Optional[Callable[[C7DocConfig], None]] = None
        self._lock = threading.Lock()

    async def load_config(self, config_path: Optional[Path] = None) -> C7DocConfig:
        """Load configuration from file with validation."""

        # Determine configuration file path
        if not config_path:
            config_path = self._get_default_config_path()

        self.config_path = config_path

        # Create default config if it doesn't exist
        if not config_path.exists():
            logger.info(
                f"Configuration file not found at {config_path}, creating default configuration"
            )
            await self.generate_default_config(config_path)

        # Validate required environment variables
        env_credentials = self.env_manager.validate_required_environment_vars()

        try:
            # Load YAML configuration
            config_data = await self.yaml_parser.load_yaml_config(config_path)

            # Apply environment variable overrides
            self._apply_environment_overrides(config_data)

            # Inject environment credentials into config data
            if "proxy" not in config_data:
                config_data["proxy"] = {}

            # Don't override if already set in config (for testing)
            if not config_data["proxy"].get("customer_id"):
                config_data["proxy"]["customer_id"] = env_credentials[
                    "BRIGHTDATA_CUSTOMER_ID"
                ]

            # Validate configuration using Pydantic
            validated_config = C7DocConfig(**config_data)

            # Perform extended validation
            await self._perform_extended_validation(validated_config)

            with self._lock:
                self.current_config = validated_config

            logger.info(f"Configuration loaded successfully from {config_path}")
            return validated_config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}") from e

    async def save_config(
        self, config: C7DocConfig, config_path: Optional[Path] = None
    ) -> None:
        """Save configuration to file."""

        if not config_path:
            config_path = self.config_path or self._get_default_config_path()

        try:
            # Convert to dictionary, excluding credentials
            config_dict = config.model_dump()

            # Remove sensitive data that should come from environment
            if "proxy" in config_dict and "customer_id" in config_dict["proxy"]:
                config_dict["proxy"]["customer_id"] = "${BRIGHTDATA_CUSTOMER_ID}"

            # Save to YAML file
            await self.yaml_parser.save_yaml_config(config_dict, config_path)

            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Configuration saving failed: {e}") from e

    async def generate_default_config(self, config_path: Optional[Path] = None) -> None:
        """Generate default configuration file with documentation."""

        if not config_path:
            config_path = self._get_default_config_path()

        # Check if configuration already exists
        if config_path.exists():
            logger.warning(f"Configuration file already exists at {config_path}")
            return

        # Create default configuration
        default_config = C7DocConfig(debug_mode=False, config_version="1.0")

        # Apply any environment overrides to defaults
        config_dict = default_config.model_dump()
        self._apply_environment_overrides(config_dict)
        updated_config = C7DocConfig(**config_dict)

        # Save to file
        await self.save_config(updated_config, config_path)

        logger.info(f"Default configuration generated at {config_path}")
        logger.info("Next steps:")
        logger.info("1. Set environment variables:")
        logger.info("   export BRIGHTDATA_CUSTOMER_ID='your_customer_id'")
        logger.info("   export BRIGHTDATA_PASSWORD='your_password'")
        logger.info("2. Edit the configuration file to customize settings")
        logger.info("3. Run validation to verify your configuration")

    async def validate_config(
        self, config_data: Optional[Dict[str, Any]] = None
    ) -> C7DocConfig:
        """Validate configuration data or current config."""

        if config_data is None:
            if not self.current_config:
                raise ConfigurationError("No configuration loaded to validate")

            # Re-validate current config
            await self._perform_extended_validation(self.current_config)
            return self.current_config

        try:
            # Apply environment overrides
            self._apply_environment_overrides(config_data)

            # Validate using Pydantic
            validated_config = C7DocConfig(**config_data)

            # Additional validation logic
            await self._perform_extended_validation(validated_config)

            return validated_config

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e

    def start_file_watcher(self, callback: Callable[[C7DocConfig], None]) -> None:
        """Start watching configuration file for changes."""

        if not self.config_path:
            raise ConfigurationError("No configuration file path set")

        self.change_callback = callback

        # Set up file watcher
        event_handler = ConfigFileHandler(self)
        self.file_watcher = Observer()
        self.file_watcher.schedule(
            event_handler, str(self.config_path.parent), recursive=False
        )
        self.file_watcher.start()

        logger.info(f"Started watching configuration file: {self.config_path}")

    def stop_file_watcher(self) -> None:
        """Stop watching configuration file."""

        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher.join()
            self.file_watcher = None

        self.change_callback = None
        logger.info("Stopped configuration file watcher")

    def get_current_config(self) -> Optional[C7DocConfig]:
        """Get the currently loaded configuration."""
        with self._lock:
            return self.current_config

    def get_config(self) -> Optional[C7DocConfig]:
        """Get the currently loaded configuration (alias for compatibility)."""
        return self.get_current_config()

    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""

        # Check environment variable override
        env_path = os.getenv("CONTEXTER_CONFIG_PATH")
        if env_path:
            return Path(env_path).expanduser()

        # Use default location
        config_dir = Path.home() / ".contexter"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.yaml"

    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> None:
        """Apply environment variable overrides to configuration data."""

        overrides = self.env_manager.get_optional_config_overrides()

        # Apply storage path override
        if "CONTEXTER_STORAGE_PATH" in overrides:
            if "storage" not in config_data:
                config_data["storage"] = {}
            config_data["storage"]["base_path"] = overrides["CONTEXTER_STORAGE_PATH"]

        # Apply logging level override
        if "CONTEXTER_LOG_LEVEL" in overrides:
            if "logging" not in config_data:
                config_data["logging"] = {}
            config_data["logging"]["level"] = overrides["CONTEXTER_LOG_LEVEL"].upper()

        # Apply debug mode override
        if "CONTEXTER_DEBUG_MODE" in overrides:
            debug_value = overrides["CONTEXTER_DEBUG_MODE"].lower() in (
                "true",
                "1",
                "yes",
                "on",
            )
            config_data["debug_mode"] = debug_value
            if debug_value:
                if "logging" not in config_data:
                    config_data["logging"] = {}
                config_data["logging"]["level"] = "DEBUG"

    async def _perform_extended_validation(self, config: C7DocConfig) -> None:
        """Perform extended validation beyond Pydantic model validation."""

        # Validate storage directory accessibility
        storage_path = Path(config.storage.base_path).expanduser()
        try:
            storage_path.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = storage_path / ".test_write"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ValueError(f"Storage directory is not writable: {e}") from e

        # Validate BrightData credentials if available
        try:
            credentials = self.env_manager.get_brightdata_credentials()
            if len(credentials["customer_id"]) < 5:
                raise ValueError("BrightData customer ID seems too short")
        except OSError:
            logger.warning("BrightData credentials not available for validation")

        # Check for potential security issues
        try:
            self.env_manager.validate_credential_security()
        except Exception as e:
            logger.warning(f"Security validation warning: {e}")

        # Validate cross-component compatibility
        if config.download.max_concurrent > config.proxy.pool_size * 3:
            logger.warning(
                f"High concurrent download count ({config.download.max_concurrent}) "
                f"relative to proxy pool size ({config.proxy.pool_size}). "
                f"This may cause connection issues."
            )

    async def _handle_config_file_change(self) -> None:
        """Handle configuration file change event."""

        try:
            logger.info("Configuration file changed, reloading...")

            # Reload configuration
            new_config = await self.load_config(self.config_path)

            # Notify callback if set
            if self.change_callback:
                try:
                    self.change_callback(new_config)
                except Exception as e:
                    logger.error(f"Configuration change callback failed: {e}")

            logger.info("Configuration reloaded successfully")

        except Exception as e:
            logger.error(f"Failed to reload configuration after file change: {e}")

    def __enter__(self) -> "ConfigurationManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.stop_file_watcher()
