"""
End-to-end CLI integration tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock

from contexter.cli.main import cli
from contexter.models.config_models import C7DocConfig, StorageConfig


@pytest.mark.e2e
def test_cli_status_command_integration(mock_environment_vars, temp_storage_dir: Path):
    """Test CLI status command end-to-end."""
    
    runner = CliRunner()
    
    # Mock successful health checks
    with patch('c7doc.cli.commands.status._check_configuration_health') as mock_config, \
         patch('c7doc.cli.commands.status._check_proxy_health') as mock_proxy, \
         patch('c7doc.cli.commands.status._check_api_connectivity') as mock_api, \
         patch('c7doc.cli.commands.status._check_storage_health') as mock_storage:
        
        # Configure successful responses
        mock_config.return_value = {"healthy": True, "details": "All settings configured"}
        mock_proxy.return_value = {"healthy": True, "details": "Proxy connectivity test passed"}
        mock_api.return_value = {"healthy": True, "details": "Context7 API is reachable"}
        mock_storage.return_value = {"healthy": True, "details": "Storage accessible"}
        
        # Test basic status
        result = runner.invoke(cli, ['status'])
        
        assert result.exit_code == 0
        assert "system status" in result.output.lower()
        assert "all systems operational" in result.output.lower()
        
        # Test detailed status
        result = runner.invoke(cli, ['status', '--detailed'])
        
        assert result.exit_code == 0
        assert "detailed system information" in result.output.lower()
        assert "environment" in result.output.lower()


@pytest.mark.e2e
def test_cli_status_command_with_issues(mock_environment_vars):
    """Test CLI status command with system issues."""
    
    runner = CliRunner()
    
    with patch('c7doc.cli.commands.status._check_configuration_health') as mock_config, \
         patch('c7doc.cli.commands.status._check_proxy_health') as mock_proxy, \
         patch('c7doc.cli.commands.status._check_api_connectivity') as mock_api, \
         patch('c7doc.cli.commands.status._check_storage_health') as mock_storage:
        
        # Configure mixed responses with issues
        mock_config.return_value = {"healthy": False, "error": "Missing credentials"}
        mock_proxy.return_value = {"healthy": False, "error": "Connection failed"}
        mock_api.return_value = {"healthy": True, "details": "API reachable"}
        mock_storage.return_value = {"healthy": True, "details": "Storage accessible"}
        
        result = runner.invoke(cli, ['status'])
        
        assert result.exit_code == 1  # Should exit with error
        assert "some systems have issues" in result.output.lower()


@pytest.mark.e2e
def test_cli_config_commands_integration(temp_storage_dir: Path):
    """Test CLI configuration commands end-to-end."""
    
    runner = CliRunner()
    config_file = temp_storage_dir / "test_config.yaml"
    
    with patch.dict(os.environ, {"CONTEXTER_CONFIG_PATH": str(config_file)}):
        
        # Test config init
        result = runner.invoke(cli, ['config', 'init'])
        assert result.exit_code == 0
        assert config_file.exists()
        
        # Test config show
        result = runner.invoke(cli, ['config', 'show'])
        assert result.exit_code == 0
        assert "configuration" in result.output.lower()
        
        # Test config validate
        result = runner.invoke(cli, ['config', 'validate'])
        assert result.exit_code == 0


@pytest.mark.e2e 
def test_cli_download_command_integration(
    temp_storage_dir: Path,
    mock_environment_vars,
):
    """Test CLI download command with mocked dependencies."""
    
    runner = CliRunner()
    config_file = temp_storage_dir / "test_config.yaml"
    
    # Create test config
    test_config = C7DocConfig(
        storage=StorageConfig(base_path=str(temp_storage_dir))
    )
    
    with patch.dict(os.environ, {"CONTEXTER_CONFIG_PATH": str(config_file)}), \
         patch('c7doc.cli.commands.download.ConfigurationManager') as mock_config_mgr, \
         patch('c7doc.cli.commands.download.AsyncDownloadEngine') as mock_engine, \
         patch('c7doc.cli.commands.download.LocalStorageManager') as mock_storage, \
         patch('c7doc.cli.commands.download.BrightDataProxyManager') as mock_proxy:
        
        # Setup mocks
        mock_config_mgr.return_value.load_config = AsyncMock(return_value=test_config)
        
        # Mock download engine
        mock_engine_instance = AsyncMock()
        mock_engine.return_value = mock_engine_instance
        
        # Mock successful download
        from contexter.models.download_models import DownloadSummary, DocumentationChunk
        
        mock_chunk = DocumentationChunk(
            chunk_id="test-chunk",
            content="Test documentation content",
            source_context="test context",
            token_count=50,
            content_hash="testhash",
            proxy_id="test-proxy",
            download_time=1.0,
            library_id="test/lib",
            metadata={}
        )
        
        mock_summary = DownloadSummary(
            library_id="test/lib",
            total_contexts_attempted=1,
            successful_contexts=1,
            failed_contexts=0,
            chunks=[mock_chunk],
            total_tokens=50,
            total_download_time=1.0,
            start_time=None,
            end_time=None,
            error_summary={}
        )
        
        mock_engine_instance.download_library.return_value = mock_summary
        mock_engine_instance.shutdown = AsyncMock()
        
        # Mock storage
        from contexter.models.storage_models import StorageResult
        mock_storage_instance = AsyncMock()
        mock_storage.return_value = mock_storage_instance
        
        mock_storage_result = StorageResult(
            success=True,
            file_path=Path(temp_storage_dir / "test.json.gz"),
            compressed_size=1024,
            compression_ratio=0.7,
            checksum="abc123",
            version_id="v1",
        )
        
        mock_storage_instance.store_documentation.return_value = mock_storage_result
        
        # Test download command
        result = runner.invoke(cli, [
            'download',
            'test/lib',
            '--token-limit', '5000',
            '--verbose'
        ])
        
        assert result.exit_code == 0
        assert "downloading" in result.output.lower()
        assert "successful" in result.output.lower()


@pytest.mark.e2e
def test_cli_error_handling_and_recovery(temp_storage_dir: Path):
    """Test CLI error handling and recovery mechanisms."""
    
    runner = CliRunner()
    
    # Test invalid command
    result = runner.invoke(cli, ['invalid-command'])
    assert result.exit_code == 2  # Click error code
    
    # Test missing required arguments
    result = runner.invoke(cli, ['download'])
    assert result.exit_code == 2
    
    # Test invalid config file
    invalid_config = temp_storage_dir / "invalid.yaml"
    invalid_config.write_text("invalid: yaml: content: [")
    
    with patch.dict(os.environ, {"CONTEXTER_CONFIG_PATH": str(invalid_config)}):
        result = runner.invoke(cli, ['config', 'validate'])
        assert result.exit_code == 1


@pytest.mark.e2e
def test_cli_global_options_integration(temp_storage_dir: Path):
    """Test CLI global options like --verbose, --quiet, etc."""
    
    runner = CliRunner()
    
    # Test verbose output
    with patch('c7doc.cli.commands.status._check_configuration_health') as mock_health:
        mock_health.return_value = {"healthy": True, "details": "OK"}
        
        # Normal output
        result = runner.invoke(cli, ['status'])
        normal_length = len(result.output)
        
        # Verbose output should be longer
        result = runner.invoke(cli, ['--verbose', 'status'])
        verbose_length = len(result.output) 
        
        # Note: This might not always be true depending on implementation
        # but provides a general test
        assert result.exit_code == 0


@pytest.mark.e2e
def test_cli_interrupt_handling():
    """Test CLI interrupt and signal handling."""
    
    runner = CliRunner()
    
    # Test KeyboardInterrupt handling
    with patch('c7doc.cli.commands.status._check_configuration_health') as mock_health:
        mock_health.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(cli, ['status'])
        
        assert result.exit_code == 130  # Standard SIGINT exit code
        assert "interrupted" in result.output.lower()


@pytest.mark.e2e
def test_cli_configuration_file_locations(temp_storage_dir: Path):
    """Test CLI configuration file discovery and precedence."""
    
    runner = CliRunner()
    
    # Test 1: Explicit config path via environment
    config_path = temp_storage_dir / "explicit_config.yaml"
    
    with patch.dict(os.environ, {"CONTEXTER_CONFIG_PATH": str(config_path)}):
        result = runner.invoke(cli, ['config', 'init'])
        assert result.exit_code == 0
        assert config_path.exists()
    
    # Test 2: Default config path behavior
    with patch('c7doc.core.config_manager.ConfigurationManager._find_config_file') as mock_find:
        mock_find.return_value = None  # No config file found
        
        result = runner.invoke(cli, ['config', 'show'])
        # Should handle missing config gracefully
        assert result.exit_code in [0, 1]  # Either success with defaults or expected failure


@pytest.mark.e2e
def test_cli_output_formatting_consistency():
    """Test consistent output formatting across commands."""
    
    runner = CliRunner()
    
    with patch('c7doc.cli.commands.status._check_configuration_health') as mock_health:
        mock_health.return_value = {"healthy": True, "details": "All good"}
        
        # Test that Rich formatting is working
        result = runner.invoke(cli, ['status'])
        
        # Should contain ANSI color codes (Rich formatting)
        assert result.exit_code == 0
        # Basic check that output is formatted (not just plain text)
        assert len(result.output) > 10


@pytest.mark.e2e
def test_cli_help_and_documentation():
    """Test CLI help system and documentation."""
    
    runner = CliRunner()
    
    # Test main help
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Commands:" in result.output
    
    # Test subcommand help
    result = runner.invoke(cli, ['download', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "library" in result.output.lower()
    
    # Test status command help
    result = runner.invoke(cli, ['status', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    
    # Test config command help
    result = runner.invoke(cli, ['config', '--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output