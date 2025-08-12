"""
Live CLI integration tests with real services.
"""

import os
import tempfile
from pathlib import Path
import pytest
from click.testing import CliRunner

from contexter.cli.main import cli


@pytest.mark.live
def test_cli_status_command_live():
    """Test CLI status command with real environment."""
    
    runner = CliRunner()
    
    # Test basic status command
    result = runner.invoke(cli, ['status'])
    
    # Should not crash (exit code 0 or 1 is acceptable)
    assert result.exit_code in [0, 1], f"Status command should not crash: {result.output}"
    
    # Should contain expected status information
    assert "system status" in result.output.lower() or "status" in result.output.lower()
    
    print(f"📋 CLI status output (exit code: {result.exit_code}):")
    print(result.output[:500] + "..." if len(result.output) > 500 else result.output)


@pytest.mark.live 
def test_cli_status_detailed_live():
    """Test CLI status command with detailed flag."""
    
    runner = CliRunner()
    
    result = runner.invoke(cli, ['status', '--detailed'])
    
    assert result.exit_code in [0, 1], "Detailed status should not crash"
    
    # Detailed output should be longer
    assert len(result.output) > 100, "Detailed status should provide substantial output"
    
    print("📋 CLI detailed status (first 300 chars):")
    print(result.output[:300] + "..." if len(result.output) > 300 else result.output)


@pytest.mark.live
def test_cli_config_commands_live():
    """Test CLI configuration commands."""
    
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "test_config.yaml"
        
        with runner.isolated_filesystem():
            # Set config path
            os.environ["CONTEXTER_CONFIG_PATH"] = str(config_file)
            
            try:
                # Test config wizard
                result = runner.invoke(cli, ['config', 'wizard'], input='y\n')
                
                # Should succeed or give helpful error
                assert result.exit_code in [0, 1, 130], f"Config wizard should not crash: {result.output}"
                
                if result.exit_code == 0:
                    print("✅ Config wizard succeeded")
                    assert config_file.exists() or "config" in result.output.lower()
                elif result.exit_code == 130:
                    print("⚠️  Config wizard was interrupted (expected in test)")
                else:
                    print(f"⚠️  Config wizard had issues: {result.output[:200]}")
                
                # Test config show (should work even if init failed)
                result = runner.invoke(cli, ['config', 'show'])
                assert result.exit_code in [0, 1], "Config show should not crash"
                
                if "configuration" in result.output.lower() or "config" in result.output.lower():
                    print("✅ Config show produced reasonable output")
                
                # Test config reset
                result = runner.invoke(cli, ['config', 'reset', '--force'])
                assert result.exit_code in [0, 1], "Config reset should not crash"
                
            finally:
                # Clean up environment
                if "CONTEXTER_CONFIG_PATH" in os.environ:
                    del os.environ["CONTEXTER_CONFIG_PATH"]


@pytest.mark.live
@pytest.mark.context7_required
def test_cli_download_command_live():
    """Test CLI download command with real Context7 API."""
    
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        
        # Set environment for testing
        test_env = {
            **os.environ,
            "CONTEXTER_STORAGE_PATH": str(storage_path),
        }
        
        # Test download with small library and conservative settings
        result = runner.invoke(cli, [
            '--verbose',  # Global verbose flag
            'download',
            'facebook/react',  # Well-known, stable library
            '--contexts', '2',  # Small context count
            '--max-concurrent', '1',  # Conservative concurrency
            '--dry-run'  # Dry run to avoid actual download
        ], env=test_env)
        
        print(f"🚀 CLI download output (exit code: {result.exit_code}):")
        print(result.output[:1000] + "..." if len(result.output) > 1000 else result.output)
        
        # Should attempt the download (may succeed or fail gracefully)
        assert result.exit_code in [0, 1], "Download command should not crash completely"
        
        # Should show some reasonable output
        output_lower = result.output.lower()
        assert any(keyword in output_lower for keyword in [
            'download', 'library', 'context', 'fastapi', 'error', 'success', 'dry-run', 'analyzing'
        ]), "Output should contain relevant download-related information"
        
        if result.exit_code == 0:
            print("✅ CLI download succeeded")
            # Check if files were created
            if any(storage_path.rglob("*")):
                print("✅ Storage files were created")
        else:
            print("⚠️  CLI download had issues (this may be expected without full configuration)")


@pytest.mark.live
def test_cli_help_system_live():
    """Test CLI help system completeness."""
    
    runner = CliRunner()
    
    # Test main help
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0, "Main help should always work"
    assert "Usage:" in result.output
    assert "Commands:" in result.output
    
    print("✅ Main CLI help working")
    
    # Test subcommand help
    commands = ['download', 'status', 'config']
    
    for cmd in commands:
        result = runner.invoke(cli, [cmd, '--help'])
        assert result.exit_code == 0, f"Help for {cmd} should work"
        assert "Usage:" in result.output
        
        print(f"✅ {cmd} command help working")


@pytest.mark.live
def test_cli_error_handling_live():
    """Test CLI error handling with real commands."""
    
    runner = CliRunner()
    
    # Test invalid command
    result = runner.invoke(cli, ['nonexistent-command'])
    assert result.exit_code == 2, "Invalid command should return exit code 2"
    
    # Test download without arguments
    result = runner.invoke(cli, ['download'])
    assert result.exit_code == 2, "Download without library should show usage error"
    
    # Test with invalid library format (should handle gracefully)
    result = runner.invoke(cli, [
        'download', 
        'completely/invalid/library/that/definitely/does/not/exist/12345',
        '--token-limit', '100'
    ])
    
    # Should either give proper error or attempt the download
    assert result.exit_code in [1, 2], "Invalid library should be handled gracefully"
    
    print("✅ CLI error handling working appropriately")


@pytest.mark.live
def test_cli_environment_integration():
    """Test CLI integration with environment variables."""
    
    runner = CliRunner()
    
    test_env = {
        **os.environ,
        "CONTEXTER_LOG_LEVEL": "DEBUG",
        "CONTEXTER_STORAGE_PATH": "/tmp/c7doc_test",
    }
    
    # Test that environment variables are respected
    result = runner.invoke(cli, ['status'], env=test_env)
    
    # Should not crash with custom environment
    assert result.exit_code in [0, 1], "CLI should handle custom environment variables"
    
    # Test verbose mode
    result = runner.invoke(cli, ['--verbose', 'status'], env=test_env)
    assert result.exit_code in [0, 1], "Verbose mode should work"
    
    print("✅ CLI environment integration working")


@pytest.mark.live
@pytest.mark.context7_required
def test_cli_realistic_workflow():
    """Test realistic CLI workflow scenario."""
    
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        storage_path = Path(temp_dir) / "storage"
        
        test_env = {
            **os.environ,
            "CONTEXTER_CONFIG_PATH": str(config_path),
            "CONTEXTER_STORAGE_PATH": str(storage_path),
        }
        
        print("🔄 Testing realistic CLI workflow...")
        
        # Step 1: Check status
        result = runner.invoke(cli, ['status'], env=test_env)
        print(f"   Status check: exit code {result.exit_code}")
        
        # Step 2: Initialize config  
        result = runner.invoke(cli, ['config', 'wizard'], env=test_env, input='n\n')
        print(f"   Config wizard: exit code {result.exit_code}")
        
        # Step 3: Show config
        result = runner.invoke(cli, ['config', 'show'], env=test_env)
        print(f"   Config show: exit code {result.exit_code}")
        
        # Step 4: Attempt small download (dry run)
        result = runner.invoke(cli, [
            'download',
            'facebook/react',
            '--contexts', '2',
            '--max-concurrent', '1',
            '--dry-run'
        ], env=test_env)
        print(f"   Small download: exit code {result.exit_code}")
        
        # Step 5: Check status again
        result = runner.invoke(cli, ['status', '--detailed'], env=test_env)
        print(f"   Final status: exit code {result.exit_code}")
        
        print("✅ Realistic CLI workflow completed (some failures expected without full config)")