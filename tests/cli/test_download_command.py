"""
Test download command functionality
"""

import pytest
from click.testing import CliRunner

from contexter.cli.main import cli


class TestDownloadCommand:
    """Test download command"""
    
    def test_download_help(self):
        """Test download command help display"""
        runner = CliRunner()
        result = runner.invoke(cli, ['download', '--help'])
        
        assert result.exit_code == 0
        assert 'Download comprehensive documentation' in result.output
        assert 'LIBRARY' in result.output
        assert '--output' in result.output
        assert '--max-concurrent' in result.output
        assert '--proxy-mode' in result.output
        assert '--dry-run' in result.output
        assert '--force' in result.output
    
    def test_download_missing_library_argument(self):
        """Test download command with missing library argument"""
        runner = CliRunner()
        result = runner.invoke(cli, ['download'])
        
        assert result.exit_code == 2
        assert 'Missing argument' in result.output
    
    def test_download_invalid_library_name(self):
        """Test download command with invalid library name"""
        runner = CliRunner()
        result = runner.invoke(cli, ['download', 'invalid@library'])
        
        assert result.exit_code == 1  # Validation errors return exit code 1
        assert 'Validation Error' in result.output
    
    def test_download_invalid_max_concurrent(self):
        """Test download command with invalid max concurrent value"""
        runner = CliRunner()
        result = runner.invoke(cli, ['download', 'fastapi', '--max-concurrent', '0'])
        
        assert result.exit_code == 1  # Validation errors return exit code 1
        assert 'Validation Error' in result.output
    
    def test_download_invalid_proxy_mode(self):
        """Test download command with invalid proxy mode"""
        runner = CliRunner()
        result = runner.invoke(cli, ['download', 'fastapi', '--proxy-mode', 'invalid'])
        
        assert result.exit_code == 2
        assert 'Invalid value for' in result.output
    
    def test_download_invalid_output_directory(self):
        """Test download command with invalid output directory"""
        runner = CliRunner()
        result = runner.invoke(cli, ['download', 'fastapi', '--output', '/non/existent/parent/dir'])
        
        assert result.exit_code == 1  # Validation errors return exit code 1
        assert 'Validation Error' in result.output
    
    def test_download_option_parsing(self):
        """Test that download command options are parsed correctly"""
        runner = CliRunner()
        
        # This will fail due to missing configuration, but options should parse
        result = runner.invoke(cli, [
            'download', 'fastapi',
            '--output', '/tmp',
            '--max-concurrent', '5',
            '--proxy-mode', 'none',
            '--dry-run'
        ])
        
        # Should not exit with argument parsing error (exit code 2)
        assert result.exit_code != 2