"""
Test main CLI functionality
"""

import pytest
from click.testing import CliRunner

from contexter.cli.main import cli


def test_cli_help():
    """Test main CLI help display"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    
    assert result.exit_code == 0
    assert 'Context7 Documentation Downloader' in result.output
    assert 'High-performance CLI' in result.output


def test_cli_version():
    """Test CLI version display"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    
    assert result.exit_code == 0
    assert '1.0.0' in result.output


def test_cli_verbose_flag():
    """Test verbose flag parsing"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--verbose', '--help'])
    
    assert result.exit_code == 0
    assert 'Context7 Documentation Downloader' in result.output


def test_cli_no_color_flag():
    """Test no-color flag parsing"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--no-color', '--help'])
    
    assert result.exit_code == 0
    assert 'Context7 Documentation Downloader' in result.output


def test_cli_subcommands_available():
    """Test that all expected subcommands are available"""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    
    assert result.exit_code == 0
    assert 'download' in result.output
    assert 'config' in result.output
    assert 'status' in result.output


def test_download_command_help():
    """Test download command help"""
    runner = CliRunner()
    result = runner.invoke(cli, ['download', '--help'])
    
    assert result.exit_code == 0
    assert 'Download comprehensive documentation' in result.output
    assert 'LIBRARY' in result.output


def test_config_command_help():
    """Test config command help"""
    runner = CliRunner()
    result = runner.invoke(cli, ['config', '--help'])
    
    assert result.exit_code == 0
    assert 'Configuration management' in result.output


def test_status_command_help():
    """Test status command help"""
    runner = CliRunner()
    result = runner.invoke(cli, ['status', '--help'])
    
    assert result.exit_code == 0
    assert 'Check system status' in result.output