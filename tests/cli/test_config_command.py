"""
Test configuration command functionality
"""

import pytest
from click.testing import CliRunner

from contexter.cli.main import cli


class TestConfigCommand:
    """Test config command"""
    
    def test_config_help(self):
        """Test config command help display"""
        runner = CliRunner()
        result = runner.invoke(cli, ['config', '--help'])
        
        assert result.exit_code == 0
        assert 'Configuration management' in result.output
        assert 'show' in result.output
        assert 'set' in result.output
        assert 'reset' in result.output
        assert 'wizard' in result.output
    
    def test_config_show_help(self):
        """Test config show subcommand help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'show', '--help'])
        
        assert result.exit_code == 0
        assert 'Display current configuration' in result.output
    
    def test_config_set_help(self):
        """Test config set subcommand help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'set', '--help'])
        
        assert result.exit_code == 0
        assert 'Set a configuration value' in result.output
        assert '--key' in result.output
        assert '--value' in result.output
    
    def test_config_reset_help(self):
        """Test config reset subcommand help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'reset', '--help'])
        
        assert result.exit_code == 0
        assert 'Reset configuration' in result.output
        assert '--force' in result.output
    
    def test_config_wizard_help(self):
        """Test config wizard subcommand help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'wizard', '--help'])
        
        assert result.exit_code == 0
        assert 'Interactive configuration' in result.output
    
    def test_config_set_missing_required_options(self):
        """Test config set with missing required options"""
        runner = CliRunner()
        
        # Missing both key and value
        result = runner.invoke(cli, ['config', 'set'])
        assert result.exit_code == 2
        assert 'Missing option' in result.output
        
        # Missing value
        result = runner.invoke(cli, ['config', 'set', '--key', 'test'])
        assert result.exit_code == 2
        assert 'Missing option' in result.output
        
        # Missing key
        result = runner.invoke(cli, ['config', 'set', '--value', 'test'])
        assert result.exit_code == 2
        assert 'Missing option' in result.output
    
    def test_config_set_invalid_key_format(self):
        """Test config set with invalid key format"""
        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'set', '--key', 'invalid key', '--value', 'test'])
        
        assert result.exit_code == 1  # Validation errors return exit code 1
        assert 'Validation Error' in result.output