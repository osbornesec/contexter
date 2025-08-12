"""
Test CLI validation utilities
"""

import pytest
import tempfile
from pathlib import Path

from contexter.cli.utils.validation import (
    validate_library_name,
    validate_output_directory,
    validate_concurrency_limit,
    validate_proxy_mode,
    validate_config_key
)


class TestLibraryNameValidation:
    """Test library name validation"""
    
    def test_valid_library_names(self):
        """Test valid library names"""
        valid_names = [
            'fastapi',
            'django',
            'react',
            'vue.js',
            'my-library',
            'my_library',
            'library123',
            # Context7 format (username/library)
            'context7/python-3',
            'python/cpython',
            'microsoft/vscode-python',
            'user/repo',
            'org/multi-word-lib'
        ]
        
        for name in valid_names:
            is_valid, error = validate_library_name(name)
            assert is_valid, f"'{name}' should be valid, but got error: {error}"
            assert error is None
    
    def test_invalid_library_names(self):
        """Test invalid library names"""
        invalid_names = [
            '',
            ' ',
            'library with spaces',
            'library@special',
            'library#hashtag',
            'library$dollar',
            'library%percent',
            'library(parentheses)',
            'library[brackets]',
            'a' * 101  # Too long
        ]
        
        for name in invalid_names:
            is_valid, error = validate_library_name(name)
            assert not is_valid, f"'{name}' should be invalid"
            assert error is not None


class TestOutputDirectoryValidation:
    """Test output directory validation"""
    
    def test_valid_existing_directory(self):
        """Test validation of existing directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            is_valid, error, path = validate_output_directory(temp_dir)
            assert is_valid, f"Error: {error}"
            assert error is None
            assert path == Path(temp_dir).resolve()
    
    def test_valid_new_directory_creation(self):
        """Test creation of new directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / 'new_directory'
            is_valid, error, path = validate_output_directory(str(new_dir))
            
            assert is_valid, f"Error: {error}"
            assert error is None
            assert path == new_dir.resolve()
            assert new_dir.exists()
    
    def test_invalid_path(self):
        """Test invalid paths"""
        # Try to create in non-existent parent
        invalid_path = "/non/existent/parent/directory"
        is_valid, error, path = validate_output_directory(invalid_path)
        
        assert not is_valid
        assert error is not None
        assert path is None


class TestConcurrencyLimitValidation:
    """Test concurrency limit validation"""
    
    def test_valid_concurrency_limits(self):
        """Test valid concurrency limits"""
        valid_limits = [1, 5, 10, 25, 50]
        
        for limit in valid_limits:
            is_valid, error = validate_concurrency_limit(limit)
            assert is_valid, f"Limit {limit} should be valid, but got error: {error}"
            assert error is None
    
    def test_invalid_concurrency_limits(self):
        """Test invalid concurrency limits"""
        invalid_limits = [0, -1, 51, 100]
        
        for limit in invalid_limits:
            is_valid, error = validate_concurrency_limit(limit)
            assert not is_valid, f"Limit {limit} should be invalid"
            assert error is not None


class TestProxyModeValidation:
    """Test proxy mode validation"""
    
    def test_valid_proxy_modes(self):
        """Test valid proxy modes"""
        valid_modes = ['auto', 'brightdata', 'none']
        
        for mode in valid_modes:
            is_valid, error = validate_proxy_mode(mode)
            assert is_valid, f"Mode '{mode}' should be valid, but got error: {error}"
            assert error is None
    
    def test_invalid_proxy_modes(self):
        """Test invalid proxy modes"""
        invalid_modes = ['', 'invalid', 'AUTO', 'None', 'bright-data']
        
        for mode in invalid_modes:
            is_valid, error = validate_proxy_mode(mode)
            assert not is_valid, f"Mode '{mode}' should be invalid"
            assert error is not None


class TestConfigKeyValidation:
    """Test configuration key validation"""
    
    def test_valid_config_keys(self):
        """Test valid configuration keys"""
        valid_keys = [
            'key',
            'proxy.customer_id',
            'storage.base_path',
            'download.max_concurrent',
            'nested.deep.key'
        ]
        
        for key in valid_keys:
            is_valid, error = validate_config_key(key)
            assert is_valid, f"Key '{key}' should be valid, but got error: {error}"
            assert error is None
    
    def test_invalid_config_keys(self):
        """Test invalid configuration keys"""
        invalid_keys = [
            '',
            '.key',
            'key.',
            'key..nested',
            '123key',
            'key with spaces',
            'key-with-dashes'
        ]
        
        for key in invalid_keys:
            is_valid, error = validate_config_key(key)
            assert not is_valid, f"Key '{key}' should be invalid"
            assert error is not None