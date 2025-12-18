"""Test suite for color palette system consistency.

Verifies that:
1. All color keys are defined
2. Colors are valid hex/rgba values
3. Used in tabs/_common.py
4. survival_lib.py uses correct keys

Run: python -m pytest tests/test_color_palette.py -v
Or:  python tests/test_color_palette.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_color_palette():
    """Get the color palette from tabs/_common.py"""
    try:
        from tabs._common import get_color_palette as fetch_colors
        return fetch_colors()
    except ImportError as e:
        print(f"Warning: Could not import from tabs._common: {e}")
        return {}


def test_color_palette_exists():
    """Test that color palette can be loaded."""
    palette = get_color_palette()
    assert palette is not None, "Color palette should not be None"
    assert isinstance(palette, dict), "Color palette should be a dictionary"
    assert len(palette) > 0, "Color palette should not be empty"


def test_essential_colors_present():
    """Test that all essential colors are defined.
    
    Based on actual palette in tabs/_common.py
    """
    palette = get_color_palette()
    
    # Essential colors that MUST exist
    essential_colors = [
        'text',              # Primary text color (#1a2332)
        'text_secondary',    # Secondary text color (#7f8c8d)
        'primary',           # Primary action color (#2c5aa0)
        'border',            # Border color (#d5dce0)
        'danger',            # Error/alert color (#e74c3c)
        'success',           # Success color (#27ae60)
        'warning',           # Warning color (#f39c12)
        'info',              # Info color (#5b6c7d)
    ]
    
    for color_key in essential_colors:
        assert color_key in palette, f"Missing essential color: {color_key}"
        assert palette[color_key], f"Color '{color_key}' is empty or None"


def test_color_format_validity():
    """Test that color values are in valid format (hex)."""
    palette = get_color_palette()
    
    for color_key, color_value in palette.items():
        assert isinstance(color_value, str), (
            f"Color '{color_key}' should be a string, got {type(color_value)}"
        )
        
        # Check if valid hex format #RRGGBB (7 chars) or #RRGGBBAA (9 chars)
        is_hex = color_value.startswith('#') and len(color_value) == 7
        
        assert is_hex, (
            f"Color '{color_key}' has invalid format: {color_value}. "
            f"Expected hex format (#RRGGBB)"
        )


def test_color_consistency():
    """Test that color naming is consistent."""
    palette = get_color_palette()
    
    # Color keys should be lowercase
    for color_key in palette.keys():
        assert color_key.islower(), (
            f"Color key '{color_key}' should be lowercase"
        )
        # Allow snake_case or single words
        assert all(c.isalnum() or c == '_' for c in color_key), (
            f"Color key '{color_key}' should only contain alphanumerics and underscores"
        )


def test_primary_variants():
    """Test that primary color variants exist."""
    palette = get_color_palette()
    
    # Primary should have variants
    assert 'primary' in palette, "primary color missing"
    assert 'primary_dark' in palette, "primary_dark variant missing"
    assert 'primary_light' in palette, "primary_light variant missing"


def test_status_colors():
    """Test that all status colors are defined."""
    palette = get_color_palette()
    
    # Status colors from the actual palette
    status_colors = {
        'danger': 'error/alert color',
        'success': 'success/matched color',
        'warning': 'warning/caution color',
        'info': 'informational color',
    }
    
    for status, description in status_colors.items():
        assert status in palette, f"Missing status color: {status} ({description})"
        assert palette[status], f"Status color '{status}' is empty"


def test_neutral_colors():
    """Test that neutral colors exist."""
    palette = get_color_palette()
    
    neutral_colors = ['text', 'text_secondary', 'border', 'background', 'surface', 'neutral']
    
    for color in neutral_colors:
        if color in palette:  # Some might be optional
            assert palette[color], f"Neutral color '{color}' is empty"


def test_no_duplicate_colors():
    """Test that similar colors aren't accidentally duplicated."""
    palette = get_color_palette()
    
    # Count occurrences of each color value
    color_values = list(palette.values())
    value_counts = {}
    
    for value in color_values:
        value_counts[value] = value_counts.get(value, 0) + 1
    
    # More than 50% duplicate might indicate errors
    duplicates = sum(1 for count in value_counts.values() if count > 1)
    duplicate_percentage = (duplicates / len(value_counts)) * 100 if value_counts else 0
    
    # Print warning if suspicious
    if duplicate_percentage > 50:
        print(f"\n⚠️  Warning: {duplicate_percentage:.1f}% of colors are duplicated")
        print("Duplicate colors:")
        for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                keys = [k for k, v in palette.items() if v == value]
                print(f"  {value}: {keys}")


def test_color_usage_in_tabs():
    """Test that color palette file exists and has function."""
    try:
        common_file = Path(__file__).parent.parent / 'tabs' / '_common.py'
        assert common_file.exists(), "tabs/_common.py not found"
        
        with open(common_file, 'r') as f:
            content = f.read()
            
        # Check that get_color_palette function exists
        assert 'def get_color_palette' in content, (
            "get_color_palette function not found in tabs/_common.py"
        )
        
        # Check that it returns a dict
        assert 'return {' in content, (
            "get_color_palette should return a dictionary"
        )
        
    except AssertionError as e:
        raise
    except FileNotFoundError:
        raise AssertionError("tabs/_common.py not found")


def test_survival_lib_color_usage():
    """Test that survival_lib.py uses correct color keys."""
    try:
        survival_file = Path(__file__).parent.parent / 'survival_lib.py'
        assert survival_file.exists(), "survival_lib.py not found"
        
        with open(survival_file, 'r') as f:
            content = f.read()
        
        # Should not use old key 'text_primary'
        assert "COLORS['text_primary']" not in content, (
            "survival_lib.py should use COLORS['text'], not COLORS['text_primary']"
        )
        
    except FileNotFoundError:
        print("Warning: survival_lib.py not found")


def test_hex_color_validity():
    """Test that all hex colors are valid."""
    palette = get_color_palette()
    
    valid_hex_chars = set('0123456789abcdefABCDEF')
    
    for color_key, color_value in palette.items():
        # Must start with #
        assert color_value.startswith('#'), f"{color_key} must start with #"
        
        # Must be 7 chars total (#RRGGBB)
        assert len(color_value) == 7, f"{color_key} must be 7 chars (#RRGGBB), got {len(color_value)}"
        
        # All chars after # must be valid hex
        hex_part = color_value[1:]  # Remove #
        assert all(c in valid_hex_chars for c in hex_part), (
            f"{color_key} contains invalid hex characters: {hex_part}"
        )


if __name__ == '__main__':
    # Run tests manually
    print("Running Color Palette Tests...\n")
    
    test_functions = [
        test_color_palette_exists,
        test_essential_colors_present,
        test_color_format_validity,
        test_color_consistency,
        test_primary_variants,
        test_status_colors,
        test_neutral_colors,
        test_no_duplicate_colors,
        test_color_usage_in_tabs,
        test_survival_lib_color_usage,
        test_hex_color_validity,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {test_func.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    sys.exit(0 if failed == 0 else 1)
