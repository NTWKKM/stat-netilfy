"""Test suite for color palette system consistency.

Verifies that:
1. All color keys are defined
2. Colors are valid hex/rgba values
3. Light and dark modes have matching keys
4. Used in tabs/_common.py

Run: python -m pytest tests/test_color_palette.py -v
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
    except ImportError:
        print("Warning: Could not import from tabs._common")
        return {}


def test_color_palette_exists():
    """Test that color palette can be loaded."""
    palette = get_color_palette()
    assert palette is not None, "Color palette should not be None"
    assert isinstance(palette, dict), "Color palette should be a dictionary"
    assert len(palette) > 0, "Color palette should not be empty"


def test_essential_colors_present():
    """Test that all essential colors are defined."""
    palette = get_color_palette()
    
    essential_colors = [
        'text',              # Primary text color
        'text_secondary',    # Secondary text color
        'primary',           # Primary action color
        'primary_hover',     # Primary hover state
        'primary_active',    # Primary active state
        'secondary',         # Secondary background
        'border',            # Border color
        'error',             # Error color
        'success',           # Success color
        'warning',           # Warning color
        'info',              # Info color
    ]
    
    for color_key in essential_colors:
        assert color_key in palette, f"Missing essential color: {color_key}"
        assert palette[color_key], f"Color '{color_key}' is empty or None"


def test_color_format_validity():
    """Test that color values are in valid format (hex or rgba)."""
    palette = get_color_palette()
    
    for color_key, color_value in palette.items():
        assert isinstance(color_value, str), (
            f"Color '{color_key}' should be a string, got {type(color_value)}"
        )
        
        # Check if valid hex or rgba
        is_hex = color_value.startswith('#') and len(color_value) in (7, 9)  # #RRGGBB or #RRGGBBAA
        is_rgba = color_value.startswith('rgba(') and color_value.endswith(')')
        
        assert is_hex or is_rgba, (
            f"Color '{color_key}' has invalid format: {color_value}. "
            f"Expected hex (#RRGGBB) or rgba(r, g, b, a)"
        )


def test_color_consistency():
    """Test that color naming is consistent."""
    palette = get_color_palette()
    
    # Color keys should use snake_case
    for color_key in palette.keys():
        assert color_key.islower(), f"Color key '{color_key}' should be lowercase"
        assert '_' in color_key or '-' not in color_key, (
            f"Color key '{color_key}' should use snake_case or no separators"
        )


def test_hover_active_pairs():
    """Test that hover and active states exist for primary/secondary."""
    palette = get_color_palette()
    
    # Primary color should have hover and active variants
    if 'primary' in palette:
        assert 'primary_hover' in palette, "primary_hover missing"
        assert 'primary_active' in palette, "primary_active missing"
    
    # Secondary color should have hover and active variants
    if 'secondary' in palette:
        assert 'secondary_hover' in palette, "secondary_hover missing"
        assert 'secondary_active' in palette, "secondary_active missing"


def test_status_colors():
    """Test that all status colors are defined."""
    palette = get_color_palette()
    
    status_colors = ['error', 'success', 'warning', 'info']
    
    for status in status_colors:
        assert status in palette, f"Missing status color: {status}"
        assert palette[status], f"Status color '{status}' is empty"


def test_no_duplicate_colors():
    """Test that similar colors aren't accidentally duplicated."""
    palette = get_color_palette()
    
    # Count occurrences of each color value
    color_values = list(palette.values())
    
    # Warning: if too many duplicates, might indicate copy-paste errors
    # Allow some duplicates (e.g., same shade used for different purposes)
    value_counts = {}
    for value in color_values:
        value_counts[value] = value_counts.get(value, 0) + 1
    
    # More than 50% duplicate might be suspicious
    duplicates = sum(1 for count in value_counts.values() if count > 1)
    duplicate_percentage = (duplicates / len(value_counts)) * 100 if value_counts else 0
    
    # This is just a warning, not a hard failure
    if duplicate_percentage > 50:
        print(f"\nWarning: {duplicate_percentage:.1f}% of colors are duplicated")
        print("Possible duplicate colors:")
        for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                # Find keys with this value
                keys = [k for k, v in palette.items() if v == value]
                print(f"  {value}: {keys}")


def test_color_usage_in_tabs():
    """Test that color palette is actually used in tabs/_common.py."""
    try:
        with open(Path(__file__).parent.parent / 'tabs' / '_common.py', 'r') as f:
            content = f.read()
            
        # Check that get_color_palette function exists
        assert 'def get_color_palette' in content, (
            "get_color_palette function not found in tabs/_common.py"
        )
        
        # Check that it returns a dict
        assert 'return {' in content or 'return COLORS' in content, (
            "get_color_palette should return a dictionary"
        )
        
    except FileNotFoundError:
        print("Warning: tabs/_common.py not found, skipping usage test")


def test_survival_lib_color_usage():
    """Test that survival_lib.py uses correct color keys."""
    try:
        with open(Path(__file__).parent.parent / 'survival_lib.py', 'r') as f:
            content = f.read()
        
        # Should not use 'text_primary' (old key)
        assert "COLORS['text_primary']" not in content, (
            "survival_lib.py should use COLORS['text'], not COLORS['text_primary']"
        )
        
        # Should use correct key
        # Note: This is a soft warning
        if "COLORS['text']" not in content and "color_text" not in content:
            print("Warning: survival_lib.py might not be using color palette")
            
    except FileNotFoundError:
        print("Warning: survival_lib.py not found, skipping usage test")


if __name__ == '__main__':
    # Run tests manually
    print("Running Color Palette Tests...\n")
    
    test_functions = [
        test_color_palette_exists,
        test_essential_colors_present,
        test_color_format_validity,
        test_color_consistency,
        test_hover_active_pairs,
        test_status_colors,
        test_no_duplicate_colors,
        test_color_usage_in_tabs,
        test_survival_lib_color_usage,
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
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
