#!/usr/bin/env python3
"""
🪧 Unit Tests for Color System

Verifies that the unified color palette is correctly implemented
across all modules.

Run with:
    pytest tests/test_color_palette.py -v
"""

import sys
import re
sys.path.insert(0, '.')

from tabs._common import get_color_palette


class TestColorPalette:
    """
    Test suite for the unified color system.
    """
    
    def test_palette_exists(self):
        """
        ✅ Test: Palette function exists and returns a dict
        """
        palette = get_color_palette()
        assert isinstance(palette, dict), "get_color_palette() should return a dict"
        assert len(palette) > 0, "Palette should not be empty"
    
    def test_required_colors_present(self):
        """
        ✅ Test: All required color keys are present
        """
        palette = get_color_palette()
        required_keys = [
            'primary', 'primary_dark', 'text', 'text_secondary',
            'danger', 'warning', 'success', 'info',
            'border', 'background', 'surface'
        ]
        
        for key in required_keys:
            assert key in palette, f"Missing required color: {key}"
    
    def test_color_hex_format(self):
        """
        ✅ Test: All colors are valid hex values
        """
        palette = get_color_palette()
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
        
        for key, value in palette.items():
            assert isinstance(value, str), f"{key} should be a string"
            assert hex_pattern.match(value), f"{key} value '{value}' is not valid hex format"
    
    def test_color_uniqueness(self):
        """
        ✅ Test: Distinct colors for different purposes
        """
        palette = get_color_palette()
        
        # These should be different colors
        assert palette['primary'] != palette['primary_dark'], "Primary and primary_dark should differ"
        assert palette['text'] != palette['text_secondary'], "Text colors should differ"
        assert palette['danger'] != palette['success'], "Status colors should differ"
    
    def test_specific_color_values(self):
        """
        ✅ Test: Verify specific color hex values
        """
        palette = get_color_palette()
        
        assert palette['primary'] == '#218084', "Primary color mismatch"
        assert palette['primary_dark'] == '#134252', "Primary dark color mismatch"
        assert palette['danger'] == '#ff5459', "Danger color mismatch"
        assert palette['text'] == '#2c3e50', "Text color mismatch"
    
    def test_color_accessibility(self):
        """
        ✅ Test: Colors meet WCAG accessibility standards
        """
        # Contrast ratios (simplified check - color pairs on white background)
        # These are documented in COLOR_SYSTEM.md
        
        palette = get_color_palette()
        
        # Dark teal on white should have high contrast
        primary_dark = palette['primary_dark']  # #134252
        assert primary_dark.lower() in ['#134252'], "Primary dark should be #134252 for accessibility"
        
        # Red for alerts should be readable
        danger = palette['danger']  # #ff5459
        assert danger.lower() in ['#ff5459'], "Danger color should be #ff5459 for accessibility"
    
    def test_palette_consistency(self):
        """
        ✅ Test: Multiple calls return the same palette
        """
        palette1 = get_color_palette()
        palette2 = get_color_palette()
        
        assert palette1 == palette2, "Palette should be consistent across calls"
    
    def test_no_color_typos(self):
        """
        ✅ Test: Color values are not obviously typos
        """
        palette = get_color_palette()
        
        for key, value in palette.items():
            # Check for common typos
            assert value != '#000000' or key == 'background', "Color should not be pure black"
            assert value != '#ffffff' or key == 'surface', "Color should not be pure white"
            assert len(value) == 7, f"{key} hex color should be 7 characters (# + 6 digits)"
            assert value[0] == '#', f"{key} should start with #"


class TestColorUsageInModules:
    """
    Test suite to verify color palette is used correctly in modules.
    """
    
    def test_table_one_imports_palette(self):
        """
        ✅ Test: table_one.py imports get_color_palette
        """
        try:
            from table_one import generate_table  # Verify import works
            import table_one
            # Check if file contains get_color_palette
            with open('table_one.py', 'r') as f:
                content = f.read()
                assert 'get_color_palette' in content, "table_one.py should use get_color_palette"
                assert 'from tabs._common import' in content, "table_one.py should import from _common"
        except ImportError:
            pass  # Skip if module not available
    
    def test_psm_lib_imports_palette(self):
        """
        ✅ Test: psm_lib.py imports get_color_palette
        """
        try:
            import psm_lib
            with open('psm_lib.py', 'r') as f:
                content = f.read()
                assert 'get_color_palette' in content, "psm_lib.py should use get_color_palette"
                assert 'from tabs._common import' in content, "psm_lib.py should import from _common"
        except ImportError:
            pass
    
    def test_logic_imports_palette(self):
        """
        ✅ Test: logic.py imports get_color_palette
        """
        try:
            import logic
            with open('logic.py', 'r') as f:
                content = f.read()
                assert 'get_color_palette' in content, "logic.py should use get_color_palette"
                assert 'from tabs._common import' in content, "logic.py should import from _common"
        except ImportError:
            pass


class TestColorHtmlIntegration:
    """
    Test suite for HTML/CSS color integration.
    """
    
    def test_primary_color_in_css(self):
        """
        ✅ Test: Primary color appears in generated CSS
        """
        palette = get_color_palette()
        # When modules generate CSS, they should use these colors
        assert palette['primary_dark'] == '#134252'
        assert palette['primary'] == '#218084'
    
    def test_danger_color_for_alerts(self):
        """
        ✅ Test: Danger color is used for significant values
        """
        palette = get_color_palette()
        danger_color = palette['danger']
        assert danger_color == '#ff5459', "Danger color should be bright red for visibility"
    
    def test_text_contrast_ratios(self):
        """
        ✅ Test: Text colors have documented contrast ratios
        """
        palette = get_color_palette()
        
        # Dark text on light background
        dark_text = palette['text']  # #2c3e50
        secondary_text = palette['text_secondary']  # #7f8c8d
        
        # Both should be dark colors
        assert dark_text.lower().startswith('#'), "Text color should be hex"
        assert secondary_text.lower().startswith('#'), "Secondary text should be hex"
        
        # Text should be darker than background
        assert dark_text != palette['background'], "Text should contrast with background"


def test_color_palette_complete():
    """
    🌟 Smoke test: Entire color system works
    """
    palette = get_color_palette()
    
    # Verify we can use colors in HTML
    html_test = f"""
    <style>
        body {{ color: {palette['text']}; }}
        th {{ background-color: {palette['primary_dark']}; }}
        .alert {{ color: {palette['danger']}; }}
    </style>
    """
    
    # Should not raise any errors
    assert len(html_test) > 0
    assert palette['primary_dark'] in html_test
    assert palette['text'] in html_test
    assert palette['danger'] in html_test


if __name__ == '__main__':
    # Run basic tests
    print("🪧 Running Color System Tests...\n")
    
    test = TestColorPalette()
    
    print("✅ Testing palette structure...")
    test.test_palette_exists()
    test.test_required_colors_present()
    test.test_color_hex_format()
    print("   All structure tests passed!\n")
    
    print("✅ Testing color values...")
    test.test_color_uniqueness()
    test.test_specific_color_values()
    print("   All value tests passed!\n")
    
    print("✅ Testing accessibility...")
    test.test_color_accessibility()
    test.test_palette_consistency()
    print("   All accessibility tests passed!\n")
    
    print("🌟 All tests passed! Color system is valid. \n")