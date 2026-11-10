"""
Parity tests for optimized metric calculations.

These tests verify that JIT-optimized versions produce identical results
to the original implementations. Once validated, these tests can be removed
and replaced with standard unit tests.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.utils.common.constants import (
    IS_PLUS_TRIGGER,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
)


def test_trials_to_true_parity():
    """Test JIT trials-to-true returns correct per-row wait times."""
    from src.utils.performance.perseries_.metricscalculations import calculate_trials_to_true

    # Test case 1: Mixed runs
    # [T, T, T, F, T, T, F, F, T] → wait times: [1, 1, 1, 0, 2, 1, 0, 0, 3]
    arr1 = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
    result1 = calculate_trials_to_true(arr1)
    expected1 = np.array([1, 1, 1, 0, 2, 1, 0, 0, 3], dtype=np.float32)
    assert np.allclose(result1, expected1), f"Expected {expected1}, got {result1}"

    # Test case 2: All true
    arr2 = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    result2 = calculate_trials_to_true(arr2)
    expected2 = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    assert np.allclose(result2, expected2), f"Expected {expected2}, got {result2}"

    # Test case 3: All false
    arr3 = np.array([0, 0, 0], dtype=np.int8)
    result3 = calculate_trials_to_true(arr3)
    expected3 = np.array([0, 0, 0], dtype=np.float32)
    assert np.allclose(result3, expected3), f"Expected {expected3}, got {result3}"

    # Test case 4: Empty array
    arr4 = np.array([], dtype=np.int8)
    result4 = calculate_trials_to_true(arr4)
    assert result4.size == 0, f"Expected empty array, got {result4}"

    # Test case 5: Single true
    arr5 = np.array([1], dtype=np.int8)
    result5 = calculate_trials_to_true(arr5)
    expected5 = np.array([1], dtype=np.float32)
    assert np.allclose(result5, expected5), f"Expected {expected5}, got {result5}"

    # Test case 6: Alternating
    # [T, F, T, F, T] → wait times: [1, 0, 2, 0, 2]
    arr6 = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    result6 = calculate_trials_to_true(arr6)
    expected6 = np.array([1, 0, 2, 0, 2], dtype=np.float32)
    assert np.allclose(result6, expected6), f"Expected {expected6}, got {result6}"

    print("✓ All trials-to-true parity tests passed")


def test_multiplicative_compound_parity():
    """Test JIT multiplicative compounding matches formula."""
    from src.utils.performance.perseries_.metricscalculations import _multiplicative_compound
    
    # Test case 1: Simple positive returns
    variations1 = np.array([10.0, 20.0, -5.0], dtype=np.float32)
    result1 = _multiplicative_compound(variations1)
    expected1 = ((1.10 * 1.20 * 0.95) - 1.0) * 100.0
    assert np.isclose(result1, expected1, rtol=1e-5), f"Expected {expected1}, got {result1}"
    
    # Test case 2: Large variations (from user's data)
    variations2 = np.array([354.85, 15.04], dtype=np.float32)
    result2 = _multiplicative_compound(variations2)
    expected2 = ((4.5485 * 1.1504) - 1.0) * 100.0
    assert np.isclose(result2, expected2, rtol=1e-4), f"Expected {expected2}, got {result2}"
    
    # Test case 3: Empty array
    variations3 = np.array([], dtype=np.float32)
    result3 = _multiplicative_compound(variations3)
    expected3 = 0.0
    assert np.isclose(result3, expected3), f"Expected {expected3}, got {result3}"
    
    # Test case 4: Single return
    variations4 = np.array([25.0], dtype=np.float32)
    result4 = _multiplicative_compound(variations4)
    expected4 = 25.0
    assert np.isclose(result4, expected4, rtol=1e-5), f"Expected {expected4}, got {result4}"
    
    # Test case 5: Multiple small returns
    variations5 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result5 = _multiplicative_compound(variations5)
    expected5 = ((1.01 * 1.02 * 1.03) - 1.0) * 100.0
    assert np.isclose(result5, expected5, rtol=1e-5), f"Expected {expected5}, got {result5}"
    
    print("✓ All multiplicative compounding parity tests passed")


def test_variation_contiguity():
    """Test that variation arrays are contiguous for cache efficiency."""
    import pandas as pd
    from src.utils.performance.perseries_.metricscalculations import calculate_variations
    
    df = pd.DataFrame({
        TRIGGER_START_VALUES: [100.0, 200.0, 300.0],
        TRIGGER_END_VALUES: [110.0, 190.0, 330.0],
        IS_PLUS_TRIGGER: [True, False, True],
    })
    
    result = calculate_variations(df)
    
    # Check contiguity
    assert result.flags['C_CONTIGUOUS'], "Result array is not C-contiguous"
    
    # Check dtype
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    
    # Check values
    expected = np.array([10.0, 5.0, 10.0], dtype=np.float32)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    
    print("✓ Variation contiguity test passed")


def test_duration_contiguity():
    """Test that duration arrays are contiguous."""
    import pandas as pd
    from src.utils.performance.perseries_.metricscalculations import calculate_durations
    
    # Create mock data
    dates = pd.bdate_range('2020-01-01', periods=10)
    original_df = pd.DataFrame({'value': range(10)}, index=dates)
    
    triggers_df = pd.DataFrame({
        TRIGGER_BEGIN_DATES: [dates[0], dates[2]],
        TRIGGER_END_DATES: [dates[3], dates[5]],
    })
    
    result = calculate_durations(triggers_df, original_df)
    
    # Check contiguity
    assert result.flags['C_CONTIGUOUS'], "Result array is not C-contiguous"
    
    # Check dtype
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    
    # Check values
    expected = np.array([3.0, 3.0], dtype=np.float32)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    
    print("✓ Duration contiguity test passed")


if __name__ == "__main__":
    test_trials_to_true_parity()
    test_multiplicative_compound_parity()
    test_variation_contiguity()
    test_duration_contiguity()
    print("\n✓✓✓ All parity tests passed! ✓✓✓")
