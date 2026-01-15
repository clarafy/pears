"""Unit tests for SequentialEncoder."""

import pytest

from pears.encoders import SequentialEncoder


class TestSequentialEncoder:
    """Test suite for SequentialEncoder."""

    def test_encode_decode(self):
        """Test basic encode/decode functionality and roundtrip."""
        encoder = SequentialEncoder(["A", "B", "C"])

        # Verify encode returns distinct integers for distinct labels
        codes = [encoder.encode("A"), encoder.encode("B"), encoder.encode("C")]
        assert len(set(codes)) == 3  # All different
        assert all(0 <= c < 3 for c in codes)  # All in valid range

        # Verify roundtrip: decode(encode(label)) == label
        for label in ["A", "B", "C"]:
            assert encoder.decode(encoder.encode(label)) == label

        # Verify symmetric: encode(decode(idx)) == idx
        for idx in range(3):
            assert encoder.encode(encoder.decode(idx)) == idx

    def test_deduplication(self):
        """Test that duplicates are removed during initialization."""
        encoder = SequentialEncoder(["A", "B", "A", "C"])
        assert len(encoder) == 3

    def test_validation(self):
        """Test is_valid_label and is_valid_idx methods."""
        encoder = SequentialEncoder(["X", "Y", "Z"])

        # Test is_valid_label
        assert encoder.is_valid_label("X") is True
        assert encoder.is_valid_label("Y") is True
        assert encoder.is_valid_label("Z") is True
        assert encoder.is_valid_label("W") is False

        # Test is_valid_idx
        assert encoder.is_valid_idx(0) is True
        assert encoder.is_valid_idx(1) is True
        assert encoder.is_valid_idx(2) is True
        assert encoder.is_valid_idx(5) is False

    def test_invalid_operations(self):
        """Test error handling for invalid encode/decode operations."""
        encoder = SequentialEncoder(["A", "B"])

        # Test encode with invalid label raises KeyError
        with pytest.raises(KeyError):
            encoder.encode("C")

        # Test decode with invalid index raises KeyError
        with pytest.raises(KeyError):
            encoder.decode(5)

    def test_deterministic_encoding(self):
        """Test that encoder produces consistent mappings when created multiple times with same input."""
        # Create two separate encoder instances with identical input
        encoder1 = SequentialEncoder(["C", "A", "B"])
        encoder2 = SequentialEncoder(["C", "A", "B"])  # Same exact input

        # Both should produce identical encodings
        assert encoder1.encode("A") == encoder2.encode("A")
        assert encoder1.encode("B") == encoder2.encode("B")
        assert encoder1.encode("C") == encoder2.encode("C")

        # Verify the sorted order explicitly (A=0, B=1, C=2)
        assert encoder1.encode("A") == 0
        assert encoder1.encode("B") == 1
        assert encoder1.encode("C") == 2
