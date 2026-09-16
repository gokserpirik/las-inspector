"""Tests for small pure helpers."""

from las_inspector import _cls_name, _fmt_num, _guess_record_len, _severity_icon, _unpack_bitfield


def test_unpack_bitfield_classification_only():
    flags = _unpack_bitfield(0x05)
    assert flags == {"synthetic": False, "keypoint": False, "withheld": False, "classification": 5}


def test_unpack_bitfield_all_flags():
    flags = _unpack_bitfield(0x80 | 0x40 | 0x20 | 0x02)
    assert flags["synthetic"] is True
    assert flags["keypoint"] is True
    assert flags["withheld"] is True
    assert flags["classification"] == 2


def test_unpack_bitfield_classification_masked_to_5_bits():
    assert _unpack_bitfield(0xFF)["classification"] == 0x1F


def test_guess_record_len_known_formats():
    assert _guess_record_len(0) == 20
    assert _guess_record_len(3) == 34
    assert _guess_record_len(6) == 30
    assert _guess_record_len(10) == 67


def test_guess_record_len_unknown_defaults_to_30():
    assert _guess_record_len(99) == 30


def test_severity_icon_known():
    assert _severity_icon("info") == "ℹ️"
    assert _severity_icon("warning") == "⚠️"
    assert _severity_icon("error") == "🚫"


def test_severity_icon_unknown_defaults_to_info():
    assert _severity_icon("critical") == "ℹ️"


def test_fmt_num_int():
    assert _fmt_num(999) == "999"
    assert _fmt_num(1_500) == "1.5k"
    assert _fmt_num(2_000_000) == "2.00M"


def test_fmt_num_float():
    assert _fmt_num(500.0) == "500.00"
    assert _fmt_num(2_500.0) == "2.5k"
    assert _fmt_num(3_000_000.0) == "3.00M"


def test_cls_name_known_and_unknown():
    assert _cls_name(2) == "Ground"
    assert _cls_name(6) == "Building"
    assert _cls_name(99) == "Class 99"
