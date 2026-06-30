"""
UI utilities for KGClaw — shared progress callbacks and display helpers.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from .display import (
    ICON,
    STYLE,
    print_banner,
    print_error,
    print_extraction_result,
    print_ontology,
    print_ontology_summary,
    print_section,
    print_stats,
    print_success,
    print_warning,
)
from .progress import make_progress_callback

__all__ = [
    "make_progress_callback",
    "print_banner",
    "print_ontology",
    "print_ontology_summary",
    "print_section",
    "print_success",
    "print_error",
    "print_warning",
    "print_extraction_result",
    "print_stats",
    "STYLE",
    "ICON",
]
