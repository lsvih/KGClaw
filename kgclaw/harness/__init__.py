"""
Harness package — KG construction workflow orchestrator.

Re-exports the Harness class and phase metadata constants
for backward compatibility with `from kgclaw.harness import Harness`.
"""
# Copyright (c) 2026 Yanzeng Li @ BNU. MIT License.


from .engine import Harness, PHASE_LABELS, PHASE_WEIGHTS, TOTAL_PHASE_WEIGHT, phase_label

__all__ = ["Harness", "PHASE_LABELS", "PHASE_WEIGHTS", "TOTAL_PHASE_WEIGHT", "phase_label"]
