#!/usr/bin/env python3
"""Fixture: off-by-one in range bound.

GROUND TRUTH (recorded at creation — scoring is computed against this):
- defect_class: off-by-one
- location: line with `range(1, len(items))`
- expected_finding: the loop skips items[0]; should be range(len(items)) or range(0, ...)
- the rest of the function is known-clean
"""
def sum_items(items):
    total = 0
    for i in range(1, len(items)):
        total += items[i]
    return total
