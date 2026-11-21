#!/usr/bin/env python3
"""Fixture: string/int type confusion in comparison.

GROUND TRUTH:
- defect_class: type_confusion
- location: `if response_code == 200:` where response_code is documented
  (and assigned) as a string
- expected_finding: response_code is assigned as a string "200"; comparing to
  int 200 is always False in Python. Convert or compare consistently.
- the rest is known-clean
"""
def is_ok(response_code):
    response_code = str(response_code)
    if response_code == 200:
        return True
    return False
