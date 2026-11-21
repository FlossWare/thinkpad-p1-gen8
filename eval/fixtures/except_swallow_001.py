#!/usr/bin/env python3
"""Fixture: bare except swallows all errors including KeyboardInterrupt.

GROUND TRUTH:
- defect_class: exception_swallowing
- location: `except: pass`
- expected_finding: bare except catches everything (including
  KeyboardInterrupt/SystemExit) and silently discards it. Catch the specific
  exception and at minimum log it.
- the rest is known-clean
"""
def safe_divide(a, b):
    try:
        return a / b
    except:
        pass
