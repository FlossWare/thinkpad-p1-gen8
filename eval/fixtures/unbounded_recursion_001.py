#!/usr/bin/env python3
"""Fixture: missing base case — unbounded recursion.

GROUND TRUTH:
- defect_class: unbounded_recursion
- location: the recursive call in `def factorial(n):`
- expected_finding: no base case for n <= 0 (or the condition is absent
  entirely) — negative or zero input recurses forever (until stack overflow).
  Needs `if n <= 1: return 1`.
- the rest is known-clean
"""
def factorial(n):
    return n * factorial(n - 1)
