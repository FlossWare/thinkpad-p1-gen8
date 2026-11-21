#!/usr/bin/env python3
"""Fixture: check-then-act race on shared dict.

GROUND TRUTH:
- defect_class: race_condition (TOCTOU)
- location: the `if key not in cache:` block
- expected_finding: check-then-act race — two threads can both see the key
  missing and both compute/insert. Needs a lock or atomic setdefault.
- the rest is known-clean
"""
import threading

cache = {}

def get_or_compute(key, compute):
    if key not in cache:
        cache[key] = compute(key)
    return cache[key]
