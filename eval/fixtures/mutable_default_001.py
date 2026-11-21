#!/usr/bin/env python3
"""Fixture: mutable default argument shared across calls.

GROUND TRUTH:
- defect_class: mutable_default_argument
- location: `def add_item(item, items=[]):`
- expected_finding: the default list is created once and shared across all
  calls — items accumulate between calls. Use `items=None` + `if items is
  None: items = []`.
- the rest is known-clean
"""
def add_item(item, items=[]):
    items.append(item)
    return items
