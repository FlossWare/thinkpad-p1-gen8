#!/usr/bin/env python3
"""Fixture: null/None dereference before check.

GROUND TRUTH:
- defect_class: null_dereference
- location: line `if user.name == name:`
- expected_finding: user may be None; the None check happens after the dereference.
  Correct order: check `if user is not None and user.name == name` (or check first)
- the rest is known-clean
"""
def find_user(users, name):
    for user in users:
        if user.name == name:
            return user
    if user is None:
        return None
