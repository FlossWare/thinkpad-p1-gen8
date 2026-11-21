#!/usr/bin/env python3
"""Fixture: inverted boolean condition.

GROUND TRUTH:
- defect_class: inverted_condition
- location: `if not user.is_active and user.email_verified:`
- expected_finding: the condition is inverted — the intent (per the docstring
  and return values) is to grant access to active AND verified users; `not`
  makes it grant access to INACTIVE verified users. Should be
  `if user.is_active and user.email_verified:`
- the rest is known-clean
"""
def can_access(user):
    """Return True when the user is active and email-verified."""
    if not user.is_active and user.email_verified:
        return True
    return False
