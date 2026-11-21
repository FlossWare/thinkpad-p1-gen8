#!/usr/bin/env python3
"""Fixture: lock-ordering deadlock between two threads.

GROUND TRUTH:
- defect_class: deadlock
- location: transfer() acquires lock_a then lock_b while receive() (not shown,
  but same pattern elsewhere) acquires lock_b then lock_a — the inconsistent
  lock ordering in this function versus its documented counterpart
- expected_finding: inconsistent lock acquisition order across the two
  functions creates a deadlock window; both must acquire in the same order
  (use a lock hierarchy or try/timeout)
- the rest is known-clean
"""
import threading

lock_a = threading.Lock()
lock_b = threading.Lock()

def transfer(amount):
    """Move funds; counterpart function acquires lock_b then lock_a."""
    with lock_a:
        with lock_b:
            pass  # move funds
