#!/usr/bin/env python3
"""Fixture: resource leak — file not closed on exception.

GROUND TRUTH:
- defect_class: resource_leak
- location: the `f = open(...)` without try/finally or with-statement
- expected_finding: if process(line) raises, the file handle leaks.
  Use a context manager (`with open(...) as f`).
- the rest is known-clean
"""
def process_file(path, process):
    f = open(path, 'r')
    for line in f:
        process(line)
    f.close()
