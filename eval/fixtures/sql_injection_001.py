#!/usr/bin/env python3
"""Fixture: SQL injection via string concatenation.

GROUND TRUTH:
- defect_class: sql_injection
- location: the `query = "SELECT ... " + user_id` line
- expected_finding: user input concatenated into SQL — injection vector.
  Must use parameterized queries.
- the rest is known-clean
"""
def get_user(cursor, user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    return cursor.fetchone()
