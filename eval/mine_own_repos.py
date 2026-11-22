#!/usr/bin/env python3
"""Mine real review fixtures from local git checkouts (own repos).

Ground-truth logic (git-log only, no API, no network):

- A commit A whose files were later touched by a "fix" commit B within
  FOLLOWUP_DAYS is a *defect fixture*: the change A is the code a reviewer
  saw; B's message describes what was actually wrong. match_terms are
  extracted from B's message + changed-file names.
- A commit A older than CLEAN_AGE_DAYS with no qualifying follow-up is a
  *clean fixture*: known-stable code. Success on a clean fixture is the
  INVERSE of the defect fixtures -- the model should NOT report bugs
  (false-positive measurement, which the synthetic fixture set cannot do).

Honesty constraints:
- Only B whose message matches FIX_RE counts as a follow-up (fix/bug/broke/
  incorrect/regression...). Feature follow-ups are not review misses.
- Merge commits, docs-only and test-only changes are skipped (A must change
  source files; B's classification uses its message).
- Every fixture records provenance: repo, commit A, commit B (if defect).

Output: fixtures under ~/.FlossWare/sources/web-scrape/github-reviews/fixtures/
(following the store's scrape structure) + manifest fragment JSON on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

FIX_RE = re.compile(
    r'\b(bug|broke|broken|incorrect|wrong|regression|crash|null|npe|'
    r'off[- ]by[- ]one|race|deadlock|leak|overflow|revert|typo)\b', re.I)
DOC_RE = re.compile(r'\.(md|rst|txt|json|ya?ml|toml|cfg|conf|ini|properties)$')
SRC_RE = re.compile(r'\.(py|java|rb|go|rs|c|h|cpp|js|ts|sh|groovy|kt|scala)$')

STOPWORDS = set('''a an and are as at be been but by code commit config data did
do does doesn fix fixes fixed for from had has have how if in into is it its
make may more must no not of on one only or our out pre re refactor refactoring
repo set should so some test tests that the their them then there these they
this to too up use used using via was we were what when where which who will
with without work would you your'''.split())


def git(repo, *args):
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:200])
    return r.stdout


def commits(repo, since):
    """Yield (sha, ts, subject) newest-first."""
    out = git(repo, 'log', '--no-merges', f'--since={since}',
              '--format=%H%x09%ct%x09%s')
    for line in out.splitlines():
        sha, ts, subj = line.split('\t', 2)
        yield sha, int(ts), subj


def files_of(repo, sha):
    return [f for f in git(repo, 'show', '--name-only', '--format=',
                           sha).splitlines() if f.strip()]


def diff_of(repo, sha):
    return git(repo, 'show', '--format=', sha)


def extract_terms(text, files):
    """Match terms from a follow-up message + file stems."""
    words = re.findall(r'[a-z_][a-z0-9_]{2,}', text.lower())
    terms = [w for w in dict.fromkeys(words)
             if w not in STOPWORDS and len(w) > 3][:8]
    for f in sorted(files)[:3]:
        stem = re.sub(r'\.[^.]+$', '', os.path.basename(f)).lower()
        if len(stem) > 3 and stem not in STOPWORDS and stem not in terms:
            terms.append(stem)
    return terms[:8]


def changed_lines(repo, sha):
    """Total added+removed lines of a commit (for the small-fix heuristic)."""
    out = git(repo, 'show', '--shortstat', '--format=', sha)
    m = re.search(r'(\d+) insertion.*?(\d+) deletion', out)
    if not m:
        return 99999
    return int(m.group(1)) + int(m.group(2))


def mine(repo, since='24 months ago', max_fixtures=40):
    """Return fixture dicts for one repo."""
    cs = list(commits(repo, since))
    info = {sha: (ts, subj, set(files_of(repo, sha))) for sha, ts, subj in cs}
    order = {sha: i for i, (sha, _, _) in enumerate(cs)}  # 0 = newest
    used_fixes = set()  # one fix commit labels at most ONE fixture

    fixtures = []
    for sha, ts, subj in cs:
        _, _, afiles = info[sha]
        src = [f for f in afiles if SRC_RE.search(f)]
        if not src or DOC_RE.match(subj.split()[-1] if subj else ''):
            continue
        age_days = (cs[0][1] - ts) / 86400

        # defect fixture: a SMALL defect-named commit B after A, sharing >=2
        # source files (or half of B's files) with A, not already used
        fix = None
        for bsha, bts, bsubj in cs:
            if order[bsha] >= order[sha]:
                continue  # older-or-equal in list == not after A
            if bts - ts > 14 * 86400:
                break  # cs is newest-first; everything older is too far
            if bsha in used_fixes or not FIX_RE.search(bsubj):
                continue
            shared = info[bsha][2] & set(src)
            if len(shared) < 2 and len(shared) < len(info[bsha][2]) / 2:
                continue
            if changed_lines(repo, bsha) > 300:
                continue
            fix = (bsha, bsubj)
            used_fixes.add(bsha)
            break
        if fix:
            bsha, bsubj = fix
            fixtures.append({
                'repo': os.path.basename(repo.rstrip('/')),
                'commit': sha, 'fix_commit': bsha,
                'class': 'defect',
                'subject': subj[:100],
                'fix_subject': bsubj[:100],
                'match_terms': extract_terms(bsubj, info[bsha][2]),
                'ts': ts})
        elif age_days > 30:
            fixtures.append({
                'repo': os.path.basename(repo.rstrip('/')),
                'commit': sha, 'fix_commit': None,
                'class': 'clean',
                'subject': subj[:100],
                'fix_subject': None,
                'match_terms': [],
                'ts': ts})
        if len(fixtures) >= max_fixtures:
            break
    return fixtures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repos', nargs='+', help='local checkout paths')
    ap.add_argument('--out-dir',
                    default=os.path.expanduser(
                        '~/.FlossWare/sources/web-scrape/github-reviews/'
                        'fixtures'))
    ap.add_argument('--max-per-repo', type=int, default=40)
    ap.add_argument('--since', default='24 months ago')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_fx, seen = [], set()
    for repo in args.repos:
        try:
            fx = mine(repo, args.since, args.max_per_repo)
        except Exception as e:  # noqa: BLE001
            print(f'# SKIP {repo}: {e}', file=sys.stderr)
            continue
        kept = 0
        for f in fx:
            key = (f['repo'], f['commit'])
            if key in seen:
                continue
            slug = f"{f['repo']}-{f['commit'][:10]}"
            path = os.path.join(args.out_dir, slug + '.diff')
            try:
                d = diff_of(repo, f['commit'])
            except Exception:  # noqa: BLE001
                continue
            if not d.strip() or len(d) > 60000:
                continue
            with open(path, 'w') as fh:
                fh.write(d)
            f['file'] = os.path.basename(path)
            f['diff_path'] = path
            all_fx.append(f)
            seen.add(key)
            kept += 1
        n_def = sum(1 for f in fx if f['class'] == 'defect')
        print(f'# {repo}: {len(fx)} candidates ({n_def} defect), {kept} kept',
              file=sys.stderr)

    manifest = {
        'prompt_template': (
            'Review this code change for bugs. The change is a git diff; '
            'the code it modifies is real, previously-shipped code. Report '
            'any defects you find, or state clearly if the change looks '
            'clean.\n\n```\n{code}\n```'),
        'fixtures': [{k: v for k, v in f.items() if k != 'diff_path'}
                     for f in all_fx],
    }
    json.dump(manifest, sys.stdout, indent=1)
    print(f'# manifest: {len(all_fx)} fixtures '
          f'({sum(1 for f in all_fx if f["class"] == "defect")} defect, '
          f'{sum(1 for f in all_fx if f["class"] == "clean")} clean)',
          file=sys.stderr)


if __name__ == '__main__':
    main()
