#!/usr/bin/env python3
"""Mine real review fixtures from public GitHub repos via the gh CLI.

Ground-truth logic: for a MERGED pull request in a high-review-culture repo,
the human review comments ARE the labeled findings a good reviewer should
catch. A comment counts as a real finding only if it is defect-shaped (see
COMMENT_RE) AND a commit later in the PR's own series touched the file the
comment is anchored to -- i.e. the finding was addressed, not just discussed.

Fixture = the PR diff (the code state the humans reviewed).
match_terms = terms extracted from the qualifying comment bodies.
class = defect whenever >=1 qualifying comment exists; PRs whose comments
are all non-qualifying are skipped (mixed evidence), not counted as clean
(public PRs are never known-clean).

Output: fixtures under ~/.FlossWare/sources/web-scrape/github-reviews/
(same store convention as mine_own_repos.py) + manifest JSON on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

# comments that describe a defect worth catching (not praise/questions/style)
COMMENT_RE = re.compile(
    r'\b(bug|broke|broken|incorrect|wrong|error|fail\w*|crash\w*|race|null|'
    r'npe|leak\w*|overflow|underflow|regression|insecure|vulnerab\w*|'
    r'off[- ]by[- ]one|deadlock|hang\w*|except\w*|edge case|missing check|'
    r"should (check|handle|validate|return|throw)|doesn't handle)\b", re.I)
NIT_RE = re.compile(r'^\s*(nit|typo|style|question:|fyi|out of curiosity)', re.I)
STOPWORDS = set('''a an and are as at be been but by code commit config data did
do does doesn for from had has have how if in into is it its make may more must
no not of on one only or our out pre re repo set should so some test tests that
the their them then there these they this to too up use used using via was we
were what when where which who will with without work would you your about
actually looks like just sure maybe think also than there's it's'''.split())


def gh(*args, accept=None):
    """Run a gh api call; return parsed JSON (or raw text for diff)."""
    cmd = ['gh', 'api']
    if accept:
        cmd += ['-H', f'Accept: {accept}']
    cmd += list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:200])
    if accept:
        return r.stdout
    return json.loads(r.stdout)


def paged(path, max_pages=10):
    """All pages of a list endpoint (per_page=100)."""
    out = []
    for page in range(1, max_pages + 1):
        sep = '&' if '?' in path else '?'
        batch = gh(f'{path}{sep}per_page=100&page={page}')
        if not batch:
            break
        out += batch
        if len(batch) < 100:
            break
    return out


def prs_of(repo, max_prs):
    """Merged PRs, newest first, one page at a time."""
    out = []
    page = 1
    while len(out) < max_prs and page <= 40:
        batch = gh(f'repos/{repo}/pulls?state=closed&sort=updated'
                   f'&direction=desc&per_page=50&page={page}')
        if not batch:
            break
        for p in batch:
            if p.get('merged_at'):
                out.append({'number': p['number'],
                            'title': p.get('title', ''),
                            'merged_at': p['merged_at']})
        page += 1
    return out[:max_prs]


def later_files(repo, commits):
    """Set of files touched by commits after the first one (capped)."""
    files = set()
    for c in commits[1:4]:  # cap the extra API calls at 3
        try:
            detail = gh(f"repos/{repo}/commits/{c['sha']}")
        except Exception:  # noqa: BLE001
            continue
        for f in detail.get('files') or []:
            if f.get('filename'):
                files.add(f['filename'])
    return files


def diff_of(repo, number):
    return gh(f'repos/{repo}/pulls/{number}',
              accept='application/vnd.github.diff')


def extract_terms(bodies):
    """Distinct content words across qualifying comment bodies."""
    terms = []
    for b in bodies:
        for w in re.findall(r'[a-z_][a-z0-9_]{2,}', b.lower()):
            if w not in STOPWORDS and w not in terms:
                terms.append(w)
    return terms[:10]


def mine_repo(repo, max_prs, out_dir, max_diff=60000):
    fixtures = []
    for pr in prs_of(repo, max_prs):
        n = pr['number']
        try:
            cs = paged(f'repos/{repo}/pulls/{n}/comments')
            if not cs:
                continue
            cms = paged(f'repos/{repo}/pulls/{n}/commits')
            if len(cms) < 2:
                continue
            files_after = later_files(repo, cms)
            good = [c for c in cs
                    if COMMENT_RE.search(c.get('body') or '')
                    and not NIT_RE.match(c.get('body') or '')
                    and c.get('path') in files_after]
            if not good:
                continue
            d = diff_of(repo, n)
            if not d.strip() or len(d) > max_diff or d.startswith('{'):
                continue
            bodies = [c['body'] for c in good]
            fixtures.append({
                'repo': repo, 'pr': n, 'class': 'defect',
                'title': pr['title'][:100],
                'n_comments': len(cs), 'n_qualifying': len(good),
                'n_commits': len(cms),
                'match_terms': extract_terms(bodies),
                'commenters': sorted({(c.get('user') or {}).get('login', '?')
                                      for c in good})[:5],
                'merged_at': pr['merged_at']})
            slug = f"{repo.replace('/', '-')}-{n}"
            path = os.path.join(out_dir, slug + '.diff')
            with open(path, 'w') as fh:
                fh.write(d)
            fixtures[-1]['file'] = os.path.basename(path)
            print(f'# {repo}#{n}: {len(good)} findings, '
                  f'{len(cms)} commits', file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f'# SKIP {repo}#{n}: {e}', file=sys.stderr)
            continue
        time.sleep(0.3)  # be polite to the API
    return fixtures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repos', nargs='+', help='owner/name (public repos)')
    ap.add_argument('--max-prs', type=int, default=30)
    ap.add_argument('--out-dir',
                    default=os.path.expanduser(
                        '~/.FlossWare/sources/web-scrape/github-reviews/'
                        'fixtures'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_fx = []
    for repo in args.repos:
        try:
            fx = mine_repo(repo, args.max_prs, args.out_dir)
        except Exception as e:  # noqa: BLE001
            print(f'# REPO FAIL {repo}: {e}', file=sys.stderr)
            continue
        print(f'# {repo}: {len(fx)} fixtures', file=sys.stderr)
        all_fx += fx

    manifest = {
        'prompt_template': (
            'Review this code change for bugs. The change is a git diff for '
            'a pull request; professional human reviewers reviewed this '
            'exact change. Report any defects you find.\n\n```\n{code}\n```'),
        'fixtures': all_fx,
    }
    json.dump(manifest, sys.stdout, indent=1)
    print(f'# manifest: {len(all_fx)} public-repo fixtures',
          file=sys.stderr)


if __name__ == '__main__':
    main()
