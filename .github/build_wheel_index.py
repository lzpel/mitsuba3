"""
Generate a static PEP 503 "simple" package index from the wheels attached to
this repository's GitHub Releases (tags starting with ``wheels-``).

The wheels themselves stay on GitHub Releases; the index only links to them,
so the GitHub Pages site stays small. Usage::

    python build_wheel_index.py --repo owner/name --out site

Requires the ``GH_TOKEN`` (or ``GITHUB_TOKEN``) environment variable to avoid
API rate limits.
"""

import argparse
import html
import json
import os
import re
import urllib.request
from pathlib import Path


def fetch_releases(repo):
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    releases, page = [], 1
    while True:
        req = urllib.request.Request(
            f'https://api.github.com/repos/{repo}/releases?per_page=100&page={page}',
            headers={'Accept': 'application/vnd.github+json'})
        if token:
            req.add_header('Authorization', f'Bearer {token}')
        with urllib.request.urlopen(req) as r:
            batch = json.load(r)
        if not batch:
            return releases
        releases += batch
        page += 1


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def collect_wheels(releases):
    """Return {project: [(filename, url, sha256 or None, release tag)]}"""
    projects = {}
    for rel in releases:
        if rel.get('draft') or not rel['tag_name'].startswith('wheels-'):
            continue
        for asset in rel['assets']:
            fname = asset['name']
            if not fname.endswith('.whl'):
                continue
            digest = asset.get('digest') or ''
            sha = digest[len('sha256:'):] if digest.startswith('sha256:') else None
            projects.setdefault(normalize(fname.split('-')[0]), []).append(
                (fname, asset['browser_download_url'], sha, rel['tag_name']))
    for files in projects.values():
        files.sort(key=lambda f: f[0])
    return projects


def page(title, body):
    return ('<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
            '<meta name="pypi:repository-version" content="1.0">\n'
            f'<title>{html.escape(title)}</title>\n</head>\n<body>\n'
            f'{body}</body>\n</html>\n')


def write_index(projects, repo, out):
    simple = out / 'simple'
    simple.mkdir(parents=True, exist_ok=True)
    (out / '.nojekyll').write_text('')

    links = ''.join(f'<a href="{p}/">{p}</a><br>\n' for p in sorted(projects))
    (simple / 'index.html').write_text(page('Simple index', links), encoding='utf-8')

    for proj, files in projects.items():
        (simple / proj).mkdir(exist_ok=True)
        body = f'<h1>Links for {html.escape(proj)}</h1>\n'
        for fname, url, sha, _ in files:
            href = html.escape(url + (f'#sha256={sha}' if sha else ''))
            body += f'<a href="{href}">{html.escape(fname)}</a><br>\n'
        (simple / proj / 'index.html').write_text(page(f'Links for {proj}', body),
                                                  encoding='utf-8')

    owner, name = repo.split('/')
    index_url = f'https://{owner}.github.io/{name}/simple/'
    rows = ''.join(
        f'<tr><td><a href="{html.escape(url)}">{html.escape(fname)}</a></td>'
        f'<td>{html.escape(tag)}</td></tr>\n'
        for files in projects.values() for fname, url, _, tag in files)
    body = f'''<h1>Prebuilt wheels for {html.escape(repo)}</h1>
<p>A <a href="simple/">PEP 503 package index</a> for wheels of this fork,
hosted on <a href="https://github.com/{html.escape(repo)}/releases">GitHub Releases</a>.
Dependencies such as <code>drjit</code> are still installed from PyPI.</p>

<h2>uv</h2>
<pre>uv add mitsuba --index mitsuba-fork={index_url}</pre>
<p>or add this to <code>pyproject.toml</code>:</p>
<pre>[[tool.uv.index]]
name = "mitsuba-fork"
url = "{index_url}"
explicit = true

[tool.uv.sources]
mitsuba = {{ index = "mitsuba-fork" }}</pre>
<p>One-off: <code>uv pip install mitsuba --index {index_url}</code></p>

<h2>pip</h2>
<pre>pip install mitsuba --extra-index-url {index_url}</pre>
<p>Fork versions carry a local label (e.g. <code>3.9.1+lzpel.1</code>) and sort
above the PyPI release with the same version number.</p>

<h2>Files</h2>
<table>
<tr><th>Wheel</th><th>Release</th></tr>
{rows}</table>
'''
    (out / 'index.html').write_text(page(f'{repo} wheels', body), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, help='owner/name')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    projects = collect_wheels(fetch_releases(args.repo))
    write_index(projects, args.repo, args.out)
    for proj, files in projects.items():
        print(f'{proj}: {len(files)} wheel(s)')


if __name__ == '__main__':
    main()
