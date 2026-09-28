#!/usr/bin/env bash
#
# Build the multi-version documentation site under ./docs/build/html/.
#
# Usage, from anywhere inside the repository:
#   uvx tox -e docs-versions                        # or: bash docs/source/docs_versioning_build.sh
#   python -m http.server 8000 -d docs/build/html   # local server for the preview
#
# The base URL is the production site in CI and the local server otherwise.
#
# One folder per major.minor family that has a release tag, built from the
# highest patch of that family, plus "latest" built from main. conf.py always
# comes from your working directory, so every version renders with the current
# build logic (uncommitted edits to conf.py included).
#
# Each version is checked out in a temporary git worktree: your working
# directory, including uncommitted and untracked files, is never modified.

# Stop on errors, undefined variables and failures inside pipelines
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# GitHub Actions sets CI=true; a local run serves the site from a local server
# Must end with a trailing slash
if [ -n "${CI:-}" ]; then
    BASE_URL="https://docs.assetlife.org/"
else
    BASE_URL="http://localhost:8000/"
fi

# Work from the repository root, whatever directory the script is run from
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Absolute paths: the builds run from the temporary worktrees
SITE_DIR="$REPO_ROOT/docs/build/html"          # final site, one folder per version
VENV_DIR="$REPO_ROOT/docs/build/venvs"         # one environment per version, removed at the end
DOCTREES_DIR="$REPO_ROOT/docs/build/doctrees"  # Sphinx cache, removed at the end
CONF="$REPO_ROOT/docs/source/conf.py"          # conf.py used for every build

# ---------------------------------------------------------------------------
# Temporary worktrees and cleanup
# ---------------------------------------------------------------------------

# Every version is checked out in a subfolder of this directory
WORK_DIR="$(mktemp -d)"

# Delete the temporary checkouts when the script ends, whether it succeeds or
# fails; this never touches your working directory
cleanup() {
    rm -rf "$WORK_DIR"
    git worktree prune   # forget the worktrees that no longer exist
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

# Print release tags (vX.Y.Z), pre-releases excluded
release_tags() {
    # `|| true`: having no release tag is valid, grep must not abort the
    # script through pipefail
    git tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true
}

# Print the highest patch of every major.minor family, newest first,
# as "<tag> <folder>" lines
select_versions() {
    local family

    # Strip the patch number, then sort families by version, newest first
    release_tags | sed -E 's/\.[0-9]+$//' | sort -V -u | tac \
        | while read -r family; do
            echo "$(release_tags | grep -F "$family." | sort -V | tail -1) $family"
        done
}

# Build one version of the documentation in a temporary worktree
#   $1 - version name ("latest" or "v0.1"), passed to conf.py as DOCS_VERSION
#        for the version switcher, and used as the output folder
#   $2 - git ref to build, fully qualified to avoid any branch/tag ambiguity
#        (refs/heads/main, refs/remotes/origin/main, refs/tags/v0.1.3)
build_version() {
    local version="$1"
    local ref="$2"
    local src="$WORK_DIR/$version"
    local status=0

    # --detach: a build needs no branch, and git refuses to check out a branch
    # (main) that is already checked out in your working directory
    git worktree add --detach --quiet "$src" "$ref^{commit}" || return 1

    # Same conf.py for every build
    cp "$CONF" "$src/docs/source/conf.py" || status=$?

    # Subshell: the `cd` only applies to this build
    if [ "$status" -eq 0 ]; then
        (
            cd "$src"

            # tox activates its own environment (.tox/docs-versions) by setting
            # VIRTUAL_ENV; uv would ignore it anyway, unsetting it avoids the warning
            unset VIRTUAL_ENV

            # A dedicated environment per version (UV_PROJECT_ENVIRONMENT is read by uv)
            UV_PROJECT_ENVIRONMENT="$VENV_DIR/$version" \
            DOCS_VERSION="$version" \
                uv run --quiet --group docs \
                sphinx-build -b html -d "$DOCTREES_DIR/$version" -E docs/source "$SITE_DIR/$version" --quiet --fail-on-warning
        ) || status=$?
    fi

    # Remove this checkout right away, only one exists at a time
    git worktree remove --force "$src" || true

    return "$status"
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

# Start from an empty site
rm -rf "$SITE_DIR"
mkdir -p "$SITE_DIR" "$VENV_DIR"

# "latest" is built from the main branch: the local branch when it exists,
# origin/main otherwise (in CI, actions/checkout only creates remote branches)
if git show-ref --verify --quiet refs/heads/main; then
    LATEST_REF=refs/heads/main
else
    LATEST_REF=refs/remotes/origin/main
fi
 
# A failure here stops the script
echo "=== Building latest from ${LATEST_REF#refs/} ==="
build_version latest "$LATEST_REF"

# Entries for the theme's version dropdown
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html
ENTRIES=$(printf '{"name": "latest (dev)", "version": "latest", "url": "%slatest/"}' "$BASE_URL")

while read -r TAG FOLDER; do
    echo "=== Building $TAG into $FOLDER ==="

    if build_version "$FOLDER" "refs/tags/$TAG"; then
        ENTRIES+=$(printf ',\n  {"name": "%s", "version": "%s", "url": "%s%s/"}' \
            "$TAG" "$FOLDER" "$BASE_URL" "$FOLDER")
    else
        # A tag that fails to build must not fail the whole site
        echo "WARNING: skipped $TAG, documentation could not be built" >&2
    fi
done < <(select_versions)

# ---------------------------------------------------------------------------
# Site files
# ---------------------------------------------------------------------------

# versions.json feeds the dropdown menu
printf '[\n  %s\n]\n' "$ENTRIES" > "$SITE_DIR/versions.json"

# Without this, GitHub Pages' Jekyll ignores folders starting with "_"
touch "$SITE_DIR/.nojekyll"

# Root index redirects to "latest" by default
printf '<meta http-equiv="refresh" content="0; url=./latest/">\n' > "$SITE_DIR/index.html"

# Only the generated HTML is needed to serve or deploy the site;
# -E rebuilds the cache anyway
rm -rf "$VENV_DIR" "$DOCTREES_DIR"
