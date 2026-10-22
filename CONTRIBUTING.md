# Contributing to AssetLife <!-- omit in toc -->

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued.
See the [Table of Contents](#table-of-contents) for different ways to help and details
about how this project handles them.
Please make sure to read the relevant section before making your contribution. It will
make it a lot easier for us maintainers and smooth out the experience for all involved.
The community looks forward to your contributions. 🎉

> [!NOTE]
> And if you like AssetLife, but just don't have time to contribute, that's
> fine. There are other easy ways to support the project and show your
> appreciation, which we would also be very happy about:
>
> - Star the project
> - Share it on social media
> - Refer this project in your project's README
> - Mention the project at local meetups and tell your friends/colleagues

## Table of Contents <!-- omit in toc -->

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving The Documentation](#improving-the-documentation)
- [Join The Project Team](#join-the-project-team)

## Code of Conduct

This project and everyone participating in it is governed by the [AssetLife
Code of Conduct][AL-COC].
By participating, you are expected to uphold this code.
Please report unacceptable behavior to <william.grison@rte-france.com>.

## I Have a Question

> [!NOTE]
> If you want to ask a question, we assume that you have read the available
> [documentation][AL-DOCS].

Before you ask a question, it is best to search for existing [issues][AL-ISSUES] that
might help you. In case you have found a suitable issue and still need clarification,
you can write your question in this issue. It is also advisable to search the internet
for answers first.

If you then still feel the need to ask a question and need clarification, we recommend
the following:

- Open an [issue][AL-NEW-ISSUE].
- Provide as much context as you can about what you're running into.
- Provide the versions of AssetLife, Python and of the relevant dependencies (NumPy,
  SciPy, etc.). See [how to get them](#how-to-get-your-versions).

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
>
> When contributing to this project, you must agree that you have authored 100% of the
> content, that you have the necessary rights to the content and that the content you
> contribute may be provided under the project [license](LICENSE) (Apache 2.0).

### Reporting Bugs

#### Before Submitting a Bug Report <!-- omit in toc -->

A good bug report shouldn't leave others needing to chase you up for more information.
Therefore, we ask you to investigate carefully, collect information and describe the
issue in detail in your report. Please complete the following steps in advance to help
us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side, e.g. using
  incompatible environment components/versions. Make sure that you have read the
  [documentation][AL-DOCS]. If you are looking for support, you might want to check
  [this section](#i-have-a-question).
- To see if other users have experienced (and potentially already solved) the same
  issue you are having, check if there is not already a bug report existing for your
  bug or error in the [bug tracker][AL-BUGS].
- Also make sure to search the internet (including Stack Overflow) to see if users
  outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (traceback)
  - OS, platform and version (Windows, Linux, macOS, x86, ARM)
  - Versions of AssetLife, Python and of the relevant dependencies (see below)
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older
    versions?

#### How To Get Your Versions <!-- omit in toc -->

```bash
python --version
python -c "import assetlife; print(assetlife.__version__)"
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

#### How Do I Submit a Good Bug Report? <!-- omit in toc -->

> [!CAUTION]
> You must never report security related issues, vulnerabilities or bugs including
> sensitive information to the issue tracker, or elsewhere in public. Instead,
> sensitive bugs must be sent by email to <william.grison@rte-france.com>.

We use GitHub issues to track bugs and errors. If you run into an issue with the
project:

- Open an [issue][AL-NEW-ISSUE]. (Since we can't be sure at this point whether it is a
  bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps*
  that someone else can follow to recreate the issue on their own. This usually
  includes your code. For good bug reports you should isolate the problem and create a
  reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are
  no reproduction steps or no obvious way to reproduce the issue, the team will ask you
  for those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro` tag
  will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`, as well as
  possibly other tags (such as `critical`), and the issue will be left to be
  [implemented by someone](#your-first-code-contribution).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for AssetLife,
**including completely new features and minor improvements to existing
functionality**. Following these guidelines will help maintainers and the community to
understand your suggestion and find related suggestions.

#### Before Submitting an Enhancement <!-- omit in toc -->

- Make sure that you are using the latest version.
- Read the [documentation][AL-DOCS] carefully and find out if the functionality is
  already covered, maybe by an individual configuration.
- Perform a [search][AL-ISSUES] to see if the enhancement has already been suggested.
  If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to
  you to make a strong case to convince the project's developers of the merits of this
  feature. Keep in mind that we want features that will be useful to the majority of
  our users and not just a small subset. If you're just targeting a minority of users,
  consider writing an add-on/plugin library.

#### How Do I Submit a Good Enhancement Suggestion? <!-- omit in toc -->

Enhancement suggestions are tracked as [GitHub issues][AL-ISSUES].

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many
  details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see
  instead** and why. At this point you can also tell which alternatives do not work for
  you.
- **Explain why this enhancement would be useful** to most AssetLife users. You may
  also want to point out the other projects that solved it better and which could
  serve as inspiration.

### Your First Code Contribution

#### Prerequisites <!-- omit in toc -->

- [Git][GIT]
- [`uv`][GH-UV], which installs Python and every other development tool for you
- [Pandoc][PANDOC], only if you build the documentation

#### Git local setup <!-- omit in toc -->

Fork the AssetLife repository on GitHub, clone your fork and configure the upstream
repository:

```bash
git clone https://github.com/YourLogin/assetlife.git
cd assetlife
git remote add upstream https://github.com/assetlife-project/assetlife.git
```

#### Development environment <!-- omit in toc -->

Install AssetLife and all the development dependencies (linters, formatters,
type-checkers, unit tests and documentation tools):

```bash
uv sync
```

This creates a virtual environment in `.venv`, with the exact tool versions pinned in
`uv.lock`. Either activate it (`source .venv/bin/activate` on Linux and macOS,
`.venv\Scripts\activate` on Windows), or prefix the commands with `uv run`, e.g.
`uv run pytest`.

> [!IMPORTANT]
> The CI uses the tool versions pinned in `uv.lock`. Use the ones from `.venv`, not a
> globally installed `ruff` or `basedpyright`, otherwise you may get different results
> than the CI. Don't upgrade these tools in a feature PR.

#### Configure your IDE <!-- omit in toc -->

Configure your IDE to:

- use the `.venv` Python interpreter;
- run the Ruff linter and the BasedPyright language server for diagnostics;
- on save, format with Ruff, apply Ruff's fixes **and sort the imports**. Formatting
  alone does not sort imports: this is a lint rule (`I001`), checked by the CI.

For VS Code (and VSCodium), install the [Python][VSC-PYTHON], [Ruff][VSC-RUFF]
and [BasedPyright][VSC-BASEDPYRIGHT] extensions. Configurations are in
[.vscode](https://https://github.com/assetlife-project/assetlife/.vscode).Note
that using Basedpyright with Pylance is not recommended. Deactivate Pylance.

#### Lefthook <!-- omit in toc -->

[Lefthook][GH-LEFTHOOK] is a Git hooks manager. Install it as a `uv` tool
([docs](https://docs.astral.sh/uv/concepts/tools/)) and install the hooks:

```bash
uv tool install lefthook --upgrade
uvx lefthook install
```

To see if everything is set up correctly, run `uvx lefthook validate`. It should print
`All good`.

Before each commit, the hooks run on your staged files:

- `uv lock`, if `pyproject.toml` changed;
- `dprint fmt`, on Markdown, TOML, YAML and JSON files;
- `ruff check --fix` then `ruff format`, on Python files and notebooks;

Fixes are added to your commit automatically. If a check still fails, the
commit is aborted: fix the reported problems and commit again. After a checkout
or a merge that changes `uv.lock`, the hooks also run `uv sync` to keep `.venv`
up to date.

Tests are not run on commit to keep it fast: run `uv run pytest` yourself. CI
runs them on pull requests and on pushes.

#### Tox <!-- omit in toc -->

[Tox][GH-TOX] runs the same checks as the CI (Github) in isolated environments. Install it as a
`uv` tool:

```bash
uv tool install tox --upgrade
```

| Command                    | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| `uvx tox p -m test`        | Run the unit tests in parallel on all supported Python versions |
| `uvx tox -e 3.12`          | Run the unit tests on a single Python version                   |
| `uvx tox -m lint`          | Run the lint and format checks (`ruff` and `dprint`)            |
| `uvx tox -m type`          | Run the static type checks (`basedpyright`)                     |
| `uvx tox -m docs`          | Build the current documentation (`sphinx`)                      |
| `uvx tox -m docs-versions` | Build the documentation for all versions (`sphinx`)             |

If a check fails, most problems can be fixed automatically:

```bash
uv run ruff check --fix   # lint errors, including unsorted imports
uv run ruff format        # Python formatting
uv run dprint fmt         # Markdown, TOML, YAML and JSON formatting
```

#### Continuous integration <!-- omit in toc -->

Every pull request runs:

- **format**: `dprint check`, `ruff check` and `ruff format --check`;
- **typecheck**: `basedpyright --warnings` on Python 3.12 and 3.14;
- **test**: `pytest` on Linux and Windows, with Python 3.12 to 3.14.

All jobs must pass before a PR can be merged. Ruff and BasedPyright must report
**no error and no warning**. Avoid silencing a rule with `# noqa` or `#
pyright: ignore` unless there is no reasonable alternative, and then explain
why in a comment. Don't be discouraged if the CI fails. A maintainer can help
and fix the errors for you.

#### Development workflow <!-- omit in toc -->

Synchronize your main branch:

```bash
git checkout main
git pull upstream main
```

Create a feature branch:

```bash
git checkout -b my_feature
```

Make changes and commit them. AssetLife recommends using [Gitmoji](https://gitmoji.dev/)
for commit messages and PR titles. For VS Code and VSCodium users, the
[`gitmoji-vscode`](https://github.com/seatonjiang/gitmoji-vscode) extension can be
convenient for this.

Run the tests and checks:

```bash
uvx tox p -m test
uvx tox -m lint
uvx tox -m type
```

Push your branch and [open a PR](#open-a-pr):

```bash
git push -u origin my_feature
```

#### Keep your branch up to date <!-- omit in toc -->

```bash
git fetch upstream
git checkout my_feature
git merge upstream/main
```

In case of merge conflicts, don't be discouraged. If you can't fix them on your own,
ask a maintainer for help.

#### Open a PR <!-- omit in toc -->

To ease the reviewing process, check that your contribution complies with the following
rules before marking a PR as “ready for review”:

- Use a clear and descriptive title.
- Avoid unrelated commits and unrelated formatting changes.
- Address the complete issue, and reference it in the description (e.g. `Closes #42`).
- Run the relevant `tox` commands (see [Tox](#tox)) and make sure the CI passes.
- Add or update tests, comments and documentation.
- Ensure the documentation builds correctly.
- Mark incomplete work as a draft PR.

### Improving The Documentation

The documentation is written in reStructuredText and built with [Sphinx][GH-SPHINX].
If you're not familiar with reStructuredText, refer to the [Sphinx
documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).

To contribute to the documentation, follow the same process described in [Your First
Code Contribution](#your-first-code-contribution). Building it requires
[Pandoc][PANDOC] to be installed on your system.

To build the documentation of your working copy, run:

```bash
uvx tox -m docs
```

The result can be opened at `docs/build/html/index.html` (e.g. `python -m http.server -d docs/build/html` to run a local server).

Before modifying the documentation:

- Read the [NumPy documentation style guide][NUMPYDOC] if you intend to modify
  docstrings of the codebase.
- API documentation uses the
  [`autosummary`](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)
  extension with the [`jinja`](https://jinja.palletsprojects.com/en/stable/)
  templating engine. Templates are located in `docs/source/_templates`.
- Pay particular attention to documenting class attributes. Sphinx does not always
  handle instance attributes reliably, especially inherited attributes. Attributes
  should therefore be referenced manually in the class documentation under the
  `Attributes` section of the docstring.
- As described in the [NumPy documentation style guide][NUMPYDOC], properties are
  listed in the `Attributes` section. Their associated docstrings will be included
  automatically.
- Some IDEs, such as PyCharm, may report false-positive warnings about variables that
  appear to be unused or unreferenced. These warnings can be ignored or disabled for
  the relevant statement.

## Join The Project Team

Please send a message to <william.grison@rte-france.com>.

## Attribution

This guide is based on the **contributing-gen**. [Make your
own](https://github.com/bttger/contributing-gen)!

[AL-BUGS]: https://github.com/assetlife-project/assetlife/issues?q=label%3Abug
[AL-COC]: CODE_OF_CONDUCT.md
[AL-DOCS]: https://docs.assetlife.org
[AL-ISSUES]: https://github.com/assetlife-project/assetlife/issues
[AL-NEW-ISSUE]: https://github.com/assetlife-project/assetlife/issues/new
[GH-LEFTHOOK]: https://github.com/evilmartians/lefthook
[GH-SPHINX]: https://github.com/sphinx-doc/sphinx
[GH-TOX]: https://github.com/tox-dev/tox
[GH-UV]: https://github.com/astral-sh/uv
[GIT]: https://git-scm.com/
[NUMPYDOC]: https://numpydoc.readthedocs.io/en/latest/format.html
[PANDOC]: https://pandoc.org/installing.html
[VSC-PYTHON]: https://marketplace.visualstudio.com/items?itemName=ms-python.python
[VSC-BASEDPYRIGHT]: https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright
[VSC-RUFF]: https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff
