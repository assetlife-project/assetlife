<!-- omit in toc -->

# Contributing to AssetlLife

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
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

<!-- omit in toc -->

## Table of Contents

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
> [Documentation](https://docs.assetlife.org).

Before you ask a question, it is best to search for existing [Issues][AL-ISSUES] that might help
you. In case you have found a suitable issue and still need clarification, you
can write your question in this issue. It is also advisable to search the
internet for answers first.

If you then still feel the need to ask a question and need clarification, we
recommend the following:

- Open an [Issue](https://github.com/assetlife-project/assetlife/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (Python, uv, ruff, lefthook, tox,
  pyright, basedpyright, etc), depending on what seems relevant.

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
>
> When contributing to this project, you must agree that you have authored 100%
> of the content, that you have the necessary rights to the content and that
> the content you contribute may be provided under the project licence.

### Reporting Bugs

<!-- omit in toc -->

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more
information. Therefore, we ask you to investigate carefully, collect
information and describe the issue in detail in your report. Please complete
the following steps in advance to help us fix any potential bug as fast as
possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g.
  using incompatible environment components/versions (Make sure that you have
  read the [documentation][AL-DOCS]. If you are looking for
  support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the
  same issue you are having, check if there is not already a bug report
  existing for your bug or error in the [bug
  tracker](https://github.com/assetlife-project/assetlife/issues?q=label%3Abug).
- Also make sure to search the internet (including Stack Overflow) to see if
  users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package
    manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with
    older versions?

<!-- omit in toc -->

#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs
> including sensitive information to the issue tracker, or elsewhere in public.
> Instead sensitive bugs must be sent by email to
> <william.grison@rte-france.com>.

We use GitHub issues to track bugs and errors. If you run into an issue with
the project:

- Open an [Issue](https://github.com/assetlife-project/assetlife/issues/new).
  (Since we can't be sure at this point whether it is a bug or not, we ask you
  not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction
  steps* that someone else can follow to recreate the issue on their own. This
  usually includes your code. For good bug reports you should isolate the
  problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If
  there are no reproduction steps or no obvious way to reproduce the issue, the
  team will ask you for those steps and mark the issue as `needs-repro`. Bugs
  with the `needs-repro` tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`, as
  well as possibly other tags (such as `critical`), and the issue will be left
  to be [implemented by someone](#your-first-code-contribution).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for
AssetlLife, **including completely new features and minor improvements to
existing functionality**. Following these guidelines will help maintainers and
the community to understand your suggestion and find related suggestions.

<!-- omit in toc -->

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation][AL-DOCS] carefully and find out
  if the functionality is already covered, maybe by an individual
  configuration.
- Perform a [search][AL-ISSUES] to
  see if the enhancement has already been suggested. If it has, add a comment
  to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's
  up to you to make a strong case to convince the project's developers of the
  merits of this feature. Keep in mind that we want features that will be
  useful to the majority of our users and not just a small subset. If you're
  just targeting a minority of users, consider writing an add-on/plugin
  library.

<!-- omit in toc -->

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues][AL-ISSUES].

- Use a **clear and descriptive title** for the issue to identify the
  suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as
  many details as possible.
- **Describe the current behavior** and **explain which behavior you expected
  to see instead** and why. At this point you can also tell which alternatives
  do not work for you.
- **Explain why this enhancement would be useful** to most AssetlLife users.
  You may also want to point out the other projects that solved it better and
  which could serve as inspiration.

### Your First Code Contribution

#### Git local setup

Fork the AssetLife repository on GitHub, clone your fork and configure the upstream repository:

```bash
$ git clone https://github.com/YourLogin/assetlife.git
$ cd assetlife
$ git remote add upstream https://github.com/assetlife-project/assetlife.git
```

#### uv

Ensure you have [`uv`][GH-UV] installed. Now you can install the dev dependencies:

```bash
$ uv sync
```

This will install all the dependencies needed to run the linters, formatters,
type-checkers and unit tests in a Python virtual environment called `.venv`.
By default, they are installed in a virtual. **Don't forget to activate the virtual environment**.

#### Configure your IDE

Configure your IDE to use:

- Ruff formatter on save
- Ruff linter for diagnostics
- Basedpyright language server for type checking

#### Lefthook

[Lefthook][GH-LEFTHOOK] is a modern Git hooks manager,
which automatically lints and formats your code before committing it, which helps
avoid CI failures.

Install lefthook as a `uv` tool ([docs](https://docs.astral.sh/uv/concepts/tools/)),
run:

```bash
$ uv tool install lefthook --upgrade
```

Now the git hooks can be installed by running:

```bash
$ uvx lefthook install
```

To see if everything is set up correctly, you can run the validation command:

```bash
$ uvx lefthook validate
All good
```

#### Tox

The linters, formatters, type-checkers, and unit tests can easily be run with
[`tox`][GH-TOX]. Again, it is usefull to avoid CI
failures.

Install tox as a `uv` tool ([docs](https://docs.astral.sh/uv/concepts/tools/)),
run:

```bash
$ uv tool install tox --upgrade
```

To run all the unit tests in parallel on all supported Python versions, run :

```bash
$ uvx tox p -m test
```

You can also use `uvx tox p` because unit tests are part of the default tox
environments.

To run all other checks, run :

```bash
$ uvx tox -m check
```

If formatting errors are raised, fix them by running `dprint fmt`.

#### Development workflow

Synchronize your main branch:

```bash
git checkout main
git pull upstream main
```

Create a feature branch:

```bash
git checkout -b my_feature
```

Make changes and commit them.

AssetLife recommends using [Gitmoji](https://gitmoji.dev/) for commit messages
and PR titles. For VSCode and VSCodium users, it can be convenient to use the
[`gitmoji-vscode`](https://github.com/seatonjiang/gitmoji-vscode) extension for
this.

```bash
git push -u origin my_feature
```

Run the tests and checks with `tox`.

```bash
$ uvx tox p -m test
$ uvx tox p -m check
```

You can also use `uvx tox p` because unit tests are part of the default tox
environments. If formatting errors are raised, fix them by running `dprint fmt`.

review the pull request checklist, and open a P

#### Keep your branch up to date

```bash
git checkout main
git pull upstream main
git checkout my_feature
git merge main
```

In case of merge conflicts, don't be discouraged. If you can't fix them on your
own, ask the help of one maintainer.

#### Open a PR

In order to ease the reviewing process, we recommend that your contribution
complies with the following rules before marking a PR as “ready for review” :

- Use a clear descriptive title.
- Avoid unrelated commits, formatting changes, or personal comments.
- Address the complete issue.
- Run relevant `tox` command (see above).
- Add or update comments and documentation.
- Ensure documentation builds correctly.
- Mark incomplete work as a draft PR.

### Improving The Documentation

The documentation is written in reStructuredText and built with [Sphinx][GH-SPHINX].

If you're not familiar with reStructuredText, refer to the
[docs](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)

The documentation requires [Pandoc](https://pandoc.org/index.html) to be installed on your system.

To contribute to the documentation, follow the same process described in [Your
first code contribution](#your-first-code-contribution)

To build the documentation, run :

```bash
$ uvx tox -m docs
```

Before modifying the documentation :

- Read the [NumPy documentation style
  guide](https://numpydoc.readthedocs.io/en/latest/format.html) if you intend
  to modify docstrings of the codebase.
- API documentation uses
  [`autosummary`](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)
  extension with [`jinja`](https://jinja.palletsprojects.com/en/stable/)
  templating engine. Templates are located at `docs/source/_templates`.
- Pay particular attention to documenting class attributes. Sphinx does not
  always handle instance attributes reliably, especially inherited attributes.
  Attributes should therefore be referenced manually in the class documentation
  under the `Attributes` section of the docstring.
- As described in the [NumPy documentation style
  guide](https://numpydoc.readthedocs.io/en/latest/format.html), properties are
  listed in the `Attributes` section. Their associated docstrings will be
  included automatically.
- Some IDEs, such as PyCharm, may report false-positive warnings about
  variables that appear to be unused or unreferenced. These warnings can be
  ignored or disabled for the relevant statement.

## Join The Project Team

Please send a message to <william.grison@rte-france.com>.

## Attribution

This guide is based on the **contributing-gen**. [Make your
own](https://github.com/bttger/contributing-gen)!

[AL-ISSUES]: https://github.com/assetlife-project/assetlife/issues
[AL-README]: https://github.com/assetlife-project/assetlife/blob/main/README.md#assetlife
[AL-DOCS]: https://docs.assetlife.org
[AL-COC]: https://github.com/jorenham/optype/blob/master/CODE_OF_CONDUCT.md
[GH-TOX]: https://github.com/tox-dev/tox
[GH-UV]: https://github.com/astral-sh/uv
[GH-LEFTHOOK]: https://github.com/evilmartians/lefthook
[GH-SPHINX]: https://github.com/sphinx-doc/sphinx
