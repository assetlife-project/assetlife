Install
=======

AssetLife is a Python package. It is uploaded on the `Python Package Index (PyPi) <https://pypi.org/>`_. Before you install AssetLife, make
sure Python **3.11 (or newer)** is installed.

Install AssetLife with `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_ :

.. code-block::

    $ python -m pip install assetlife

**Optional but recommended :** create and activate a virtual environment before.

For Linux users :

.. code-block::

    $ /usr/bin/python3.** -m venv <venv_location>/assetlife
    $ source <venv_location>/assetlife/bin/activate

For Windows users :

.. code-block::

    $ py -3.** -m venv <venv_location>\assetlife
    $ .\<venv_location>\assetlife\Scripts\activate

**From source :** to install AssetLife from source, go to the `AssetLife repository <https://github.com/assetlife-project/assetlife>`_. Clone the codebase and install AssetLife with `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_.

.. code-block::

    $ git clone https://github.com/assetlife-project/assetlife.git
    $ cd assetlife
    $ python -m pip install .

For contributors, optional development dependencies (type checker, documentation builder, etc.) are available. Use this command instead :

.. code-block::

    $ python -m pip install -e ".[dev]"
