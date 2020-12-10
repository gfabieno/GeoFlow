# Style guide

## Project-specific style conventions

Hanging indents should be at the same level as the opening parenthesis, bracket or brace, as in
```python
parser.add_argument("--case",
                    type=str,
                    default="Case1Dsmall",
                    help="Name of the case from `Cases_define` to use")
```

Identifiers must be descriptive and short enough to maintain ease-of-use.

Module names should be written in `CamelCase`, as opposed to PEP 8 conventions.

Use docstring conventions of PEP 257. Docstrings should follow the reStructuredText format. Have the docstring title on the line following the opening quotes. Have the indentation at the same level as the opening and closing quotes. Always use the appropriate capitalization and punctuation in all parts of the docstrings. Here is a template for docstrings:
```python
"""
Describe the contents in a short, descriptive, one-line title.

A lengthier description, if needed. Always wrap lines at line lengths of 79
characters. Omit if the title is sufficient and keep 1 blank line between
items of the docstring.

:param variable: This is a description of the first variable. Wrap lines at the
                 same level as the start of the description.
:type variable: SomeType
:param another_variable: Describe variables in a sequential fashion.
:type another_variable: SomeType

:return: Describe the output. Omit if there is no output.
"""
```
If there is multiple outputs, use this alternative `:return:`:
```python
:return:
    first_output: Describe the first output.
    second_output: Describe the second output, and so on.
```

Always use type hinting, that is, specify the type of inputs by following arguments by a colon and the expected type, as in `argument: Type`. Use whitespace around `=` if you use default arguments, as in `argument: Type = default_value`.

Keep the quantity of blank lines to a minimum. For instance, do not use blank lines between a function's docstring and contents.

Use accents (\`\`) when referring to code in docstrings and commits. This is Markdown syntax.


## Lesser known PEP 8 and `git` conventions that are in use in this project

Use full sentences and proper capitalization and punctuation in docstrings, but also comments and commit messages. The only exception is commit titles not having periods.

Imports should be grouped in the following order:
1. Standard library imports.
2. Related third party imports.
3. Local application/library specific imports.
You should put a blank line between each group of imports.

If operators with different priorities are used, consider adding whitespace around the operators with the lowest priority(ies). Use your own judgment; however, never use more than one space, and always have the same amount of whitespace on both sides of a binary operator, as in
```python
i = i + 1
submitted += 1
x = x*2 - 1
hypot2 = x*x + y*y
c = (a+b) * (a-b)
```

Docstring titles and commit message titles should start with an infinitive verb, for instance
```python
"""
Generate a complete dataset.
"""
```
or
```Merge pull request```

Commit messages should follow [these guidelines](https://chris.beams.io/posts/git-commit/):
    1. Separate subject from body with a blank line
    2. Limit the subject line to 50 characters
    3. Capitalize the subject line
    4. Do not end the subject line with a period
    5. Use the imperative mood in the subject line
    6. Wrap the body at 72 characters
    7. Use the body to explain what and why vs. how
Most often, a descriptive title is sufficient. Use the following template if your commit needs more extensive description:
```
Summarize changes in around 50 characters or less

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of the commit and the rest of the text as the body. The
blank line separating the summary from the body is critical (unless
you omit the body entirely); various tools like `log`, `shortlog`
and `rebase` can get confused if you run the two together.

Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how (the code explains that).
Are there side effects or other unintuitive consequences of this
change? Here's the place to explain them.

Further paragraphs come after blank lines.

 - Bullet points are okay, too

 - Typically a hyphen or asterisk is used for the bullet, preceded
   by a single space, with blank lines in between, but conventions
   vary here

If you use an issue tracker, put references to them at the bottom,
like this:

Resolves: #123
See also: #456, #789
```

Commits should be small and unitary. This makes it easier to review changes and to detect where bugs were introduced, for instance. Leave minimal uncommitted work in your working directory at all times. This also makes it easier to track ongoing work on features. Don't mind having too many commits: commits will be squashed into a single commit after review, when a pull request is accepted.

Use significative branch names, that is, label branches by features instead of by your own name. Use hyphens between words, for instance `add-feature`.

Some IDEs (PyCharm is one of them) recommend wrapping lines at 80 characters instead of 79, although PEP 8 advocates having 79 characters per line. You can manually change this setting to comply with PEP 8.
