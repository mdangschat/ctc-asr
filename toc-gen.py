"""Generate the table of contents and insert it at the top of `README.md`.

This script always assumes that the first heading is the document title
and does NOT include it in the table of contents.
It is assumed that only the first heading is H1 (#) and that all
subsequent headings are at least H2 (##).

Add the following to your `README.md` file (in the same folder):
[...]
## Contents
<!-- TOC_START -->

<!-- TOC_END -->
[...]
"""

import re


_HEADER_REGEX = r'([#]+) ([^\n]+)'
_PUNCTUATION_REGEX = r'[^\w\- ]'
_HEADER_TEMPLATE = '{indent}* [{name}](#{anchor})'
_START_TOC = '<!-- TOC_START -->'
_END_TOC = '<!-- TOC_END -->'


def __anchor(name):
    anchor = name.lower().replace(' ', '-')
    anchor = re.sub(_PUNCTUATION_REGEX, '', anchor)
    return anchor


def __parse_header(header):
    r = re.match(_HEADER_REGEX, header)
    if r:
        level = len(r.group(1))
        name = r.group(2)
        return level, __anchor(name), name


def __iter_headers(md):
    headers = (line for line in md.splitlines()
               if line.startswith('#'))
    for header in headers:
        yield header


def __get_header_item(header):
    level, anchor, name = __parse_header(header)
    # Levels are 1 for H1, 2 for H2, etc. Assuming all listed headings are
    # at least H2, then it should have zero indention.
    indent = '  ' * max(0, level - 2)
    return _HEADER_TEMPLATE.format(**locals())


def __gen_items(md):
    for header in __iter_headers(md):
        item = __get_header_item(header)
        yield item


def __read_md(filename):
    with open(filename, 'r') as f:
        return f.read()


def gen_toc(filename):
    md = __read_md(filename)
    i = md.index(_START_TOC) + len(_START_TOC) + 2
    j = md.index(_END_TOC)
    with open(filename, 'w') as f:
        f.write(md[:i])
        for i, item in enumerate(__gen_items(md)):
            if i == 0:
                continue

            f.write(item + '\n')
        f.write('\n' + md[j:])


if __name__ == '__main__':
    gen_toc('README.md')
