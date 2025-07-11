# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Union


class NewLineToken:
    def __init__(self, indent=0):
        self.indent = indent

    def __str__(self):
        return '\n' + ' ' * self.indent


class Doc:
    default_indent = 2

    def __init__(self):
        self.docs: List[Union[str, NewLineToken]] = []

    def empty(self):
        for doc in self.docs:
            if isinstance(doc, str) and doc.strip():
                return False
        return True

    def join(self, lst):
        return doc_join(lst, self)

    def append(self, doc):
        if isinstance(doc, list):
            for item in doc:
                self.append(item)
        elif isinstance(doc, Doc):
            self.docs.extend(doc.docs)
        elif isinstance(doc, str):
            self.docs.append(doc)
        else:
            raise NotImplementedError()

    def indent(self, inc=None):
        if inc is None:
            inc = self.default_indent
        doc = Doc()
        for token in self.docs:
            if isinstance(token, NewLineToken):
                doc.docs.append(NewLineToken(indent=token.indent + inc))
            else:
                doc.docs.append(token)
        return doc

    def trim(self):
        i = 0
        while i < len(self.docs) and isinstance(self.docs[i], NewLineToken):
            i += 1
        j = len(self.docs)
        while j > i and isinstance(self.docs[j - 1], NewLineToken):
            j -= 1
        doc = Doc()
        for k in range(i, j):
            token = self.docs[k]
            if isinstance(token, NewLineToken):
                doc.docs.append(NewLineToken(indent=token.indent))
            else:
                doc.docs.append(token)
        return doc

    def __add__(self, other):
        doc = Doc()
        doc.docs = [token for token in self.docs]
        doc += other
        return doc

    def __radd__(self, other):
        doc = Doc()
        doc.docs = []
        doc.append(other)
        doc.append(self)
        return doc

    def __iadd__(self, other):
        self.append(other)
        return self

    def __str__(self):
        return "".join(str(s) for s in self.docs)


class NewLine(Doc):
    def __init__(self, indent=0):
        super().__init__()
        self.docs.append(NewLineToken(indent))


class Text(Doc):
    def __init__(self, s: str):
        super().__init__()
        assert isinstance(s, str)
        self.docs.append(s)
        self.format_str = s

    def format(self, *args) -> Doc:
        format_str: str = self.format_str
        texts = format_str.split('{}')
        if len(texts) != len(args) + 1:
            raise ValueError(f'format string {format_str} does not match the number of args: {len(args)}')
        return doc_join([Text(texts[i]) + args[i] for i in range(len(args))], "") + Text(texts[-1])


def doc_join(seq: List, sep):
    doc = Doc()
    for i in range(len(seq)):
        if i != 0:
            doc += sep
        doc += seq[i]
    return doc


def doc_join_lines(
    seq: List[Union[str, Doc]], left: Union[Doc, str], right: Union[Doc, str], indent=None, line_end_sep=','
):
    doc = Doc()
    if indent is None:
        indent = 4
    if len(seq) == 0:
        doc += left + right
        return doc
    else:
        num_lines = len(seq)
        doc += left
        for i in range(num_lines):
            doc += (NewLine() + seq[i]).indent(indent)
            if i != num_lines - 1:
                doc += line_end_sep
        doc += NewLine() + right
        return doc


def doc_comment(doc: Doc, comment_string='# '):
    docs = list(doc.docs)
    new_docs: List[Union[NewLineToken, str]] = []
    for i, token in enumerate(docs):
        if isinstance(token, NewLineToken):
            new_docs.append(NewLineToken())
            if token.indent > 0:
                new_docs.append(' ' * token.indent)
        else:
            new_docs.append(token)
    docs = new_docs
    new_docs = []
    if docs and not isinstance(docs[0], NewLineToken):
        new_docs.append(comment_string)
    for token in docs:
        if isinstance(token, NewLineToken):
            new_docs.append(token)
            new_docs.append(comment_string)
        else:
            new_docs.append(token)
    docs = new_docs
    ret = Doc()
    ret.docs = docs
    return ret
