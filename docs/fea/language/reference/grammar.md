# Grammar reference

This outline describes the source structure. Closed declaration members, procedure fields,
operation signatures, and units are semantic grammar supplied by the typed registries.

```text
module            = import* declaration*
import            = "import" import-kind identifier "from" string
declaration       = geometry | mesh | material | model | study | sweep | execution

geometry          = "geometry" identifier ("from" geometry-reference)? block
mesh              = "mesh" identifier "on" geometry-reference block
material          = "material" identifier ("from" expression)? block
model             = "model" identifier region-parameter-list? block
study             = "study" identifier block
sweep             = "sweep" identifier "over" study-reference block
execution         = "execution" identifier block

region-parameter-list = "(" region-parameter ("," region-parameter)* ")"
region-parameter  = "region" identifier ":" region-type
block             = "{" statement* "}"
statement         = input | let | assignment | nested-block | output | include | retain
expression        = literal | reference | list | record | unary | binary | call | group
reference         = identifier ("." identifier)*
call              = reference "(" argument-list? ")"
argument-list     = argument ("," argument)*
argument          = expression | identifier "=" expression
quantity          = number unit-expression
```

Line breaks terminate statements when the preceding token cannot continue an expression. A
semicolon may be used as an explicit terminator. Commas separate call arguments, list items, and
record fields. `//` introduces a line comment.

Identifiers are case-sensitive. Keywords are reserved in grammatical positions but may be parsed
as name segments where the owning declaration defines that namespace. Strings use single or double
quotes and support escaped newline, carriage-return, tab, quote, and backslash characters.

Operator precedence, from tighter to looser, is exponentiation, unary sign, multiplication and
division, addition and subtraction, and comparisons. Parentheses override precedence. Assignment
is a statement, not an expression.

The formatter is the canonical source layout authority. Parser recovery retains diagnostics and
continues at safe statement or declaration boundaries so one error does not hide the rest of a
file.
