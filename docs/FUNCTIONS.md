---
title: "Functions"
category: "Reference"
section: "functions"
last_updated: "June 2, 2026"
---

# Functions

Functions are the main way to organize reusable logic in RunMat. You can write named
functions in their own files, define helper functions alongside a script, nest functions
inside one another to share state, and pass functions around as values with handles.
RunMat follows MATLAB's function semantics, so most existing code runs unchanged.

## Defining and calling functions

A function declares its outputs, a name, and its inputs. The simplest functions take a
few arguments and return one value:

```matlab
function y = square(x)
  y = x * x;
end

disp(square(5)); % 25
```

A function can return **multiple values** by listing them in brackets. Callers choose how
many they want — request all of them, or just the first:

```matlab
function [s, d] = sumdiff(a, b)
  s = a + b;
  d = a - b;
end

[s, d] = sumdiff(10, 3); % s = 13, d = 7
s2 = sumdiff(10, 3);     % s2 = 13 (second output discarded)
```

### Where functions live

A function defined in its own file should be named after the file: `square` lives in
`square.m`. Once that file is on a source root, you can call `square` from any script or
function. RunMat does **not** auto-run a function file when you execute it the way it runs
a script — the file defines the function, and you call it by name. See
[Projects](/docs/runtime/getting-started/projects) for how source roots are resolved.

You can also define **local functions** in the same file, after the code that uses them.
A script file or a function file may end with a list of additional functions that are
visible only within that file:

```matlab
disp(area(2, 3)); % 6

function a = area(w, h)
  a = w * h;
end
```

## Optional arguments with `nargin`

`nargin` reports how many arguments the caller actually passed, which lets you fill in
defaults:

```matlab
function y = scale(x, factor)
  if nargin < 2
    factor = 1;
  end
  y = x * factor;
end

scale(5)    % 5
scale(5, 2) % 10
```

`nargout` works the same way for outputs, so a function can avoid computing results the
caller did not ask for.

## Variable arguments: `varargin` and `varargout`

Use `varargin` to accept any number of trailing arguments. Inside the function it is a
cell array of the extra values:

```matlab
function n = countargs(varargin)
  n = numel(varargin);
end

countargs(1, 2, 3, 4) % 4
```

`varargout` does the same for outputs, letting a function return a variable number of
values:

```matlab
function varargout = spread()
  varargout{1} = 1;
  varargout{2} = 2;
  varargout{3} = 3;
end

[a, b, c] = spread(); % a = 1, b = 2, c = 3
```

## Recursion

Functions can call themselves:

```matlab
function y = fact(n)
  if n <= 1
    y = 1;
  else
    y = n * fact(n - 1);
  end
end

fact(5) % 120
```

## Nested functions

A nested function is defined **inside** another function's body. Unlike a local function,
a nested function shares the lexical scope of its parent: it can read and write the
parent's variables, and changes are visible back in the parent. This makes nested
functions useful for accumulating state or building helpers that close over local data:

```matlab
function r = runningTotal()
  total = 0;
  function add(x)
    total = total + x; % updates the parent's `total`
  end
  add(5);
  add(7);
  r = total; % 12
end
```

## Anonymous functions and closures

An anonymous function is written with `@(args) expression`. It captures the values of any
variables it references **at the time it is created** — capture is by value, so later
changes to those variables do not affect the handle:

```matlab
k = 10;
addk = @(x) x + k;
addk(5)  % 15
k = 100;
addk(5)  % still 15 — k was captured as 10
```

## Function handles

A function handle is a first-class value that refers to a function. Create one with `@`
in front of a function name, or as an anonymous function, then call it like any function
or pass it to other functions:

```matlab
h = @fact;       % handle to the fact function above
h(4)             % 24
feval(h, 3)      % 6 — call a handle by value

func2str(@sin)   % "sin" — recover the name/source of a handle
```

Handles are what make higher-order builtins work. `arrayfun` applies a handle to each
element of an array, and `cellfun` applies it to each element of a cell array:

```matlab
arrayfun(@(x) x^2, [1 2 3])     % [1 4 9]
cellfun(@numel, {'ab', 'cde'})  % [2 3]
```

## Persistent and global state

A `persistent` variable keeps its value between calls to the same function. It is private
to that function and initialized empty on the first call:

```matlab
function n = bump()
  persistent c
  if isempty(c)
    c = 0;
  end
  c = c + 1;
  n = c;
end

bump(); bump();
bump() % 3
```

A `global` variable is shared across every function (and the base workspace) that declares
it `global` with the same name:

```matlab
global COUNT
COUNT = 0;
tick(); tick();
disp(COUNT); % 2

function tick()
  global COUNT
  COUNT = COUNT + 1;
end
```

## Argument validation with `arguments` blocks

An `arguments` block declares the size, class, default value, and validators for a
function's inputs. It runs before the function body, so invalid calls are rejected with a
clear error before any work happens:

```matlab
function a = area(w, h)
  arguments
    w (1,1) double {mustBePositive}
    h (1,1) double = 1
  end
  a = w * h;
end

area(2, 3) % 6
area(4)    % 4 — h defaults to 1
```

Each declaration can specify, in order:

- a **size** like `(1,1)` or `(1,:)`,
- a **class** such as `double` or `char`,
- a brace-enclosed list of **validators** like `{mustBePositive}`,
- and a **default value** with `= value`, which also makes the argument optional.

Validation failures raise specific identifiers you can catch — for example a size
mismatch raises `RunMat:ArgumentValidationSize`:

```matlab
try
  area([1 2], 3); % w must be 1x1
catch e
  disp(e.identifier); % RunMat:ArgumentValidationSize
end
```

RunMat ships the common `mustBe*` validators (such as `mustBePositive`,
`mustBeNonnegative`, `mustBeInteger`, and `mustBeNonempty`).

## Introspection helpers

Functions can inspect their own call context:

- `nargin` / `nargout` — how many inputs/outputs are in play for the current call.
- `narginchk(min, max)` / `nargoutchk(min, max)` — assert the count is in range, raising
  `RunMat:NotEnoughInputs` (or the too-many counterpart) otherwise.

```matlab
function y = checked(a, b)
  narginchk(2, 2);
  y = a + b;
end

checked(1, 2) % 3
checked(1)    % error: RunMat:NotEnoughInputs
```

## Command syntax

When you call a function with bare words instead of parentheses, RunMat parses each word
as a string argument. These two calls are equivalent:

```matlab
disp hello       % command syntax
disp('hello')    % function syntax
```

Command syntax is convenient for quick interactive calls; use function syntax whenever you
need to pass computed values rather than literal words.

## Notes and limitations

- A function file is not auto-executed like a script. Running it defines the function;
  call the function by name to run it.
- Variables created dynamically by `eval` are not statically visible to code compiled
  later in the same scope. Prefer ordinary assignment when you can.
- `arguments` blocks support size, class, default, and the common `mustBe*` validators;
  more elaborate MATLAB validation forms may not all be available yet.

For how functions fit into RunMat's broader MATLAB compatibility, see
[MATLAB Language Compatability](/docs/runtime/getting-started/compatability). For how calls
are resolved and dispatched at runtime, see
[Callable Resolution & Function Dispatch](/docs/runtime/vm/dispatch).
