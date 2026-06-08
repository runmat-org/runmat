---
title: "Classes (classdef)"
category: "Reference"
section: "classes"
last_updated: "June 1, 2026"
---

# Classes (classdef)

RunMat supports MATLAB-style object-oriented programming through `classdef`. You can
define value classes and handle classes with properties, methods, inheritance, operator
overloading, events, enumerations, and custom indexing, then use them with the same
syntax you would in MATLAB.

A class is defined in its own file named after the class: a class `Vec2` lives in
`Vec2.m`. RunMat discovers class files from your project's source roots, so once `Vec2.m`
is on the path you can construct and use `Vec2` from any script or function. See
[Projects](/docs/runtime/getting-started/projects) for how source roots and packages are
resolved.

## A first class

Save this as `Vec2.m`:

```matlab
classdef Vec2
  properties
    x
    y
  end
  methods
    function obj = Vec2(x, y)
      if nargin > 0
        obj.x = x;
        obj.y = y;
      else
        obj.x = 0;
        obj.y = 0;
      end
    end
    function m = magnitude(obj)
      m = sqrt(obj.x^2 + obj.y^2);
    end
    function c = plus(a, b)
      c = Vec2(a.x + b.x, a.y + b.y);
    end
  end
  methods (Static)
    function u = unit()
      u = Vec2(1, 0);
    end
  end
end
```

Now use it from a script:

```matlab
p = Vec2(3, 4);
disp(class(p));      % Vec2
disp(isa(p, 'Vec2')); % 1
disp(p.magnitude()); % 5
```

Constructing `Vec2(3, 4)` calls the constructor and returns a genuine object: `class`
reports `Vec2` and `isa` recognizes the type.

### Calling methods

Methods can be called with either dot syntax or function syntax. Both dispatch to the
same method:

```matlab
p = Vec2(3, 4);
m1 = p.magnitude();   % dot-call
m2 = magnitude(p);    % function-style call
```

Static methods are called on the class itself:

```matlab
u = Vec2.unit();
disp(u.x); % 1
```

## Value classes vs. handle classes

By default a class is a **value class**: assigning it or passing it to a function copies
it. A method only changes the caller's object if it returns the modified object and the
caller assigns the result.

A class that derives from `handle` is a **handle class** with reference semantics: all
variables that point to the object see mutations, and methods can modify the object
in place.

```matlab
% V.m  -> value class
classdef V
  properties
    x
  end
  methods
    function obj = V(x)
      obj.x = x;
    end
    function bump(obj)        % does NOT affect the caller
      obj.x = obj.x + 1;
    end
    function obj = bump2(obj) % returns the updated copy
      obj.x = obj.x + 1;
    end
  end
end
```

```matlab
% H.m  -> handle class
classdef H < handle
  properties
    x
  end
  methods
    function obj = H(x)
      obj.x = x;
    end
    function bump(obj)        % mutates in place
      obj.x = obj.x + 1;
    end
  end
end
```

```matlab
v = V(1);
v.bump();          % discarded: value copy was mutated, not v
a = v.x;           % 1
v = v.bump2();     % reassigned
b = v.x;           % 2

h = H(1);
h.bump();          % mutates the object in place
c = h.x;           % 2
```

Handle classes also support the lifecycle builtins `isvalid` and `delete`.

## Properties

Properties are declared in `properties` blocks. A block can carry attributes that change
how its properties behave.

```matlab
classdef Circle
  properties
    radius = 0          % regular property with a default
  end
  properties (Dependent)
    area                % computed on access
  end
  methods
    function obj = Circle(r)
      if nargin > 0
        obj.radius = r;
      end
    end
    function a = get.area(obj)
      a = pi * obj.radius^2;
    end
  end
end
```

```matlab
c = Circle(2);
disp(c.radius); % 2
disp(c.area);   % 12.5664
```

Supported property attributes include:

| Attribute | Behavior |
| --- | --- |
| `Constant` | Read-only class constant accessed as `ClassName.Prop`. Assigning to it raises `RunMat:PropertyReadOnly`. |
| `Dependent` | Value is produced by a `get.Prop` method (and optionally written through `set.Prop`) rather than stored. |
| `Static` | Class-level storage accessed as `ClassName.Prop`. Accessing it through an instance raises `RunMat:PropertyStaticAccess`. |
| Access (`Access`, `GetAccess`, `SetAccess` = `public` / `private` / `protected`) | Controls who may read or write the property. |

```matlab
classdef C
  properties (Constant)
    K = 3
  end
end
% C.K returns 3; C.K = 9 raises RunMat:PropertyReadOnly
```

## Methods

Methods live in `methods` blocks. A block can be marked with attributes that apply to all
methods inside it.

| Attribute | Behavior |
| --- | --- |
| `Static` | Called as `ClassName.method(...)`; receives no implicit object. |
| `Access = private` / `protected` | Restricts who may call the method. A blocked call raises `RunMat:MethodPrivate`. |
| `Abstract` | Declares a method with no body that subclasses must implement (see below). |
| `Sealed` | Prevents subclasses from overriding the method (`RunMat:MethodSealed`). |

### Access control and constructors

Constructors follow the same access rules, which lets you implement factory patterns:

```matlab
classdef A
  methods (Access = private)
    function obj = A()
    end
  end
  methods (Static)
    function obj = make()
      obj = A();   % allowed: inside the class
    end
  end
end
```

```matlab
% A()       -> RunMat:MethodPrivate
b = A.make(); % OK
disp(class(b)); % A
```

## Inheritance

A subclass lists its base class after `<`. Override a method by redefining it, and call
the base implementation with `method@Base(obj)`.

```matlab
% A.m
classdef A
  methods
    function v = f(obj)
      v = 3;
    end
  end
end
```

```matlab
% B.m
classdef B < A
  methods
    function v = f(obj)
      v = f@A(obj) + 4;  % call the base method, then extend it
    end
  end
end
```

```matlab
b = B();
disp(b.f()); % 7
```

To call a base **constructor**, use `obj@Base(args)` as the first statement of the
subclass constructor.

### Abstract classes

Mark a class or methods `Abstract` to require subclasses to provide an implementation.
Instantiating an abstract class, or a subclass that has not implemented every abstract
method, raises `RunMat:AbstractMethodMissing`.

```matlab
% A.m
classdef (Abstract) A
  methods (Abstract)
    y = f(obj)
  end
end
```

```matlab
% B.m
classdef B < A
  methods
    function y = f(obj)
      y = 42;
    end
  end
end
```

```matlab
b = B();
disp(b.f()); % 42
```

## Operator overloading

Define methods with MATLAB's operator names to overload operators for your class:

```matlab
classdef Money
  properties
    amount
  end
  methods
    function obj = Money(v)
      obj.amount = v;
    end
    function out = plus(a, b)
      out = Money(a.amount + b.amount);
    end
  end
end
```

```matlab
a = Money(10);
b = Money(7);
c = a + b;
disp(class(c));  % Money
disp(c.amount);  % 17
```

Common operator-to-method mappings:

| Operator | Method | Operator | Method |
| --- | --- | --- | --- |
| `a + b` | `plus` | `a == b` | `eq` |
| `a - b` | `minus` | `a ~= b` | `ne` |
| `a * b` | `mtimes` | `a < b` | `lt` |
| `a .* b` | `times` | `a > b` | `gt` |
| `a / b` | `mrdivide` | `a <= b` | `le` |
| `-a` | `uminus` | `a >= b` | `ge` |

## Events

A handle class can declare events and notify listeners. Register callbacks with
`addlistener` and raise events with `notify`.

```matlab
% H.m
classdef H < handle
  events
    Tick
  end
  methods
    function fire(obj)
      notify(obj, 'Tick');
    end
  end
end
```

```matlab
global G;
G = 0;
h = H();
addlistener(h, 'Tick', @on_tick);
h.fire();
disp(G); % 1

function on_tick(src, evt)
  global G;
  G = G + 1;
end
```

Callbacks can be function handles, static methods (`@CB.on_tick`), or function names.

## Custom indexing

Override `subsref` and `subsasgn` to customize how `.`, `()`, and `{}` behave on your
objects. Delegate to the default behavior with `builtin('subsref', ...)` /
`builtin('subsasgn', ...)`.

```matlab
classdef DotSpy
  properties
    x
  end
  methods
    function obj = DotSpy(v)
      obj.x = v;
    end
    function out = subsref(obj, S)
      if strcmp(S(1).type, '.')
        out = 999;
      else
        out = builtin('subsref', obj, S);
      end
    end
  end
end
```

```matlab
d = DotSpy(3);
disp(d.x); % 999 (intercepted)
```

## Enumerations

Declare an `enumeration` block to define a fixed set of named members, accessed as
`ClassName.Member`.

```matlab
classdef Color
  enumeration
    Red
    Blue
  end
end
```

```matlab
c = Color.Red;
```

## Packages

A package groups related classes under a namespace. You create one by placing class files
in a folder whose name starts with `+`; the folder name (without the `+`) is the package
name.

For example, this project puts a `Point` class in a `geom` package:

```text
myproject/
  +geom/
    Point.m       % defines classdef Point
  main.m
```

```matlab
% +geom/Point.m
classdef Point
  properties
    x
  end
  methods
    function obj = Point(v)
      obj.x = v;
    end
  end
end
```

From `main.m`, refer to the class by its **qualified name**, `geom.Point`:

```matlab
p = geom.Point(42);
disp(class(p)); % geom.Point
disp(p.x);      % 42
```

To use the short name instead, import the class first:

```matlab
import geom.Point;
p = Point(42);   % same as geom.Point(42)
```

Packages also work with inheritance: a class can extend a packaged base class and call its
methods with `method@package.Base(obj)`.

```matlab
classdef B < pkg1.A
  methods
    function v = f(obj)
      v = f@pkg1.A(obj) + 1;  % call pkg1.A's f, then extend it
    end
  end
end
```

See [Module Composition](/docs/runtime/compiler/modules) for package resolution and
`import` precedence rules.

## Notes and limitations

- Each class must be defined in its own `.m` file named after the class.
- Object display is currently compact (for example `disp(p)` shows a summary line rather
  than MATLAB's `ClassName with properties:` block).
- The metaclass operator `?ClassName` is accepted but does not yet return a full
  `meta.class` object.

For how `classdef` fits into RunMat's broader MATLAB compatibility, see
[MATLAB Language Compatability](/docs/runtime/getting-started/compatability). For the
runtime internals of object dispatch, see
[Callable Resolution & Function Dispatch](/docs/runtime/vm/dispatch).
