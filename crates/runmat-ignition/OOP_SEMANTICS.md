# OOP Semantics in Ignition

Ignition models MATLAB-style classes with properties, methods, static members, and overloaded indexing via a runtime registry exposed by `runmat_builtins` and used by `runmat_runtime`.

## Class registration and lookup

- Classes are registered at runtime using the `RegisterClass` instruction, which carries:
  - `name`, optional `super_class`
  - `properties: {name → PropertyDef}` with flags: `is_static`, `is_dependent`, `get_access`, `set_access`, and optional default value
  - `methods: {name → MethodDef}` with flags: `is_static`, `access`, and a `function_name` to invoke
- Lookup helpers:
  - `lookup_property(class, field) → (PropertyDef, owner)` resolves property with inheritance
  - `lookup_method(class, name) → (MethodDef, owner)` resolves method with inheritance
  - `get_class(name)` retrieves the full class def

## Properties

### Instance properties

- Read: `LoadMember(field)` enforces:
  - Not static (`PropertyDef.is_static == false`)
  - Getter access (`get_access != Private`)
  - Dependent properties: if `is_dependent`, the VM prefers calling builtin `get.<field>(obj)`; if absent, it looks for a backing field `<field>_backing`
- Write: `StoreMember(field)` enforces:
  - Not static
  - Setter access (`set_access != Private`)
  - Dependent properties: tries builtin `set.<field>(obj, rhs)` before falling back to direct storage
  - GC barrier: `gc_record_write(old_value, rhs)` when mutating stored references

### Static properties

- Read: `LoadStaticProperty(class, field)` enforces staticness and access, returning a concrete value:
  - If a stored static value exists in the registry for the owner class, return it
  - Else return the property default if provided, else numeric 0
- Write: `StoreMember` on a `ClassRef` sets static properties via `set_static_property_value_in_owner(owner, field, rhs)` with access checks

## Methods

### Instance methods

- `CallMethod(name, argc)` dispatches to:
  1) `lookup_method(obj.class_name, name)` if present and not static
  2) Qualified builtin name `Class.method` with receiver as first arg
  3) Unqualified builtin `method` with receiver as first arg
- Private methods produce `MATLAB:...` access errors
- `LoadMethod(name)` returns a closure capturing the receiver: `Closure { function_name: "Class.method", captures: [obj] }`

### Static methods

- `CallStaticMethod(class, name, argc)` resolves using `lookup_method(class, name)` and enforces `is_static == true`
- `LoadMethod` on a `ClassRef` yields a closure without captures for static methods

## Overloaded indexing

Objects can overload `subsref`/`subsasgn` to handle:

- `o(i, j, ...)` and `o(i, j, ...) = rhs` with the kind string `'()'`
- `o{i, j, ...}` and `o{i, j, ...} = rhs` with the kind string `'{}'`
- `o.field` and `o.field = rhs` with the kind string `'.'`

The VM constructs selector representations to match MATLAB conventions:
- For `()`: a `CellArray` containing numeric or special markers (`":"`, `'end'`, and range descriptors where needed)
- For `{}`: a `CellArray` of indices
- For `.`: a `String` field name

Calls are performed through the `call_method` builtin with the receiver as the first argument and the kind + selectors following. If a class defines neither `subsref` nor `subsasgn`, the VM surfaces `MATLAB:MissingSubsref` / `MATLAB:MissingSubsasgn`.

## Operator overloading

Unary and binary operators prefer object-specific methods exposed as builtins:

- Unary: `uplus`, `uminus`
- Binary arithmetic/matrix: `plus`, `minus`, `mtimes`, `mrdivide`, `power`
- Elementwise: `times`, `rdivide`, `ldivide`, `power`
- Relational: `eq`, `ne`, `lt`, `le`, `gt`, `ge`

When a method is missing, the VM falls back to numeric semantics in `runmat_runtime` where sensible and normalizes failures via mex.

## Dynamic fields and structures

- `LoadMemberDynamic` / `StoreMemberDynamic` support `s.(expr)` and `s.(expr) = rhs` for both objects (subject to access rules) and structs
- For cell-of-struct assignments/reads, the VM maps over elements and returns a cell with the same shape

## Method handles

- `LoadMethod(name)` on an object returns a bound closure handle
- On a class reference, `LoadMethod(name)` returns a static method closure (no captures) when `is_static`
- These closures can be invoked via `feval` or compiled calls when used as function values

## Metaclass and class references

- Metaclass literals (parser-level `?T`) and `classref('T')` enable direct static access lowering to `LoadStaticProperty` and `CallStaticMethod`

## Error identifiers

- Access control violations: private properties/methods → `MATLAB:...` identifiers with precise messages
- Missing overloads: `MATLAB:MissingSubsref` / `MATLAB:MissingSubsasgn`

See `INSTR_SET.md` for the exact instructions and `vm.rs` for enforcement details.
