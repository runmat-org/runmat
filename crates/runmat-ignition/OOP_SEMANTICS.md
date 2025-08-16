# OOP Semantics in Ignition

Ignition integrates an object system sufficient to model MATLAB classes with properties, methods, and overloaded indexing.

## Properties and methods
- Instance properties: `LoadMember`/`StoreMember` check access control.
- Instance methods: `LoadMethod`/`CallMethod` dispatch through the runtime registry.
- Static properties/methods: `LoadStaticProperty`/`CallStaticMethod` with a class reference (`classref('T')`) or metaclass `?T` lowering.

## Overloaded indexing
- The VM routes `o(...)`/`o{...}`/`o.field` to `subsref` and `o(...)=rhs`/`o{...}=rhs`/`o.field=rhs` to `subsasgn`, by constructing selector cells mirroring MATLAB’s conventions.
- If an object lacks `subsref`/`subsasgn`, errors are surfaced as `MATLAB:MissingSubsref`/`MATLAB:MissingSubsasgn`.

## Operator overloading
- Binary/unary operators delegate to runtime methods (`plus`, `minus`, `mtimes`, etc.). If not implemented, numeric fallbacks apply; failures are normalized via mex.

## Class registry
- The registry lives in `runmat-runtime` and is test‑seeded via `__register_test_classes()`.
- Access checks and test fixtures (e.g., `Point`, `OverIdx`) are maintained in `runmat-runtime` for clarity.
