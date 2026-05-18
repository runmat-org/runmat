# RunMat Architecture Notes

Status: historical/exploratory. `TARGET_MODEL.md` and `PLAN.0.md` through `PLAN.6.md` are the authoritative target and implementation plan. Some sketches below intentionally predate later decisions such as entrypoint target/origin/policy, Rust-like lazy futures, spawn safety, and the MIR-centered analysis pipeline.

## Goal

RunMat should feel MATLAB-native at the source level while using a modern, explicit compiler and runtime internally. The target is the best runtime for math: strong tensor and linear algebra linting, shape-aware analysis, excellent LSP hints, safe parallel dispatch, and aggressive tensor fusion, without forcing users to rewrite existing MATLAB-style code.

## Core Principles

- MATLAB compatibility is the default user experience.
- Internal compiler truth should be explicit and structured, not reconstructed later.
- HIR should model semantic ownership and identity, not VM slot layout.
- Runtime slot numbering is an implementation detail derived from HIR.
- Package composition should come from config, while `import` remains source-level name ergonomics.
- Tensor, type, shape, purity, and fusibility facts should become first-class compiler products.

## Settled Direction

### HIR

HIR should grow explicit structure that survives lowering:

- explicit function structure
- explicit binding identity
- explicit lexical ownership
- explicit module and entrypoint structure
- stable IDs for semantic entities, including `ModuleId`, `EntrypointId`, `FunctionId`, `ClassId`, `BindingId`, `ExprId`, and `StmtId`

HIR should be the semantic backbone of the system. Downstream crates should stop reconstructing locals, functions, and scopes from flattened HIR bodies plus side maps.

Functions should stop being statements. Bindings should stop being VM slots. The semantic model should become explicit enough that VM slot layout, capture layout, and workspace export rules are all derived from HIR rather than being implicit in current lowering maps.

### Nested Functions

Default nested function behavior should preserve true MATLAB semantics:

- shared lexical capture
- mutation of captured parent bindings is visible across nested functions

We intend to add a single refinement keyword: `isolated`.

`isolated` should be a strict restriction, not a second function model:

- an `isolated` function may not lexically capture enclosing local bindings
- required outer values must be passed explicitly as parameters
- `isolated` does not imply purity, determinism, or side-effect freedom

This keeps full compatibility by default while giving the compiler and runtime one explicit escape hatch for simpler lowering and stronger optimization.

### Scripts, Modules, and Entrypoints

Internally, scripts should disappear as a primitive concept.

The system should instead be designed around:

- config-defined packages and dependencies
- config-defined entrypoints
- source files lowering into modules, functions, and entrypoint bodies

This means "script" remains a source authoring style, but not a compiler or runtime primitive.

Planned config concepts:

- `[dependencies]` for package composition
- `[[entrypoints]]` for execution roots
- likely `[sources]` or equivalent for local source roots and project discovery

### Imports and Composition

Package composition should be config-driven.

- `[dependencies]` defines what packages are available
- source-level `import` controls name visibility and ergonomics
- canonical internal symbol identity should be qualified and stable

Goal: make porting existing MATLAB code mostly a matter of adding config rather than rewriting source trees.

### Resolved Composition Model

The resolved direction is:

- external composition is driven by config
- source-level `import` is not the dependency mechanism
- scripts disappear internally into entrypoints
- module and package identity are canonical and qualified

This is intended to preserve MATLAB-friendly source ergonomics while giving RunMat a modern build and package graph.

### Config as the Composition Source of Truth

The config file should be the place where composition is declared.

Target concepts:

- `[dependencies]` declares external packages or local package dependencies
- `[sources]` or equivalent declares local source roots for project discovery
- `[[entrypoints]]` declares what executable roots exist in the project

This means:

- dependency availability is explicit and reproducible
- source discovery is explicit enough for tooling and builds
- execution roots are explicit rather than inferred from arbitrary files

### Proposed `runmat.toml` Surface

The intended surface should be close to this:

```toml
[package]
name = "thermal-demo"
version = "0.1.0"

[sources]
roots = ["src", "lib", "examples"]

[dependencies]
linalg = { path = "../linalg" }
signal = { path = "../signal" }

[[entrypoints]]
name = "main"
path = "src/main"

[[entrypoints]]
name = "demo"
path = "examples/thermal_surface_demo"
```

The important semantic intent is:

- `[package]` names the current package or project
- `[sources]` declares where local source discovery should happen
- `[dependencies]` declares other packages available for composition
- `[[entrypoints]]` declares executable roots by stable logical names

The final textual shape can evolve, but those concepts should remain.

### Provisional Exact `runmat.toml` Schema

For the next major revision, the working assumption should be that `runmat.toml` has a single canonical project-manifest shape. We can revise it later, but implementation should target one coherent schema rather than multiple equally valid syntaxes.

Preferred working shape:

```toml
[package]
name = "thermal-demo"
version = "0.1.0"

[sources]
roots = ["src", "lib", "examples"]

[dependencies]
linalg = { path = "../linalg" }
signal = { path = "../signal" }

[[entrypoints]]
name = "main"
path = "src/main"

[[entrypoints]]
name = "demo"
path = "examples/thermal_surface_demo"
```

This gives us:

- dependencies that read naturally like Cargo
- entrypoints that behave like first-class runnable targets
- room to grow fields later without changing section meaning
- a shape that is easy to parse, validate, and extend

### Provisional Manifest Types

The config should roughly correspond to something like:

```rust
pub struct RunMatProjectManifest {
    pub package: PackageManifest,
    pub sources: SourcesManifest,
    pub dependencies: BTreeMap<String, DependencySpec>,
    pub entrypoints: Vec<EntrypointSpec>,
}

pub struct PackageManifest {
    pub name: String,
    pub version: Option<String>,
}

pub struct SourcesManifest {
    pub roots: Vec<String>,
}

pub struct DependencySpec {
    pub path: Option<String>,
    pub version: Option<String>,
    pub package: Option<String>,
}

pub struct EntrypointSpec {
    pub name: String,
    pub path: Option<String>,
    pub module: Option<String>,
    pub function: Option<String>,
}
```

This is still provisional, but it is concrete enough for implementation planning.

### Manifest Interpretation Rules

#### `[package]`

- `name` is required
- `version` is optional for local development, but useful for packaging later
- package name should become the canonical package identity unless overridden by future tooling conventions

#### `[sources]`

- `roots` is required and must contain at least one path
- paths are relative to the directory containing `runmat.toml`
- roots define the local code search space for the package
- roots should not implicitly pull in dependency code

#### `[dependencies]`

- dependency keys are the local dependency names seen by the package
- `path` is the initial required supported form
- `version` and other sources can come later
- `package` can be reserved for future dependency renaming or alternate package identity if needed

Important rule:

- the dependency table key is the canonical top-level package name unless explicit alias/package remapping is introduced later

#### `[[entrypoints]]`

- `name` is the stable logical name of the runnable target
- exactly one of the following entrypoint forms should be valid:
  - `path`
  - `module` + `function`
- `path` identifies a script-like file or function-bearing source file relative to the project root
- `module` + `function` identifies a canonical callable root

### Entrypoint Resolution Rules

For the first implementation, the simplest consistent rule set should be:

- `path = "src/main"` means resolve the source file at that project-relative location, with `.m` inferred if omitted
- if the target file is script-like, lower it to a synthetic entry function
- if the target file already contains a top-level function root, resolve the entrypoint to that function
- `module` + `function` should bypass file-style entry and resolve directly through canonical module identity

### Validation Rules

At minimum, manifest validation should enforce:

- `[package].name` exists
- `[sources].roots` exists and is non-empty
- every source root exists or produces a clear diagnostic
- every dependency has at least one supported locator field
- every entrypoint has exactly one valid target form
- no duplicate dependency names
- no duplicate entrypoint names
- `path` entrypoints resolve under the project root

### Diagnostics Principles for Manifest Errors

Manifest diagnostics should be project-level and user-facing.

Examples:

- missing required `[sources].roots`
- entrypoint `main` points to missing path `src/main`
- dependency `signal` has unsupported locator fields
- entrypoint `server` sets both `path` and `module`

These should talk in terms of:

- manifest sections
- project-relative paths
- dependency names
- entrypoint names

not internal loader details.

### Config Semantics

#### `[sources]`

`[sources]` should define where local user code lives.

Likely behavior:

- treat configured roots as local source search roots
- discover script-like files, function files, class files, and module files under those roots
- allow familiar MATLAB-style project layouts to remain mostly intact

This is the key to minimal-friction local project porting.

#### `[dependencies]`

`[dependencies]` should define external package availability.

Likely supported forms:

- local path dependency
- versioned dependency later
- possibly git or registry dependency later

Important semantic rule:

- dependency keys become canonical top-level package identities unless explicitly aliased

#### `[[entrypoints]]`

`[[entrypoints]]` should define named execution roots.

Each entry should identify a script-like file or module/function target that can lower to an executable entry function.

The system should support entrypoints for:

- script-like runnable demos
- top-level app entry functions
- examples
- tests and benchmarks later if desired

### Entrypoint Target Shape

The working shape should now be considered:

```toml
[[entrypoints]]
name = "main"
path = "src/main"

[[entrypoints]]
name = "demo"
path = "examples/thermal_surface_demo"

[[entrypoints]]
name = "server"
module = "app.server"
function = "main"
```

The semantic requirement remains:

- entrypoints must identify stable execution roots
- script-like targets lower to synthetic entry functions
- function-oriented targets resolve to canonical callable identities

### User Code and Config Examples

These examples are here to lock down the intended user-facing ergonomics.

#### Example 1: Simple Port of a MATLAB-Style Local Project

Project layout:

```text
project/
  runmat.toml
  src/
    main.m
    solve_system.m
    normalize_data.m
```

`runmat.toml`:

```toml
[package]
name = "project"

[sources]
roots = ["src"]

[[entrypoints]]
name = "main"
path = "src/main"
```

`src/main.m`:

```matlab
A = rand(10, 10);
b = rand(10, 1);
x = solve_system(A, b);
disp(x);
```

Notes:

- This should be a minimal-friction port.
- Local helper functions should be discoverable via `src` without needing a dependency declaration.
- `main.m` should lower to a synthetic entry function.
- `A`, `b`, and `x` should be workspace-visible when run in interactive or desktop mode.

#### Example 2: Local Project with Subdirectories and Source Roots

Project layout:

```text
project/
  runmat.toml
  app/
    run_analysis.m
  lib/
    solve_system.m
    filters/
      smooth_signal.m
```

`runmat.toml`:

```toml
[package]
name = "project"

[sources]
roots = ["app", "lib"]

[[entrypoints]]
name = "analysis"
path = "app/run_analysis"
```

Notes:

- This allows a project with multiple MATLAB-style source roots to port without flattening the tree.
- Local source roots should be enough to discover `solve_system` and `smooth_signal`.
- No dependency declaration should be needed for code that is part of the same package.

#### Example 3: Dependency Plus MATLAB-Style Imports

Project layout:

```text
app/
  runmat.toml
  src/
    main.m
```

`runmat.toml`:

```toml
[package]
name = "app"

[sources]
roots = ["src"]

[dependencies]
linalg = { path = "../linalg" }
signal = { path = "../signal" }

[[entrypoints]]
name = "main"
path = "src/main"
```

`src/main.m`:

```matlab
import linalg.*
import signal.fft.fft2

A = rand(10, 10);
b = rand(10, 1);
x = solve(A, b);
F = fft2(A);
```

Notes:

- `linalg` and `signal` become available because of `[dependencies]`.
- `import` only controls visibility and unqualified resolution.
- Without `[dependencies]`, the imports should fail to resolve.
- The canonical internal identities should still be qualified even if source code uses unqualified imported names.

#### Example 4: Function-Oriented Entrypoint

Project layout:

```text
project/
  runmat.toml
  src/
    app_main.m
```

`runmat.toml`:

```toml
[package]
name = "project"

[sources]
roots = ["src"]

[[entrypoints]]
name = "main"
path = "src/app_main"
```

`src/app_main.m`:

```matlab
function app_main()
    disp("hello");
end
```

Notes:

- The entrypoint should resolve to a callable root even though the file already contains an explicit function.
- This is the function-oriented counterpart to script-like entrypoints.
- Internally, execution should still begin from an explicit entrypoint item.

#### Example 5: Notebook or REPL-Like Session Intuition

Notional project config:

```toml
[package]
name = "scratch"

[sources]
roots = ["."]
```

Interactive submission:

```matlab
t = linspace(0, 1, 100);
signal = sin(2 * pi * t);
```

Notes:

- Interactive execution may not require a declared named entrypoint for each cell or snippet.
- The runtime should synthesize ephemeral entrypoints internally.
- The same semantic model should still apply: submitted code lowers to a synthetic entry function and exports top-level bindings into the session workspace.

### Import and Composition Diagnostics

The model should support clear diagnostics for:

- importing a package that is not available in `[dependencies]`
- ambiguous unqualified imports
- unresolved entrypoint targets
- source roots that do not contain the requested entrypoint/module/function

These diagnostics should talk in terms of:

- packages
- entrypoints
- source roots
- canonical symbol identities

rather than low-level loader internals.

### Role of `import`

`import` should remain a source-level visibility and resolution convenience, closer to MATLAB semantics than to a build-system primitive.

`import` should be responsible for:

- bringing names into scope
- enabling unqualified references where allowed
- expressing user intent in source code

`import` should not be responsible for:

- discovering packages
- fetching dependencies
- defining the package graph
- implicitly creating execution roots

This is one of the main balances we resolved:

- keep the surface familiar to MATLAB users
- keep the composition model modern and explicit underneath

### Canonical Internal Identity

Internally, symbols should have canonical qualified identities.

That means:

- functions are identified by a canonical module or package path plus local name
- imports may expose unqualified names, but do not change canonical identity
- tooling, analysis, and runtime should reason about canonical identities rather than source-local spelling alone

This is important for:

- cross-module analysis
- stable LSP navigation
- caching
- dependency resolution
- import conflict diagnostics

### Local Project Code vs External Dependencies

We resolved that local project code should remain easy to port.

Practical split:

- local source trees should be discovered through configured source roots
- external packages should be declared in `[dependencies]`

This allows a MATLAB-style codebase to port with minimal changes:

- keep familiar source files and directory layout where practical
- add config describing source roots, dependencies, and entrypoints
- avoid forcing users to rewrite all imports or code structure up front

### Scripts and Entrypoints

Source-level scripts may continue to exist as an authoring style, but internally:

- script-like files lower to synthetic entry functions
- entrypoints are first-class config concepts
- execution begins from declared entrypoints, not from implicit "current file" semantics inside the compiler model

This keeps script execution intuitive while still giving the system:

- explicit roots
- cleaner caching
- cleaner tooling
- cleaner package composition

### Qualified Names and Aliasing

The current bias is:

- canonical package/module names should be explicit
- imports may provide source-level convenience
- aliasing, if supported, should be explicit rather than implicit

This follows the general "Rust-shaped" resolution we discussed:

- config declares what exists
- canonical names remain stable
- local source imports are ergonomic overlays

### Porting Target

The porting target is intentionally ambitious:

- an existing MATLAB codebase should mostly port by adding config
- local source files should remain source-compatible as often as possible
- dependencies and entrypoints should become explicit in config

The desired user experience is:

- add `runmat.toml`
- declare source roots
- declare dependencies
- declare entrypoints
- run the project with little or no source rewrite

### HIR Relationship

The HIR should reflect resolved composition semantics:

- modules should carry resolved imports
- functions and bindings should belong to canonical modules
- calls should resolve to canonical callable identities where possible
- entrypoints should be explicit semantic items

This is part of why module composition and semantic call resolution belong in the same architecture conversation rather than being bolted on later.

### Analysis Facts

HIR should carry stable semantic IDs and resolved structure.

Derived facts should live in analysis products keyed by those IDs:

- types
- tensor and shape facts
- purity/effect summaries
- fusibility and dispatch facts
- capture and aliasing facts where relevant

In short:

- HIR = semantic truth
- analysis store = inferred truth

### Workspace Variables

Workspace behavior should be explicit and intuitive. The desktop variable inspector should not be an accident of VM slot numbering or lexical locals leaking out of execution.

Target rule:

- the workspace shows values that belong to the active module or interactive session workspace
- function locals do not appear in the workspace unless explicitly surfaced by tooling or debugging views
- nested-function captures do not appear as separate workspace variables
- module-level bindings and interactive top-level bindings do appear in the workspace
- entrypoint execution updates the workspace only through bindings that are designated as workspace-visible

This implies a distinction between:

- lexical bindings
- module/workspace bindings
- runtime frame locals

Those are related, but not the same thing.

### Runtime and Memory

Language-level behavior should remain MATLAB-ish:

- no ownership syntax for ordinary code
- value-like semantics
- ordinary scripts and functions should work unchanged

Implementation strategy should be modern and explicit:

- deterministic release should be primary
- GC should support heap-retained structures, closures, containers, and act as backup
- device residency and actual freeing should be modeled separately
- runtime ownership and lifetime should not leak into source semantics

Long term, the runtime should distinguish:

- liveness
- residency
- ownership

Those should not remain conflated.

## What This Unlocks

If done well, this architecture should support:

- more accurate tensor and linear algebra linting
- richer LSP symbol and shape hints
- better cross-function and cross-module analysis
- safer and more aggressive tensor fusion
- cleaner parallel dispatch boundaries
- a more principled accelerate and GC integration story

## Refined Core HIR Type Model

The current `runmat-hir` model centers `VarId` and keeps functions inside `HirStmt::Function`. The target model should instead center explicit semantic items.

### Core IDs

Likely core IDs:

- `ModuleId`
- `FunctionId`
- `EntrypointId`
- `BindingId`
- `ExprId`
- `StmtId`

At this stage, `ScopeId` is intentionally not required as a core primitive. We may add it later if CFG, diagnostics, or block-local analyses need it, but explicit function ownership plus nested blocks should be enough for the first migration.

### Top-Level Containers

Preferred high-level shape:

- `HirAssembly` as the resolved collection of modules and entrypoints
- `HirModule` as the authored code unit
- `HirEntrypoint` as the execution root
- `HirFunction` as the uniform function representation

`HirEntrypoint` should point at a `FunctionId`, not at raw statement lists. Script-like files should lower to synthetic entry functions.

### `HirModule`

`HirModule` should own:

- module identity
- qualified module name
- source identity
- imports
- function IDs
- class IDs or class items
- optionally a synthetic entry function generated from script-like top-level executable code

This lets scripts disappear internally while preserving source-level ergonomics.

### `HirFunction`

`HirFunction` should become the central executable semantic node. It should carry:

- `FunctionId`
- owning `ModuleId`
- optional parent `FunctionId` for nested functions
- function name
- function kind
- parameters
- outputs
- locals
- captures
- function modifiers
- body block
- source span

Likely function kinds:

- named function
- anonymous function
- synthetic entry function
- class method

Likely function modifiers:

- `isolated`
- `has_varargin`
- `has_varargout`

### Bindings

Bindings should become first-class semantic items. A `BindingId` should represent the semantic identity of a declared or implicit binding, not its runtime slot.

Each binding should likely carry:

- binding identity
- owner (`FunctionId` or `ModuleId`)
- source name
- declaration span
- binding role
- binding storage class
- workspace visibility classification

Binding roles likely include:

- parameter
- output
- local
- module binding
- implicit `ans`

Binding storage classes likely include:

- lexical
- global
- persistent

Important refinement from pressure testing: captures should not be modeled as distinct bindings in core HIR. A captured binding is still the same semantic binding owned by its declaring function. The capture should be represented as a relation recorded on the capturing function.

### Captures

Nested shared lexical capture semantics require:

- parent-owned bindings remain the semantic source of truth
- nested functions record which outer bindings they capture
- mutation through nested functions still targets the same logical binding

So the model should prefer:

- `HirFunction.captures: Vec<CapturedBinding>`

rather than introducing separate capture bindings in the core HIR identity model.

Runtime may later lower captured mutable bindings into environment cells, but that is a runtime/layout concern, not a semantic HIR identity concern.

### Statements and Blocks

Bodies should remain block-structured. Statements should have stable IDs.

Key change:

- `HirStmt::Function` should go away

Functions should instead be module-owned or class-owned items. This removes the need for downstream passes to rediscover function inventories by walking statement lists.

Statements should continue to cover:

- expression statements
- assignments
- control flow
- `global` and `persistent` declarations
- imports
- class definitions or class item references
- loop and return control

### Assignment Targets

Assignment targets should likely unify under a single place/lvalue concept rather than using separate "assign plain var" vs "assign lvalue" forms.

Likely target concept:

- `HirPlace`

with variants for:

- binding
- member
- dynamic member
- index
- cell index

### Expressions

Expressions should carry stable `ExprId`s and remain structurally similar to the current HIR where it still makes sense.

Important change: calls should become semantically resolved rather than stringly typed.

Preferred structure:

- `HirExprKind::Call(HirCall)`
- `HirExprKind::AnonymousFunction(FunctionId)`

instead of string-only `FuncCall(String, ...)` and inline anonymous-function bodies.

### Calls

Calls should distinguish:

- direct function calls
- builtin calls
- imported calls
- dynamic expression calls
- method or dotted-invoke syntax forms

The callee should carry semantic identity where possible, not just source spelling.

This is a major enabler for:

- module composition
- LSP symbol resolution
- builtin summaries
- purity/fusion analysis
- shape-aware linting

### Classes

Class methods should share the same `HirFunction` representation as ordinary functions rather than being represented only as statements nested inside class member blocks.

This keeps one executable semantic model across:

- named functions
- nested functions
- anonymous functions
- synthetic entry functions
- class methods

### Focused Class / Struct / Object Model Pass

The current codebase already has meaningful class and object runtime behavior, but the HIR and type system do not yet model that behavior explicitly enough.

Current runtime-side evidence includes:

- `runmat-builtins::ObjectInstance`
- `runmat-builtins::ClassDef`
- class/property/method lookup helpers
- `Value::Object`
- `Value::HandleObject`
- runtime overloaded dispatch for `subsref`, `subsasgn`, and operator methods

This means the next revision should not treat classes as merely parser decoration. It should explicitly bridge the HIR/type system to the runtime object model.

### Structural Structs vs Nominal Classes

The language should make a strong distinction between:

- structural structs
- nominal classes/objects

#### Structs

Structs should remain structural.

Properties:

- field sets matter
- known fields can be refined through control flow
- field-sensitive inference should remain possible
- `Type::Struct { known_fields }` or its `TypeFact` successor is still the right conceptual model

This matches the current inference design, which already refines struct field knowledge through control flow and member/lvalue analysis.

#### Classes / Objects

Classes should be nominal.

Properties:

- identity is based on declared class identity, not field shape
- inheritance matters
- property lookup and method lookup should resolve through the class hierarchy
- runtime behavior may distinguish value-object and handle-object semantics

This is different from structs and should stay different in the design.

### Add `ClassId`

The HIR/type system should add:

- `ClassId`

Classes should become first-class semantic items in the same way that functions are becoming first-class semantic items.

That implies:

- modules own class identities
- methods refer to `FunctionId`
- class lookup is not just name-string matching in late runtime code

### Provisional `HirClass` Direction

The current `HirClass` sketch is too light. We should aim more in this direction:

```rust
pub struct ClassId(pub u32);

pub struct HirClass {
    pub id: ClassId,
    pub module: ModuleId,
    pub name: String,
    pub super_class: Option<ClassId>,
    pub properties: Vec<ClassProperty>,
    pub methods: Vec<FunctionId>,
    pub events: Vec<ClassEvent>,
    pub enumerations: Vec<ClassEnumeration>,
    pub arguments: Vec<ClassArgumentBlock>,
    pub span: Span,
}

pub struct ClassProperty {
    pub name: String,
    pub modifiers: ClassPropertyModifiers,
}

pub struct ClassPropertyModifiers {
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub get_access: AccessLevel,
    pub set_access: AccessLevel,
}

pub struct ClassEvent {
    pub name: String,
}

pub struct ClassEnumeration {
    pub name: String,
}

pub struct ClassArgumentBlock {
    pub names: Vec<String>,
}

pub enum AccessLevel {
    Public,
    Private,
}
```

The exact field set can evolve, but the important point is that:

- classes should carry real semantic identity
- methods should be normal `HirFunction`s
- class members should be structured semantic items rather than opaque nested statement payloads

### Value Objects vs Handle Objects

The runtime already distinguishes:

- `Value::Object`
- `Value::HandleObject`

The next revision should decide whether the language type system also distinguishes these nominally.

Likely direction:

- all nominal class instances have a `ClassId`
- the type model may additionally distinguish handle semantics from value semantics

Conceptually:

- `Type::ClassInstance { class: ClassId }`
- `Type::HandleClassInstance { class: ClassId }`

or an equivalent representation in `TypeFact`

This is still open, but the important point is that handle/value object semantics should no longer be invisible to the language type system.

### Metaclass / Class Reference Semantics

The parser and current HIR already expose `MetaClass(String)`-style syntax.

That should mature into a nominal class-reference concept tied to `ClassId`, not remain just a stringly special case.

Conceptually:

- metaclass expressions should resolve to class identity
- static method/property lookup should work through the same class metadata used by the runtime

### Member / Property / Method Lookup

The next HIR and analysis model should make a distinction between:

- structural field access on structs
- nominal property access on class instances
- method lookup on class instances
- static property/method lookup on classes

#### Struct Member Lookup

For structs:

- `Member(base, field)` should use structural field knowledge
- field existence may be partial/unknown
- result typing may stay partially open

#### Class Property Lookup

For nominal classes:

- property lookup should resolve against class metadata
- inheritance should be respected
- access control can be validated semantically if desired
- return type should come from property metadata or fall back conservatively

#### Class Method Lookup

For nominal classes:

- method calls should resolve against class metadata
- inheritance should be respected
- static vs instance methods should be distinct
- the resolved method should point to a `FunctionId` or builtin/runtime declaration identity where possible

### Overloaded Indexing and Operators

The runtime already supports class-like overloaded behavior for:

- `subsref`
- `subsasgn`
- arithmetic/comparison operator methods such as `plus`, `times`, `mtimes`, etc.

The next semantic model should acknowledge that object behavior may be driven through method-based dispatch for:

- `.`
- `()`
- `{}`
- arithmetic operators
- comparison operators

This does not mean HIR should lower everything into raw method calls immediately, but it does mean:

- semantic lookup must be able to connect object syntax and operators to nominal class methods when applicable
- analysis and runtime should share that dispatch model

### Relationship to the Current Runtime Class Registry

The existing runtime class registry already contains useful concepts:

- class name
- parent class
- property definitions
- method definitions

That means the next revision should try to align the semantic class model with the runtime registry model rather than inventing a completely unrelated abstraction.

Long-term direction:

- runtime registry metadata and language-facing class metadata should converge
- HIR resolution should target those nominal class identities
- analysis should use the same method/property definitions for typing and effects

### Implication for `Type`

This class review suggests the following shape:

- keep `Type::Struct { ... }` for structural data
- stop treating runtime-domain nominal objects as bespoke enum cases long-term
- move toward nominal class-instance types keyed by `ClassId`

The current `DataDataset`, `DataArray`, and `DataTransaction` types should be seen as stepping stones toward that nominal class model, not as the permanent pattern for new runtime domains.

### Immediate Spec Adjustment

The HIR/type revision should now explicitly include:

- `ClassId`
- richer `HirClass`
- nominal class instance typing
- explicit member/method/property semantic lookup model

Without this, the next revision would still be under-specifying an area where the current runtime already has meaningful behavior.

### Runtime Classes and Rust-Defined Language Metadata

The next revision should go one step further than just "support classes better in HIR." It should set the system up so that Rust-defined runtime domain objects can be declared once and projected consistently into the language, tooling, and runtime.

This matters because types like:

- `DataDataset`
- `DataArray`
- `DataTransaction`

were introduced partly because the system did not yet have first-class nominal class/object semantics in HIR and typing.

The long-term direction should be:

- core language gets real nominal object/class types
- Rust runtime types can declare language-facing class metadata
- runtime registration, typing, LSP docs, bindings generation, and method/property signatures are all derived from one shared metadata source

### Design Goal

Rust runtime code should be able to define a typed runtime object once, and RunMat should automatically understand it as a first-class nominal language type.

That means one Rust-side declaration should feed:

- runtime class registration
- nominal type identity
- member/method lookup
- analysis and type summaries
- LSP/docs/help metadata
- TS or other bindings generation

This keeps domains clean and consistent and avoids duplicating type information in multiple systems.

### Language-Level Semantic Model

This implies the core language type model should eventually distinguish:

- structural structs
- nominal class instances
- handle-class instances if RunMat keeps handle/value object distinction
- metaclass-like type references

At a conceptual level, the language should move toward something like:

- `Type::Struct { ... }` for structural data
- `Type::ClassInstance { class: ClassId }` for nominal instances
- `Type::HandleClassInstance { class: ClassId }` if needed
- a metaclass/class-reference concept tied to `ClassId`

The exact final type names can change, but the important architectural distinction is:

- structs are structural
- classes/objects are nominal

### Rust-Side Declaration Model

The intended direction is that runtime object types declared in Rust should be able to participate declaratively.

Conceptually, something like:

```rust
#[runmat_class(name = "DataDataset", kind = "handle")]
struct DataDataset {
    // runtime implementation details
}

#[runmat_methods]
impl DataDataset {
    #[runmat_method]
    fn array(&self, name: String) -> DataArray { ... }

    #[runmat_method]
    fn begin(&self) -> DataTransaction { ... }
}
```

The exact macro syntax is not the point here. The architectural point is:

- Rust runtime classes should declare language-facing metadata declaratively
- that metadata should be generated once and reused everywhere

### Generated Metadata as the Source of Truth

The generated metadata layer should ideally include:

- class name
- nominal identity
- value-vs-handle semantics
- inheritance information
- properties
- methods
- method signatures
- documentation/help text
- analysis hooks or summary hooks where needed
- perhaps runtime traits such as mutability, persistence, or serialization support

This metadata should then feed:

- runtime class registry
- semantic call/member resolution
- LSP metadata generation
- bindings generation
- docs/help output
- analysis summaries

### Why This Is Better Than Special-Casing Domain Types

Today, some domain/runtime concepts appear directly in the generic type enum, such as:

- `Type::DataDataset`
- `Type::DataArray`
- `Type::DataTransaction`

That was a pragmatic intermediate step, but the long-term architecture should prefer:

- nominal class metadata plus domain-specific analysis hooks

rather than permanently growing the core generic type enum with every new runtime subsystem.

### Migration Direction for Current Dataset Types

The intended migration should be:

- `DataDataset`, `DataArray`, and `DataTransaction` become real nominal runtime classes
- builtin/method typing logic for them moves behind class metadata and summary hooks
- special cases in `infer_expr_type_with_env` are gradually replaced by class-aware summary lookup
- LSP/bindings/docs derive from the same generated class metadata

These types are strong first candidates for the new system because they already need:

- nominal identity
- method/property signatures
- rich return typing
- shape-aware behavior
- consistency across runtime, LSP, and bindings

### Relationship to Existing Builtins Metadata

The repository already has multiple projection surfaces for runtime/builtin metadata, including:

- `runmat/bindings/ts/src/builtins.ts`
- `runmat/crates/runmat-lsp/src/core/builtins_json.rs`

That is a signal that a more central metadata source would be valuable even beyond object/class typing.

The target direction should be:

- Rust-side declarations produce one shared language-facing metadata representation
- TS bindings, LSP docs, builtin docs, and runtime registration all consume that same representation

### Important Guardrail

The goal is not to expose arbitrary Rust implementation details directly as language types.

The correct model is:

- Rust declarations generate language-facing schema/metadata
- RunMat consumes that metadata as part of its own nominal type/class system

This keeps:

- language semantics stable
- runtime implementation flexible
- tooling and docs synchronized

without coupling the user language directly to internal Rust layout details.

### Implication for HIR and Type Design

This pushes the HIR/type design in a clearer direction:

- add `ClassId`
- treat classes as first-class semantic items
- keep `Struct` structural
- move runtime-domain object types toward nominal class identities
- allow analysis hooks to attach to nominal classes/methods rather than hard-coded string checks alone

### What This Leaves Open

This direction still leaves some design questions for later:

- exact macro/derive API surface
- exact runtime trait model for class registration
- how value-object vs handle-object semantics map into the type system
- how much method typing should be declarative metadata vs custom inference hooks
- whether a single metadata generator should also drive docs/help output and external bindings directly

But the architectural direction itself should now be considered part of the target design.

### Provisional Exact HIR Struct Sketch

The following is a provisional type sketch for the next HIR revision. The exact field names can still change, but this is the level of specificity we should now be designing against.

```rust
pub struct ModuleId(pub u32);
pub struct FunctionId(pub u32);
pub struct ClassId(pub u32);
pub struct EntrypointId(pub u32);
pub struct BindingId(pub u32);
pub struct ExprId(pub u32);
pub struct StmtId(pub u32);

pub struct QualifiedName {
    pub segments: Vec<String>,
}

pub struct HirAssembly {
    pub modules: Vec<HirModule>,
    pub functions: Vec<HirFunction>,
    pub classes: Vec<HirClass>,
    pub bindings: Vec<HirBinding>,
    pub entrypoints: Vec<HirEntrypoint>,
}

pub struct HirModule {
    pub id: ModuleId,
    pub name: QualifiedName,
    pub source_id: SourceId,
    pub imports: Vec<HirImport>,
    pub top_level_functions: Vec<FunctionId>,
    pub classes: Vec<ClassId>,
    pub synthetic_entry_function: Option<FunctionId>,
}

pub struct HirEntrypoint {
    pub id: EntrypointId,
    pub name: String,
    pub module: ModuleId,
    pub function: FunctionId,
    pub kind: EntrypointKind,
}

pub enum EntrypointKind {
    ScriptLike,
    Function,
    ReplCell,
    NotebookCell,
}

pub struct HirFunction {
    pub id: FunctionId,
    pub module: ModuleId,
    pub parent: Option<FunctionId>,
    pub enclosing_class: Option<ClassId>,
    pub name: FunctionName,
    pub kind: FunctionKind,
    pub params: Vec<BindingId>,
    pub outputs: Vec<BindingId>,
    pub locals: Vec<BindingId>,
    pub captures: Vec<CapturedBinding>,
    pub modifiers: FunctionModifiers,
    pub body: HirBlock,
    pub span: Span,
}

pub enum FunctionKind {
    Named,
    Anonymous,
    SyntheticEntrypoint,
    ClassMethod { is_static: bool },
}

pub enum FunctionName {
    Named(String),
    Anonymous,
}

pub struct FunctionModifiers {
    pub isolated: bool,
    pub has_varargin: bool,
    pub has_varargout: bool,
}

pub struct CapturedBinding {
    pub binding: BindingId,
    pub from_function: FunctionId,
}

pub struct HirBinding {
    pub id: BindingId,
    pub owner: BindingOwner,
    pub name: String,
    pub role: BindingRole,
    pub storage: BindingStorage,
    pub workspace_visibility: WorkspaceVisibility,
    pub declared_span: Span,
}

pub enum BindingOwner {
    Module(ModuleId),
    Function(FunctionId),
}

pub enum BindingRole {
    Parameter,
    Output,
    Local,
    ModuleBinding,
    ImplicitAns,
}

pub enum BindingStorage {
    Lexical,
    Global,
    Persistent,
}

pub enum WorkspaceVisibility {
    Hidden,
    TopLevel,
    ModuleVisible,
    ImplicitAns,
}

pub struct HirBlock {
    pub statements: Vec<HirStmt>,
}

pub struct HirStmt {
    pub id: StmtId,
    pub kind: HirStmtKind,
    pub span: Span,
}

pub enum HirStmtKind {
    ExprStmt { expr: HirExpr, suppressed: bool },
    Assign { target: HirPlace, value: HirExpr, suppressed: bool },
    MultiAssign { targets: Vec<Option<HirPlace>>, value: HirExpr, suppressed: bool },
    If {
        cond: HirExpr,
        then_block: HirBlock,
        elseif_blocks: Vec<(HirExpr, HirBlock)>,
        else_block: Option<HirBlock>,
    },
    While {
        cond: HirExpr,
        body: HirBlock,
    },
    For {
        binding: BindingId,
        iter: HirExpr,
        body: HirBlock,
    },
    Switch {
        expr: HirExpr,
        cases: Vec<(HirExpr, HirBlock)>,
        otherwise: Option<HirBlock>,
    },
    TryCatch {
        try_block: HirBlock,
        catch_binding: Option<BindingId>,
        catch_block: HirBlock,
    },
    GlobalDecl { bindings: Vec<BindingId> },
    PersistentDecl { bindings: Vec<BindingId> },
    Break,
    Continue,
    Return,
    Import(HirImport),
}

pub enum HirPlace {
    Binding(BindingId),
    Member(Box<HirExpr>, String),
    MemberDynamic(Box<HirExpr>, Box<HirExpr>),
    Index(Box<HirExpr>, Vec<HirExpr>),
    IndexCell(Box<HirExpr>, Vec<HirExpr>),
}

pub struct HirExpr {
    pub id: ExprId,
    pub kind: HirExprKind,
    pub span: Span,
}

pub enum HirExprKind {
    Number(String),
    String(String),
    Constant(String),
    Binding(BindingId),
    Unary(parser::UnOp, Box<HirExpr>),
    Binary(Box<HirExpr>, parser::BinOp, Box<HirExpr>),
    Tensor(Vec<Vec<HirExpr>>),
    Cell(Vec<Vec<HirExpr>>),
    Range(Box<HirExpr>, Option<Box<HirExpr>>, Box<HirExpr>),
    Colon,
    End,
    Index(Box<HirExpr>, Vec<HirExpr>),
    IndexCell(Box<HirExpr>, Vec<HirExpr>),
    Member(Box<HirExpr>, String),
    MemberDynamic(Box<HirExpr>, Box<HirExpr>),
    Call(HirCall),
    FuncHandle(HirCallableRef),
    MetaClass(ClassId),
    AnonymousFunction(FunctionId),
}

pub struct HirCall {
    pub callee: HirCallableRef,
    pub args: Vec<HirExpr>,
    pub syntax: CallSyntax,
}

pub enum CallSyntax {
    Plain,
    Method,
    DottedInvoke,
}

pub enum HirCallableRef {
    Function(FunctionId),
    ClassConstructor(ClassId),
    Builtin(QualifiedName),
    Imported(QualifiedName),
    DynamicExpr(Box<HirExpr>),
    Unresolved(QualifiedName),
}

pub struct HirImport {
    pub path: QualifiedName,
    pub wildcard: bool,
    pub alias: Option<String>,
    pub span: Span,
}

pub struct HirClass {
    pub id: ClassId,
    pub module: ModuleId,
    pub name: QualifiedName,
    pub super_class: Option<ClassId>,
    pub kind: ClassKind,
    pub properties: Vec<ClassProperty>,
    pub methods: Vec<ClassMethod>,
    pub events: Vec<ClassEvent>,
    pub enumerations: Vec<ClassEnumeration>,
    pub arguments: Vec<ClassArgumentBlock>,
    pub span: Span,
}

pub enum ClassKind {
    Value,
    Handle,
}

pub struct ClassProperty {
    pub name: String,
    pub modifiers: ClassPropertyModifiers,
}

pub struct ClassPropertyModifiers {
    pub is_static: bool,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub get_access: AccessLevel,
    pub set_access: AccessLevel,
}

pub struct ClassMethod {
    pub function: FunctionId,
    pub name: String,
    pub is_static: bool,
    pub access: AccessLevel,
}

pub struct ClassEvent {
    pub name: String,
}

pub struct ClassEnumeration {
    pub name: String,
}

pub struct ClassArgumentBlock {
    pub names: Vec<String>,
}

pub enum AccessLevel {
    Public,
    Private,
}
```

### Intent of the Provisional Sketch

This sketch intentionally makes a few decisions explicit:

- the top-level container is `HirAssembly`, not a flat `HirProgram`
- `HirAssembly` owns the canonical tables for functions, classes, and bindings
- executable things are represented uniformly as `HirFunction`
- `HirStmt::Function` no longer exists
- `HirStmt::ClassDef` no longer exists
- `BindingId` is semantic identity, not runtime slot layout
- calls carry semantic callee identity where possible
- anonymous functions become real functions referenced by `FunctionId`
- captures are recorded as relations, not as duplicate bindings
- workspace visibility is part of binding semantics rather than a post-hoc slot convention

Inventory invariants:

- `HirModule.top_level_functions` contains only functions declared directly at module scope
- `HirClass.methods` contains the method membership and metadata for a class
- nested functions are reached through `HirFunction.parent`, not duplicated in module or class inventories
- every function still points back to its owning `ModuleId`, so module-local indexing remains direct

### What Is Still Intentionally Open

This sketch is intentionally still leaving some room in a few places:

- exact field set for class/property/method metadata beyond the current sketch
- whether `QualifiedName` should intern segments or store strings directly
- whether declaration-order/source-item indexing needs a separate module-item table in addition to the canonical entity tables
- whether `WorkspaceVisibility` should be explicit on bindings or derived from owner plus entrypoint context
- exact runtime representation of captures and closure environments

### Immediate Design Consequences

If we adopt this shape, then several current types become transitional rather than foundational:

- `VarId`
- `HirProgram`
- `LoweringResult.variables: HashMap<String, usize>`
- `LoweringResult.functions: HashMap<String, HirStmt>`
- string-only `FuncCall(String, ...)`
- inline anonymous-function bodies in expressions

That is acceptable. The purpose of this revision is to replace those current flattening and reconstruction patterns with explicit semantic structure.

## Pressure-Test Outcomes

The proposed model was pressure-tested against several MATLAB-style patterns.

The numbered cases below are the canonical examples. The short sections that follow them are distilled takeaways from those same examples rather than a second, competing example list.

## Concrete Target MATLAB-Style Snippets

The architecture should stay anchored to recognizable MATLAB-style source examples. These are not just illustrative. They are target patterns the HIR, runtime, workspace model, and module system should be able to explain clearly.

### 1. Script-Like Top-Level Entry

```matlab
x = rand(10, 10);
y = x * 2;
disp(y);
```

Notes:

- This should lower to one module, one synthetic entry function, and one entrypoint pointing at that function.
- `x` and `y` should be top-level entrypoint-visible bindings.
- In interactive or desktop execution, `x` and `y` should appear in the workspace inspector after execution.
- `disp(y)` should not affect workspace visibility.

### 2. Script Plus Local Functions

```matlab
x = main(3);

function y = main(n)
    y = n + 1;
end
```

Notes:

- Top-level executable code should lower to a synthetic entry function.
- `main` should lower to a normal module-owned function, not a statement in the synthetic entry body.
- The call from top level should resolve to `FunctionId(main)`.
- `x` should be workspace-visible at the top level.
- `n` and `y` inside `main` should not appear in the workspace inspector.

### 3. Multiple Outputs and Variadics

```matlab
function [a, b, varargout] = f(x, y, varargin)
    a = x + y;
    b = x - y;
end
```

Notes:

- `f` should have explicit parameter bindings, output bindings, and function modifiers.
- `varargin` and `varargout` should be represented as ordinary bindings with special modifier-backed semantics.
- The HIR should not need a separate function model for multi-output or variadic functions.

### 4. True Nested Shared Lexical Capture

```matlab
function y = outer(a)
    acc = 0;

    function bump(x)
        acc = acc + x;
    end

    bump(a);
    y = acc;
end
```

Notes:

- This is the compatibility baseline for nested functions.
- `acc` should be one semantic binding owned by `outer`.
- `bump` should record that it captures `acc`.
- Mutation in `bump` should target the same logical binding seen by `outer`.
- `acc` should not appear twice in the workspace or debugger model just because it is captured.

### 5. `isolated` Nested Function

```matlab
function y = outer(a)
    z = a + 1;

    isolated function w = inner(x, z)
        w = x + z;
    end

    y = inner(a, z);
end
```

Notes:

- `inner` should still be represented as a normal function with a modifier, not a separate semantic category.
- `inner` should have no captures.
- Outer values needed by `inner` should be passed explicitly.
- If `inner` referenced outer `z` without taking it as a parameter, lowering should reject it because `isolated` forbids lexical capture.

### 6. Anonymous Function With Capture

```matlab
function f = outer(a)
    f = @(x) x + a;
end
```

Notes:

- Anonymous functions should lower to real `HirFunction`s.
- The expression should point at that anonymous function by `FunctionId`.
- The anonymous function should record that it captures `a`.
- This is a strong reason not to keep anonymous function bodies inline inside expression nodes.

### 7. Globals and Persistents

```matlab
function y = f(x)
    global G
    persistent cache

    if isempty(cache)
        cache = x;
    end

    y = cache + G;
end
```

Notes:

- `G` and `cache` should have binding storage metadata that distinguishes them from ordinary lexical locals.
- The `global` and `persistent` declarations should remain visible in the body for source fidelity and diagnostics.
- Storage semantics should not be inferred later from ad hoc statement walks.
- `cache` and `G` may participate in module or workspace-visible state depending on execution context, but that should be driven by binding/storage semantics rather than slot maps.

### 8. Imports and Resolved Calls

```matlab
import linalg.*
import signal.fft.fft2

y = fft2(x);
z = chol(A);
```

Notes:

- Imports should be explicit semantic items on the module.
- `fft2(x)` should resolve through import/module composition machinery rather than staying a stringly call name forever.
- `chol(A)` may resolve to a builtin or dependency-provided symbol depending on config and import state.
- This is why calls should carry semantic callee identity where possible.

### 9. Class Method

```matlab
classdef MyThing
    methods
        function y = twice(obj, x)
            y = x * 2;
        end
    end
end
```

Notes:

- Class methods should share the same `HirFunction` representation as ordinary functions.
- The class should own references to method functions rather than embedding a one-off executable representation.
- This keeps a single execution model for named functions, nested functions, anonymous functions, synthetic entry functions, and methods.

### 10. Workspace Visibility Intuition

```matlab
a = 1;
b = helper(a);

function y = helper(x)
    tmp = x + 1;
    y = tmp * 2;
end
```

Notes:

- After script-like execution, `a` and `b` should be workspace-visible.
- `x`, `tmp`, and `y` inside `helper` should not be workspace-visible.
- The workspace should reflect top-level semantic bindings, not every runtime slot touched during execution.

### 11. REPL / Notebook Cell Intuition

```matlab
t = linspace(0, 1, 100);
signal = sin(2 * pi * t);
```

Notes:

- In interactive execution, the synthetic entry function for the submitted code should export `t` and `signal` into the session workspace.
- Re-running later cells or snippets should update semantic top-level bindings, not append arbitrary frame-local state.
- The desktop variable inspector should remain stable and intuitive across repeated interactive executions.

### 12. Invalid `isolated` Capture

```matlab
function y = outer(a)
    z = a + 1;

    isolated function w = inner(x)
        w = x + z;
    end

    y = inner(a);
end
```

Notes:

- This should be rejected during lowering or early semantic analysis.
- `inner` is marked `isolated`, so it may not lexically capture `z`.
- The diagnostic should explain the fix clearly: pass `z` as a parameter instead of capturing it from the enclosing scope.
- The existence of `isolated` should not create a second closure model. This is simply an invalid use of an otherwise normal nested function.

### 13. No Local Leakage Into Workspace

```matlab
result = run_once(3);

function y = run_once(x)
    tmp = x + 1;
    y = tmp * 2;
end
```

Notes:

- After execution, `result` should be visible in the workspace if this ran as a script-like entrypoint or REPL submission.
- `x`, `tmp`, and `y` inside `run_once` should not appear in the workspace inspector.
- This should hold even if the runtime used additional temporary slots internally.
- Workspace state should be derived from semantic top-level bindings, not from frame contents or slot ranges.

### Script-Like Files

Top-level script code lowers cleanly into:

- one module
- one synthetic entry function
- one entrypoint pointing at that function

This strongly supports the "scripts disappear internally" direction.

### Script Plus Local Functions

A file with top-level executable code plus local functions also lowers cleanly:

- synthetic entry function for top-level code
- additional named functions as module-owned items
- calls resolve to `FunctionId`s

This supports removing `HirStmt::Function`.

### Multiple Outputs and Varargs

The function-centered model naturally supports:

- multiple outputs
- `varargin`
- `varargout`

through explicit parameter/output binding lists plus function modifiers.

### Nested Shared Lexical Capture

The model works for true MATLAB nested functions, but it exposed one important correction:

- captures should be relations, not duplicated bindings

This keeps lexical semantics honest and avoids confusing the semantic identity model with runtime closure layout.

### `isolated` Nested Functions

`isolated` works cleanly as a function modifier:

- same function representation
- no captures allowed
- required outer values must be passed explicitly

This reinforces the decision to keep `isolated` as the only refinement keyword.

### Anonymous Functions

Anonymous functions pressure-tested well once treated as real `HirFunction`s referenced from expressions by `FunctionId`.

This is cleaner than carrying inline anonymous function bodies inside expression nodes.

### Globals and Persistents

`global` and `persistent` declarations showed that bindings need explicit storage-class metadata in addition to their lexical role.

These declarations should remain visible in the body for source fidelity and diagnostics, but storage semantics should live on the binding model itself as well.

### Imports and Resolved Calls

The module/composition direction strongly benefits from semantically resolved calls in HIR rather than stringly typed call names.

### Class Methods

Class methods further reinforce the need for one uniform function representation shared across all executable forms.

## Workspace Model

The workspace should be defined semantically, not by raw variable arrays or "whatever binding survived lowering."

### User-Facing Rule

The variable inspector should show the workspace that a user expects:

- interactive top-level variables in a REPL-like session
- module-level or entrypoint-visible bindings that are meant to persist after execution
- `ans` when applicable

It should not show:

- ordinary function locals
- nested-function internal locals
- captured bindings as duplicate synthetic variables
- temporary runtime slots

### Conceptual Model

The system should distinguish:

- module bindings: bindings owned by a module and eligible to participate in workspace state
- entrypoint top-level bindings: bindings in a synthetic entry function that are marked for workspace export
- function locals: bindings owned by ordinary functions and not workspace-visible
- runtime slots: implementation-level layout derived from one of the above

This means the current session/workspace maps should eventually derive from explicit HIR binding metadata rather than plain `HashMap<String, usize>` slot maps.

### Workspace Export Intuition

For script-like entrypoints, the intuitive behavior is:

- top-level assigned names become workspace-visible
- helper-function locals do not
- nested-function captures only affect workspace indirectly by mutating workspace-visible or module-visible bindings

For ordinary library functions:

- local bindings are not workspace-visible
- results flow through return values or explicit module/global/persistent storage

For REPL or notebook-like execution:

- the synthetic entry function for the submitted code should export assigned top-level bindings into the session workspace

### Binding Metadata Implication

Bindings likely need a workspace classification in addition to role and storage. This can be represented directly on bindings or derived from owner plus entrypoint context, but it should be explicit enough that the desktop UI and runtime session state follow one principled rule.

Useful conceptual categories:

- hidden lexical local
- workspace-visible top-level binding
- module binding
- implicit `ans`

### Runtime Relationship

Runtime variable arrays may still remain dense and slot-based for performance, but:

- workspace export/import should map through binding identity
- slot numbering should not define workspace semantics
- variable inspector output should be built from semantic binding sets, not raw frame contents

### Provisional Exact Workspace / Export Contract

For the next major revision, the workspace model should be treated as an explicit semantic contract rather than an implementation side effect.

The intended contract is:

- workspace state is a mapping from semantic binding identity to current user-visible value
- only bindings classified as workspace-visible may enter that mapping
- ordinary function locals never enter that mapping
- runtime slots are an implementation detail and may not define visibility

### Workspace Visibility Classes

The working visibility classes should be:

- `Hidden` — never appears in the workspace inspector
- `TopLevel` — may appear in workspace after entrypoint or interactive execution
- `ModuleVisible` — belongs to module-level state and may appear in workspace depending on host/view mode
- `ImplicitAns` — special implicit result binding shown as `ans`

These may eventually remain explicit on `HirBinding` or be partially derived, but the semantic categories themselves should be treated as part of the contract.

### Binding Categories and Workspace Eligibility

#### Ordinary Function Locals

- parameters are `Hidden`
- outputs are `Hidden`
- ordinary locals are `Hidden`
- nested-function locals are `Hidden`
- anonymous-function locals are `Hidden`

These bindings may appear in debugger or tracing tools, but not in the ordinary workspace inspector.

#### Entrypoint Top-Level Bindings

Bindings created in a synthetic entry function should default to `TopLevel` visibility unless a later language feature explicitly marks otherwise.

This is what preserves intuitive script and REPL behavior.

#### Module Bindings

Bindings owned directly by a module should be `ModuleVisible`.

Examples:

- module-level state
- exported package data in the future
- perhaps module-scoped caches or constants where applicable

These are semantically different from ordinary entrypoint locals, even if a host UI chooses to show both in a combined inspector.

#### `ans`

`ans` should be represented as an `ImplicitAns` binding.

It should:

- update when an expression result is surfaced implicitly
- remain distinct from ordinary top-level locals
- appear in the workspace inspector when relevant

#### Captured Bindings

Captured bindings must not create duplicate workspace variables.

If a captured binding is already workspace-visible because it is top-level or module-visible, the workspace should simply reflect the current value of that same binding.

### Exact Export Rules by Execution Mode

#### Script-Like Entrypoint Execution

When a declared script-like entrypoint runs:

- all `TopLevel` bindings assigned during execution become candidates for workspace export
- bindings that remain unassigned at the end do not appear
- hidden locals remain hidden
- helper-function locals remain hidden
- captured bindings do not duplicate
- `ans` appears if implicit-result rules produce it

The exported workspace after the run should represent the final values of the entrypoint-visible bindings, not a trace of all intermediate slots touched during execution.

#### Function-Oriented Entrypoint Execution

When an entrypoint resolves directly to a function-oriented root:

- the function’s internal parameters, outputs, and locals remain `Hidden`
- workspace changes happen only through:
  - explicit module-visible bindings
  - globals
  - persistents
  - `ans`, if the host chooses to surface a top-level return value as `ans`

Function-oriented entrypoints should not implicitly dump their internal local scope into the workspace.

#### REPL / Notebook Cell Execution

Interactive execution should behave like synthetic script-like entrypoint execution:

- submitted code lowers to a synthetic entry function
- assigned `TopLevel` bindings from that synthetic entry are exported to the session workspace
- rerunning later cells or snippets updates the same semantic workspace mapping
- hidden locals and internal helper scopes remain hidden

This keeps interactive behavior intuitive while still using the same semantic model as declared entrypoints.

#### Library Function Calls

Ordinary library function invocation should not directly export locals to workspace.

Library functions affect observable state only through:

- explicit return values
- mutations to already-visible captured/module/global/persistent state
- host-driven top-level assignment of returned values

### Globals and Persistents

Globals and persistents should be part of the same semantic workspace story, but with distinct storage semantics.

#### Globals

Global bindings should:

- have `BindingStorage::Global`
- resolve through the program’s global storage model
- appear in workspace views when the host chooses to expose global state
- not rely on ad hoc slot naming for visibility

#### Persistents

Persistent bindings should:

- have `BindingStorage::Persistent`
- survive function returns according to persistent semantics
- remain semantically distinct from ordinary top-level workspace bindings
- be exposable in workspace or debugging views according to host policy, but through binding/storage semantics rather than a separate accidental mechanism

### Removals and Clearing

The workspace model should define removals semantically too.

At minimum, the runtime/session layer should support the concept that a workspace-visible binding can:

- be assigned a new value
- remain unchanged
- be removed from visible workspace state

Removal should be driven by semantic operations such as:

- explicit `clear`
- host reset or workspace import/export replacement
- rerunning an entrypoint in a mode that replaces the previous workspace snapshot

The workspace should not remove bindings merely because an internal runtime slot disappeared or a local frame was torn down.

### Replacement vs Incremental Update Modes

Hosts may want two distinct workspace update modes:

- `ReplaceVisibleSet` — after a script-like run, the visible top-level workspace becomes exactly the exported visible set for that run plus retained module/global/persistent state according to policy
- `MergeVisibleSet` — interactive execution merges newly exported bindings into the existing semantic workspace

The host/session policy can choose which mode to use, but the semantic export set should be the same in both cases.

This is especially useful for distinguishing:

- one-shot script execution
- long-lived REPL or notebook sessions

### Host and UI Contract

The desktop variable inspector should consume semantic workspace state, not raw runtime arrays.

The host-facing workspace API should therefore expose something conceptually like:

- semantic binding identity
- display name
- visibility classification
- storage classification
- current value
- provenance such as module or entrypoint owner if useful

That allows the host to present:

- a simple variable inspector
- richer debug views
- filtered views for module/global/persistent state

without needing separate ad hoc state models.

### Relationship to Current Implementation

This exact contract is intentionally stricter than the current implementation, which is still based on:

- `variable_array`
- `variable_names`
- `workspace_values`
- VM thread-local assigned-name sets

The migration goal is:

- preserve intuitive current user behavior
- remove slot-based accidental semantics
- make workspace export an explicit consequence of semantic binding classification

### Immediate Runtime Consequence

If we adopt this contract, the runtime/session layer will need an explicit semantic export step after execution:

- determine the exported binding set from entrypoint/module/global/persistent semantics
- map those bindings to current values
- update the host workspace state from that semantic export set

That export step should replace the current "whatever assigned slots were tracked" model over time.

## Static Analysis, Typing, and Shape Propagation

The target analysis model should be built on semantic HIR identities rather than VM slot indices or ad hoc lowering maps.

Core principle:

- HIR gives stable semantic identities
- analysis computes facts over those identities
- runtime, linting, LSP, fusion, and workspace tooling consume the same fact graph

This should replace the current mix of:

- `Vec<Type>` indexed by `VarId`
- ad hoc `HashMap<VarId, Type>`
- function-return maps keyed by string names

with structured analysis stores keyed by:

- `BindingId`
- `ExprId`
- `FunctionId`
- `ModuleId`

### Fact Layers

The analysis model should separate fact categories cleanly.

#### Binding Facts

Facts about semantic bindings:

- declared role
- storage class
- current inferred type
- current inferred shape/tensor facts
- capture and mutability metadata
- workspace visibility

These answer questions like:

- what do we know about binding `A` at this program point?
- is `cache` persistent?
- is this binding scalar, logical, tensor-like, or unknown?

#### Expression Facts

Facts about expressions:

- inferred type
- inferred shape
- literal/value category
- call effect and fusibility hints
- materialization or aliasing hints if needed later

These answer:

- what is the result of `A * B`?
- is `x + 1` scalar, vector, matrix, tensor, or unknown?
- is this call builtin-resolved and fusible?

#### Function Summaries

Facts about functions as callable units:

- parameter constraints
- output summaries
- effect summaries
- capture summaries
- purity or side-effect classification
- fusibility and dispatch characteristics
- shape transfer summaries

These answer:

- if I call this function, what do I know about outputs?
- does it mutate captured, module, global, or persistent state?
- can it participate in fusion?
- is it safe for parallel execution?

#### Module Summaries

Facts about modules/packages:

- exported functions and symbols
- imported symbols
- module-level bindings
- dependency-resolved symbol tables
- module/global/persistent interactions

These become important once fuller module composition lands.

### Consistency Rule

The most important invariant is:

- analysis must be defined over semantic ownership, not runtime layout

This means:

- analysis never depends on VM slot numbering
- analysis never treats captures as copied bindings
- analysis never depends on stringly typed call identities once resolution has happened
- module and import resolution should happen before, or as part of, semantic call analysis

### Analysis Pass Structure

The analysis pipeline should be staged.

#### Pass 1: Resolution and Structural Indexing

Build the semantic index:

- binding ownership tables
- function ownership and nesting
- capture relations
- resolved call identities
- module/import resolution state
- block and statement traversal scaffolding

This pass should not do deep inference. Its purpose is to establish semantic structure.

#### Pass 2: Local Flow Analysis

Within a function or entrypoint body, propagate facts through statements and expressions.

This should track environments like:

- `BindingId -> TypeFact`
- `BindingId -> ShapeFact`

through control flow.

Assignments update binding facts. Expressions compute expression facts from operand facts. Branches and loops merge environments through lattice joins rather than ad hoc overwrite behavior.

This is the intended replacement for the currently duplicated env/join logic spread across globals, function outputs, and function variable inference.

#### Pass 3: Interprocedural Summary Propagation

Once local flow facts exist, compute interprocedural summaries:

- function output summaries
- effect summaries
- capture read/write summaries
- resolved call edges
- builtin summary application
- fixpoint iteration for recursive or mutually dependent call graphs when needed

This gives cross-function and eventually cross-module consistency.

### Type and Shape Lattices

Type and shape propagation should use lattice-based joins and conservative refinement rather than plain overwrite behavior.

#### Type Lattice

At minimum, the type lattice likely needs:

- unreachable / bottom
- unknown
- scalar numeric families
- logical
- string
- tensor and array categories
- struct/object/class categories
- unions or widening categories where needed

Join should be monotonic and conservative.

#### Shape Lattice

The shape lattice should support:

- unknown rank/shape
- known scalar
- known vector
- known matrix
- known tensor rank
- partially known dimensions
- fully known dimensions
- symbolic or parametric dimensions later if useful

Examples of useful shape facts:

- `10x20`
- `Nx20`
- rank-2 unknown dims
- scalar
- rank-known but dims-unknown

Operations like broadcasting, matrix multiply, transpose, slicing, indexing, and reduction should all operate on this lattice.

### Safe Propagation Rule

Shape and type facts should only become more precise when justified. Joins must never invent certainty that is not supported across all paths.

Examples:

- if one branch says `10x20` and another says unknown, the joined result is not `10x20`
- if one branch says `10x20` and another says `10xN`, the joined result should widen conservatively
- if one branch says scalar and another says vector, the joined result should be a safe common super-fact rather than a guessed exact shape

This is critical for safe linting, correct optimization, and trustworthy LSP hints.

### Captures and Nested Functions

Nested shared lexical capture should be analyzed using the same `BindingId` identity as the parent binding.

Implications:

- captured bindings are still the same semantic binding
- nested-function environments may reference outer-owned bindings
- function summaries must record capture reads and writes
- if a nested function mutates a binding, parent or caller-visible summaries must account for that mutation

This is another reason captures should be modeled as relations rather than duplicated bindings.

For `isolated` functions:

- no captures are permitted
- summaries become simpler
- optimization and effect reasoning become easier

### Globals, Persistents, and Workspace

Lexical locals, module/workspace bindings, globals, and persistents should all live in the same semantic fact system, but with different storage classes and visibility semantics.

This should allow analysis to express facts such as:

- this function writes persistent `cache`
- this entrypoint exports binding `x` to the workspace
- this nested function mutates captured binding `acc`
- this local `tmp` is dead after return and never workspace-visible

The workspace model, LSP model, and runtime model should all consume the same semantic classification.

### Calls, Builtins, and Imports

Strong typing and shape analysis depend on semantically resolved calls.

Call analysis should behave like this:

- if the callee is builtin or intrinsic, use curated builtin summaries
- if the callee is a user function in the same semantic world, use its function summary
- if the callee is imported, resolve it to a canonical symbol first and then apply its summary
- if the callee is unresolved or truly dynamic, degrade conservatively

This is why the future HIR should move away from string-only `FuncCall(String, ...)`.

### Fusion and Parallel Safety

Fusion and dispatch planning should consume analysis summaries rather than invent a separate semantic model.

Useful derived classifications include:

- pure vs impure
- captures mutable state vs does not
- touches global/persistent/module state vs does not
- shape-stable vs shape-unknown
- fusible vs non-fusible
- parallel-safe vs requires serialization

This keeps optimization behavior consistent with type and effect analysis.

### Proposed Analysis Stores

Implementation details can vary, but conceptually we likely want:

- `SemanticIndex`
- `TypeFacts`
- `ShapeFacts`
- `EffectFacts`
- `ExecutionFacts`

Whether these are separate stores or one composite `AnalysisStore` is an implementation detail. The key point is that they are keyed by stable semantic IDs and can be shared across runtime, linting, LSP, and optimization.

### Provisional Exact Analysis Fact Model

For the next major revision, the analysis system should be designed against a concrete fact model rather than only informal categories.

The following is a provisional working sketch.

```rust
pub struct AnalysisStore {
    pub semantic_index: SemanticIndex,
    pub binding_facts: HashMap<BindingId, BindingFact>,
    pub expr_facts: HashMap<ExprId, ExprFact>,
    pub function_summaries: HashMap<FunctionId, FunctionSummary>,
    pub module_summaries: HashMap<ModuleId, ModuleSummary>,
}

pub struct SemanticIndex {
    pub function_owners: HashMap<FunctionId, ModuleId>,
    pub binding_owners: HashMap<BindingId, BindingOwner>,
    pub captures: HashMap<FunctionId, Vec<CapturedBinding>>,
    pub resolved_calls: HashMap<ExprId, HirCallableRef>,
}

pub struct BindingFact {
    pub role: BindingRole,
    pub storage: BindingStorage,
    pub workspace_visibility: WorkspaceVisibility,
    pub ty: TypeFact,
    pub shape: ShapeFact,
    pub init: InitFact,
}

pub struct ExprFact {
    pub ty: TypeFact,
    pub shape: ShapeFact,
    pub effects: EffectSummary,
    pub fusibility: FusibilityFact,
    pub parallel_safety: ParallelSafetyFact,
}

pub struct FunctionSummary {
    pub params: Vec<ParameterSummary>,
    pub outputs: Vec<OutputSummary>,
    pub effects: EffectSummary,
    pub call_behavior: CallBehaviorFact,
}

pub struct ModuleSummary {
    pub exported_functions: Vec<FunctionId>,
    pub module_bindings: Vec<BindingId>,
    pub effects: EffectSummary,
}

pub struct ParameterSummary {
    pub ty: TypeFact,
    pub shape: ShapeFact,
}

pub struct OutputSummary {
    pub ty: TypeFact,
    pub shape: ShapeFact,
}
```

Binding ownership, role, storage, and workspace visibility are convenient to expose on `BindingFact`, but capture relations should continue to live in `SemanticIndex.captures` rather than being duplicated onto every binding record.

### Provisional Flow Environment

Inside a function or entrypoint body, local flow should operate on an explicit environment:

```rust
pub struct FlowEnv {
    pub bindings: HashMap<BindingId, BindingFlowFact>,
}

pub struct BindingFlowFact {
    pub ty: TypeFact,
    pub shape: ShapeFact,
    pub init: InitFact,
}
```

This is the in-function dataflow state that gets joined across branches and widened across loops. The post-analysis `binding_facts` and `expr_facts` in `AnalysisStore` should be derived from these flow results.

### Provisional `TypeFact`

The least disruptive first step is to keep `runmat_builtins::Type` as the underlying type domain and wrap it in a flow-friendly fact type, while expanding `Type` itself during this revision to carry nominal class cases.

```rust
pub enum TypeFact {
    Unreachable,
    Known(Type),
}
```

Important notes:

- `Known(Type::Unknown)` remains valid and means reachable but unknown
- `Unreachable` is distinct and represents dead control-flow
- `Type` must grow nominal class-oriented cases such as `ClassInstance { class: ClassId, ... }` and `ClassRef(ClassId)` so that the class/object design and the analysis design stay aligned
- `Type::unify` remains the core compatibility path for type joins in the first migration

This avoids inventing a completely separate type universe during the same revision that is already replacing HIR and analysis structure.

### Provisional `ShapeFact`

Shape facts should be first-class and distinct from `Type`, even if some current `Type` variants still embed shape data during the migration.

```rust
pub enum ShapeFact {
    Unreachable,
    Unknown,
    Scalar,
    Ranked { rank: usize },
    Shaped { dims: Vec<DimFact> },
}

pub enum DimFact {
    Known(usize),
    Symbolic(String),
    Unknown,
}
```

Working interpretation:

- `Scalar` means scalar semantics
- `Ranked { rank }` means rank known, dimensions not fully known
- `Shaped { dims }` means dimension-level information is known at least partially
- `Symbolic` allows future relationships like `N x M` without forcing fully concrete sizes

### Provisional `InitFact`

Initialization state should be explicit in flow analysis.

```rust
pub enum InitFact {
    Unassigned,
    MaybeAssigned,
    DefinitelyAssigned,
}
```

This is useful for:

- definite-assignment diagnostics
- workspace export filtering
- safer propagation through branches

### Provisional `EffectSummary`

The first effect system does not need to be fully general. It should, however, make hidden mutation and optimization barriers explicit.

```rust
pub struct EffectSummary {
    pub reads_captures: BTreeSet<BindingId>,
    pub writes_captures: BTreeSet<BindingId>,
    pub reads_globals: BTreeSet<BindingId>,
    pub writes_globals: BTreeSet<BindingId>,
    pub reads_persistents: BTreeSet<BindingId>,
    pub writes_persistents: BTreeSet<BindingId>,
    pub reads_modules: BTreeSet<BindingId>,
    pub writes_modules: BTreeSet<BindingId>,
    pub may_call_unknown: bool,
    pub may_allocate: bool,
}
```

This is enough to support:

- capture mutation tracking
- global/persistent/module mutation summaries
- conservative barriers for fusion and parallel execution

### Provisional Execution-Related Facts

Two compact classifications should be enough initially:

```rust
pub enum FusibilityFact {
    Unknown,
    Fusible,
    NonFusible(FuseBlocker),
}

pub enum ParallelSafetyFact {
    Unknown,
    Safe,
    ReadsSharedState,
    WritesSharedState,
}

pub enum FuseBlocker {
    UnknownCall,
    SharedStateWrite,
    UnsupportedBuiltin,
    ShapeInstability,
}

pub enum CallBehaviorFact {
    Unknown,
    Pure,
    ReadsSharedState,
    WritesSharedState,
}
```

These can be refined later, but they give the optimizer/runtime a consistent vocabulary immediately.

### Join Rules

The analysis system should standardize joins rather than letting each pass define its own merge behavior.

#### `TypeFact::join`

Working rule:

- `Unreachable ⊔ x = x`
- `Known(a) ⊔ Known(b) = Known(a.unify(&b))`

This keeps the current `Type::unify` logic useful while making reachability explicit.

#### `ShapeFact::join`

Working rule:

- `Unreachable ⊔ x = x`
- `Unknown ⊔ x = Unknown`
- `Scalar ⊔ Scalar = Scalar`
- `Shaped(dims_a) ⊔ Shaped(dims_b)`:
  - if ranks differ, result becomes `Unknown`
  - else join dimensions pointwise
- `Shaped(dims) ⊔ Ranked { rank }`:
  - if `dims.len() == rank`, result becomes `Ranked { rank }`
  - else `Unknown`
- `Ranked { a } ⊔ Ranked { b }`:
  - if `a == b`, keep `Ranked { a }`
  - else `Unknown`

Dimension join:

- `Known(a) ⊔ Known(b)`:
  - `Known(a)` if `a == b`
  - `Unknown` otherwise
- `Symbolic(a) ⊔ Symbolic(b)`:
  - `Symbolic(a)` if equal
  - `Unknown` otherwise
- any join with `Unknown` yields `Unknown`

#### `InitFact::join`

Working rule:

- `DefinitelyAssigned ⊔ DefinitelyAssigned = DefinitelyAssigned`
- `Unassigned ⊔ Unassigned = Unassigned`
- all mixed cases become `MaybeAssigned`

#### `EffectSummary::join`

Working rule:

- set fields union
- boolean flags OR together

This is intentionally conservative.

### Loop Widening

Loops should use explicit widening rather than unlimited refinement.

Working first-step rule:

- iterate a loop body to a small fixed point budget
- if shapes keep becoming more complex, widen to `Ranked { rank }` or `Unknown`
- if types continue oscillating, widen via `Type::unify`
- preserve `Unreachable` only if the loop body truly never reaches a join point

This gives predictable behavior without requiring a full abstract interpretation framework on day one.

### Relationship to Current `Type`

The current `runmat_builtins::Type` already includes optional shape-bearing variants such as `Tensor { shape }` and `Logical { shape }`, but it still treats runtime objects mostly as `Unknown` and still carries transitional `Data*` cases.

For the migration:

- keep using `Type` as the underlying type universe
- extend `Type` so nominal classes and metaclass references stop degrading to `Unknown`
- introduce `ShapeFact` as the canonical analysis-time shape domain
- gradually move optimizer/LSP/lint consumers toward `ShapeFact`
- treat embedded type-shape data in `Type` as compatibility scaffolding rather than the long-term analysis boundary

This should reduce migration risk while still letting the new analysis architecture land cleanly.

### Why This Is a Better Fit

This model gives us:

- exact key spaces (`BindingId`, `ExprId`, `FunctionId`, `ModuleId`)
- explicit reachability and initialization
- one join model
- one effect vocabulary
- a shape domain richer than current `Option<Vec<Option<usize>>>` without forcing full symbolic algebra immediately
- a path for LSP, linting, runtime, and fusion to consume the same facts

### Still Intentionally Open

The following still need later refinement:

- whether `TypeFact` should eventually become richer than `Known(Type)` vs `Unreachable`
- whether `DimFact::Symbolic(String)` should instead use interned dimension symbols
- whether `ModuleSummary` needs explicit export metadata vs only function/binding lists
- whether `CallBehaviorFact` and `ParallelSafetyFact` should stay separate long term or merge into a richer effect domain

### Main Refactor Guidance

When migrating the current inference code, the goal should not be to simply port existing maps to new IDs.

The goal should be to use the new HIR to build:

- one environment model
- one join model
- one summary model
- one resolved call model

That is a major part of the payoff of the refactor.

## Current State vs Target State

The major revision should be grounded in the actual current implementation, not just the desired end state. The following sections map the current system to the target architecture for the four key workstreams still in flight.

### 1. Core HIR and Identity Model

#### Current State

Today the HIR is centered on `VarId(pub usize)` in `runmat/crates/runmat-hir/src/ids.rs`.

That ID currently does too much at once:

- semantic identity for a variable-like entity
- index into `var_types`
- bridge into session/workspace slot maps
- input to VM remapping

Current main HIR definitions live in `runmat/crates/runmat-hir/src/hir.rs`:

- `HirProgram { body, var_types }`
- `HirExpr { kind, ty, span }`
- `HirStmt`, including `HirStmt::Function`
- `LoweringResult` with many side tables

Current semantic limitations:

- functions are statements
- scopes exist in lowering context, not as durable HIR structure
- anonymous functions are inline expression nodes
- captures are implicit
- downstream crates reconstruct local/function structure by walking bodies and remapping `VarId`s

Relevant current files and symbols:

- `runmat/crates/runmat-hir/src/ids.rs` — `VarId`, `SourceId`
- `runmat/crates/runmat-hir/src/hir.rs` — `HirProgram`, `HirStmt`, `HirExpr`, `LoweringResult`
- `runmat/crates/runmat-hir/src/lowering/ctx.rs` — lowering `Scope`, `Ctx`
- `runmat/crates/runmat-hir/src/remapping.rs` — function-local slot remapping
- `runmat/crates/runmat-hir/src/inference/shared.rs` — function reconstruction helpers like `collect_function_defs`
- `runmat/crates/runmat-core/src/session/compile.rs` — session/workspace bridge still built around names and slots

#### Target State

The target model should move to explicit semantic items:

- `HirAssembly`
- `HirModule`
- `HirEntrypoint`
- `HirFunction`
- `HirClass`
- stable semantic IDs such as `ModuleId`, `FunctionId`, `ClassId`, `BindingId`, `ExprId`, and `StmtId`

Key target changes:

- functions stop being statements
- bindings stop being VM slots
- captures become explicit relations
- anonymous functions become real functions
- slot layout becomes derived runtime metadata rather than semantic identity

#### Main Delta

We are moving from a flat, reconstructive HIR to an explicit, semantic, ownership-aware HIR.

Important migration hazard:

- current function identity is still spread across string-keyed maps such as `LoweringResult.functions`, `collect_function_defs`, and VM bytecode function tables, so name-collision bugs need to be treated as a first-class migration concern rather than cleanup deferred until the end

### 2. Config, Imports, Modules, and Entrypoints

#### Current State

The existing config system is runtime/tooling config, not project/package composition config.

Current config lives primarily in:

- `runmat/crates/runmat-config/src/lib.rs` — `RunMatConfig`, `ConfigLoader`
- `runmat/docs/CONFIG.md` — current precedence and discovery behavior
- `runmat/crates/runmat-cli/src/app/bootstrap.rs` — CLI bootstrap

What exists today:

- config discovery and environment overrides
- runtime/language/telemetry/jit/gc/etc. settings
- file-path driven CLI execution
- REPL / wasm / kernel-specific source naming
- MATLAB-style path search in runtime builtins

What does not exist yet:

- `[sources]`
- `[dependencies]`
- `[[entrypoints]]`

Current import and source discovery behavior is split across:

- `runmat/crates/runmat-hir/src/validation/imports.rs`
- `runmat/crates/runmat-vm/src/compiler/imports.rs`
- `runmat/crates/runmat-runtime/src/builtins/common/path_search.rs`
- `runmat/crates/runmat-runtime/src/builtins/common/path_state.rs`

Important current property:

- `import` helps source-level resolution, especially for builtins and already-known functions
- it is not a real package/dependency mechanism

Execution roots today are mostly determined by:

- a CLI script path
- REPL submission
- notebook/kernel cell execution
- wasm bootstrap

#### Target State

The target composition model should be config-driven.

Config concepts:

- `[sources]` for local project discovery
- `[dependencies]` for external package availability
- `[[entrypoints]]` for named execution roots

Target semantic split:

- local project code comes from configured source roots
- external composition comes from dependencies
- `import` remains a source-level visibility/resolution convenience
- scripts remain as source style but lower to synthetic entry functions internally

Canonical internal symbol identity should remain qualified even when source uses unqualified imported names.

#### Main Delta

We are moving from a host/path-centric execution model to an explicit project/package/entrypoint model.

Important migration hazard:

- import and source resolution are currently split across HIR validation, VM compiler import heuristics, and runtime path search, so composition should be migrated in phases rather than assuming one immediate cut-over point

### 3. Workspace and Export Semantics

#### Current State

Workspace state today is largely operational and slot-driven.

Primary current implementation points:

- `runmat/crates/runmat-core/src/session/mod.rs` — `RunMatSession`
- `runmat/crates/runmat-core/src/session/run.rs` — post-execution workspace reconciliation
- `runmat/crates/runmat-core/src/session/workspace.rs` — workspace snapshot/export/import
- `runmat/crates/runmat-vm/src/runtime/workspace.rs` — TLS workspace state
- `runmat/crates/runmat-vm/src/ops/stack.rs` — store/load behavior
- `runmat/crates/runmat-vm/src/runtime/globals.rs` — globals and persistents

Current key structures:

- `workspace_values: HashMap<String, Value>`
- `variable_array: Vec<Value>`
- `variable_names: HashMap<String, usize>`
- VM thread-local workspace tracking and assigned-name sets

Current behavior:

- top-level `StoreVar` operations feed workspace tracking
- locals generally do not
- `ans` is special-cased
- globals and persistents live in separate TLS maps
- JIT and interpreter paths are not perfectly uniform in how workspace state is refreshed

#### Target State

The workspace should become semantic rather than slot-driven.

The variable inspector should show:

- module/workspace-visible bindings
- top-level entrypoint-visible bindings
- REPL/notebook top-level semantic bindings
- `ans` when applicable

It should not show:

- ordinary function locals
- captured bindings as duplicate variables
- runtime temporary slots

Target rule:

- workspace export/import should map through semantic binding identity
- slot numbering should not define workspace semantics

#### Main Delta

We are moving from an operational slot/TLS workspace model to a semantic top-level binding/export model.

Important migration hazard:

- the workspace cannot become fully semantic until binding identity flows through `runmat-core` session compilation and VM layout derivation instead of stopping at `VarId` plus host-side name maps

### 4. Static Analysis, Typing, and Shape Propagation

#### Current State

Current analysis is useful but fragmented across several representations and key spaces.

Important current locations:

- `runmat/crates/runmat-builtins/src/lib.rs` — `Type`, `Type::unify`
- `runmat/crates/runmat-hir/src/hir.rs` — `HirExpr.ty`, `HirProgram.var_types`, `LoweringResult`
- `runmat/crates/runmat-hir/src/inference/` — global/function/expr inference
- `runmat/crates/runmat-static-analysis/src/lints/` — lint consumers
- `runmat/crates/runmat-lsp/src/core/analysis.rs` — semantic model construction
- `runmat/crates/runmat-vm/src/bytecode/compile.rs` and `runmat/crates/runmat-vm/src/accel/graph.rs` — VM/fusion consumers

Current fact keys and carriers include:

- `HirExpr.ty`
- `HirProgram.var_types: Vec<Type>`
- `LoweringResult.var_types`
- `HashMap<VarId, Type>`
- `HashMap<String, Vec<Type>>` for function returns
- `HashMap<String, HashMap<VarId, Type>>` for inferred function environments

Current propagation shape:

- infer function outputs
- infer function variable environments
- infer globals
- re-walk statements and expressions in multiple related passes

Current structural problems:

- duplicated join/env logic
- function summaries keyed by strings
- facts split between HIR, lowering result, LSP/static-analysis, and VM/fusion
- VM/fusion consume `HirProgram.var_types`, while LSP/lints consume richer lowering-result maps

#### Target State

The target analysis model should be keyed by stable semantic IDs and shared across consumers.

Target conceptual stores:

- `SemanticIndex`
- `TypeFacts`
- `ShapeFacts`
- `EffectFacts`
- `ExecutionFacts`

Target properties:

- one environment model
- one join model
- one summary model
- one resolved-call model
- runtime, LSP, linting, workspace, and fusion all consume the same semantic fact graph

#### Main Delta

We are moving from fragmented slot/string-keyed inference to a semantic-ID keyed fact-store model shared across the system.

Important migration hazard:

- current inference still reconstructs callable summaries from statement walks and string names, so `FunctionId`-keyed summaries should replace those reconstruction passes early in the migration rather than after the fact-store shape already lands

### 5. Cross-Cutting Migration Hazards

The current implementation has several cross-cutting patterns that the migration plan needs to account for explicitly.

- function identity is still duplicated across `String` keys in lowering, inference, session state, and bytecode tables
- import resolution is currently split across HIR, VM compiler logic, and runtime path search
- class/static resolution still mixes parser-shaped classdef lowering, dotted builtin-name heuristics, and the runtime class registry
- object values still frequently degrade to `Type::Unknown`, which means the type migration and class migration have to move together

That means the transition should not be treated as "new HIR first, then everything else eventually." The HIR migration needs a deliberately staged bridge for classes, imports, session/workspace export, and function-summary ownership.

## Recommended Order of Work

1. Lock down HIR, runtime, module, and class invariants, including the struct-vs-class split and the ownership rules for functions, classes, bindings, and entrypoints.
2. Freeze the new HIR core types and IDs: `HirAssembly`, `HirModule`, `HirFunction`, `HirClass`, `HirBinding`, and the semantic ID set.
3. Specify capture semantics precisely, including nested shared lexical capture, closure environment requirements, and the exact meaning of `isolated`.
4. Define config concepts for `[sources]`, `[dependencies]`, and `[[entrypoints]]`.
5. Define the phase-1 resolved import model so there is an explicit bridge from today's VM/path-based import resolution to tomorrow's assembly/module resolution.
6. Migrate `runmat-hir` first so it emits explicit modules, functions, classes, bindings, captures, and resolved semantic references.
7. Land the nominal class bridge early: `ClassId`, richer `HirClass`, class-aware lookup, and nominal `Type` support should move together rather than being deferred behind the rest of the HIR work.
8. Migrate `runmat-vm`, `runmat-core`, `runmat-lsp`, and `runmat-static-analysis` onto semantic IDs, semantic exports, and shared function/class summaries.
9. Implement fuller project/package composition on top of the resolved module/import model.
10. Finish the accelerate and GC lifetime hardening pass once the runtime ownership model is stable.

## Still To Lock Down

- a few exact HIR storage details such as arena/indexing choices and whether there is a separate module-item declaration-order table
- exact entrypoint and source-root config model
- exact runtime representation of captured bindings
- exact aliasing and lifetime model for accelerated values
- exact module discovery and qualified-name strategy for local projects and dependencies
- remaining workspace edge cases for modules, entrypoints, REPL execution, and desktop inspection

## Discrete Implementation Slices

The safest way to land this revision is as a sequence of explicit implementation slices rather than one large cross-workspace rewrite.

Each slice should produce a coherent resting state that:

- compiles across the affected crates
- has one clear temporary compatibility boundary
- reduces, rather than spreads, the current string-key and slot-key risks

The slices below are intentionally sequenced to contain the known cross-crate hazards:

- duplicated function identity across `String` keys
- import resolution spread across HIR, VM, and runtime path search
- parser-shaped class lowering drifting away from runtime class metadata
- object values degrading to `Type::Unknown`
- workspace semantics still depending on slots and assigned-name tracking

### Slice 1. Semantic ID and HIR Skeleton Freeze

Primary crates:

- `runmat-hir`
- `runmat-builtins`

Secondary crates:

- `runmat-parser`
- `runmat-static-analysis`
- `runmat-vm`

Landing goal:

- add the new semantic ID set
- introduce the canonical `HirAssembly` / `HirModule` / `HirFunction` / `HirClass` / `HirBinding` type skeletons
- keep old lowering operational behind a compatibility boundary while the new shapes are introduced

Expected concrete outcomes:

- `runmat-hir` defines `ModuleId`, `FunctionId`, `ClassId`, `EntrypointId`, `BindingId`, `ExprId`, and `StmtId`
- the provisional exact HIR sketch becomes real Rust types, even if some fields remain lightly populated at first
- old `HirProgram`-based paths remain available temporarily only as a migration bridge

Risk being retired in this slice:

- the project stops lacking a canonical semantic key space
- downstream crates can begin targeting stable IDs without each inventing their own transition model

Guardrail:

- do not attempt full lowering migration in the same step; the success condition is stable type ownership, not full semantic completeness yet

#### Slice 1 Exact First Implementation Checklist

The first landing for Slice 1 should be intentionally narrow. It should create the new semantic skeleton without forcing the rest of the compiler to consume it yet.

##### `runmat-hir`

Files to touch first:

- `runmat/crates/runmat-hir/src/ids.rs`
- `runmat/crates/runmat-hir/src/hir.rs`
- `runmat/crates/runmat-hir/src/lib.rs`

Files that should remain mostly untouched in the first landing:

- `runmat/crates/runmat-hir/src/lowering/ctx.rs`
- `runmat/crates/runmat-hir/src/lowering/stmt.rs`
- `runmat/crates/runmat-hir/src/lowering/expr.rs`
- `runmat/crates/runmat-hir/src/inference/`
- `runmat/crates/runmat-hir/src/remapping.rs`

Exact first-step checklist:

1. Extend `ids.rs` with the new semantic IDs:
   - `ModuleId`
   - `FunctionId`
   - `ClassId`
   - `EntrypointId`
   - `BindingId`
   - `ExprId`
   - `StmtId`
   - keep `VarId` and `SourceId` intact for compatibility during the bridge

2. Expand `hir.rs` with the new semantic container and item types:
   - `QualifiedName`
   - `HirAssembly`
   - `HirModule`
   - `HirEntrypoint`
   - `HirFunction`
   - `HirBinding`
   - `HirBlock`
   - `HirStmt` / `HirStmtKind`
   - `HirExpr` / `HirExprKind`
   - `HirCall`
   - `HirCallableRef`
   - `HirImport`
   - `HirClass` and related class/member structs

3. Keep the current `HirProgram`, `HirStmt`, `HirExpr`, and `LoweringResult` surface available during this first landing, even if that means the file temporarily contains both the legacy and next-generation structures side by side.

4. Update `lib.rs` exports so downstream crates can begin referring to the new semantic types and IDs without breaking the current lowering pipeline.

5. Add brief invariants directly near the new type definitions:
   - `HirAssembly` is the canonical owner of functions, classes, and bindings
   - modules index top-level items only
   - nested functions are represented through parent relations rather than duplicated inventories
   - classes own method membership via `FunctionId`

6. Do not yet make `lower()` emit `HirAssembly`.
   - the first landing should only establish the type layer
   - the existing `lower()` pipeline should continue producing `LoweringResult`

Suggested acceptance criteria:

- `runmat-hir` builds with both the legacy and new semantic type layers present
- no downstream crate is forced to migrate in the same commit
- the new semantic types are importable from `runmat_hir`

##### `runmat-builtins`

Files to touch first:

- `runmat/crates/runmat-builtins/src/lib.rs`

Exact first-step checklist:

1. Do not fully redesign `Type` in Slice 1.
   - that belongs to Slice 2
   - Slice 1 should only add the minimum scaffolding needed so the new HIR types are not blocked on builtins-level naming or class-kind concepts

2. If needed for type references, add only lightweight shared semantic scaffolding that is safe before the nominal-type migration, such as:
   - a small class-kind enum if it is genuinely useful outside HIR
   - comments or transitional notes marking `Object`, `HandleObject`, and `ClassRef` as precursors to the nominal class-type pass

3. Do not change:
   - `Type::unify`
   - object-to-type lowering behavior
   - `DataDataset` / `DataArray` / `DataTransaction` behavior

Suggested acceptance criteria:

- `runmat-builtins` remains behaviorally unchanged in Slice 1
- any scaffolding added here exists only to unblock HIR type definitions, not to partially land Slice 2 early

##### Cross-crate rule for the first landing

The first Slice 1 commit should not require synchronized edits across `runmat-vm`, `runmat-core`, `runmat-lsp`, or `runmat-static-analysis`.

If a proposed Slice 1 change forces those crates to update immediately, it is likely no longer a skeleton freeze and should be deferred into Slice 2, 3, 4, or 5.

### Slice 2. Nominal Type and Runtime Class Bridge

Primary crates:

- `runmat-builtins`
- `runmat-hir`

Secondary crates:

- `runmat-runtime`
- `runmat-lsp`
- `runmat/bindings/ts`

Landing goal:

- make nominal classes first-class in the type domain and align the type layer with the existing runtime class registry

Expected concrete outcomes:

- `Type` grows nominal class-oriented cases rather than degrading objects and class refs to `Unknown`
- class metadata surfaces line up with the runtime registry concepts of class name, parent, methods, and properties
- `DataDataset`, `DataArray`, and `DataTransaction` are documented and treated as transitional nominal-class bridge cases rather than the permanent pattern

Risk being retired in this slice:

- class work and type work stop being blocked on each other
- downstream crates no longer have to choose between nominal runtime behavior and opaque type information

Guardrail:

- do not fully replace all `Data*` special cases yet; first make the nominal path real and consumable

### Slice 3. Function Ownership, Capture Graph, and Binding Ownership in HIR

Primary crates:

- `runmat-hir`

Secondary crates:

- `runmat-static-analysis`
- `runmat-vm`
- `runmat-core`

Landing goal:

- migrate lowering so functions, bindings, locals, and captures become explicit semantic items rather than reconstructed side effects of statement lowering

Expected concrete outcomes:

- `HirStmt::Function` stops being the durable semantic representation
- anonymous functions lower to `FunctionId`
- binding ownership and storage are explicit on `HirBinding`
- nested capture relations are recorded explicitly
- class methods are represented as normal `HirFunction`s owned through `HirClass.methods`

Risk being retired in this slice:

- function identity stops being spread across ad hoc statement walks
- captures stop being implicit lexical accidents hidden in lowering context state

Guardrail:

- keep the runtime/VM side consuming a compatibility view if needed, but do not introduce a second new function identity scheme during the transition

### Slice 4. Replace String-Keyed Function Reconstruction with `FunctionId` Summaries

Primary crates:

- `runmat-static-analysis`
- `runmat-hir`

Secondary crates:

- `runmat-lsp`
- `runmat-vm`

Landing goal:

- move analysis off `String`-keyed function summary maps and off reconstruction helpers like `collect_function_defs`

Expected concrete outcomes:

- one `FunctionId`-keyed summary model
- one environment model keyed by `BindingId`
- one resolved-call model keyed by semantic IDs
- analysis no longer depends on function name uniqueness as a hidden invariant

Risk being retired in this slice:

- name-collision and shadowing hazards stop leaking into inference and analysis
- HIR and analysis stop drifting on what the callable graph actually is

Guardrail:

- this slice should land before a large VM migration, otherwise the VM and analysis layers will fork the function model again

### Slice 5. VM Slot Derivation and Semantic Call Lowering

Primary crates:

- `runmat-vm`
- `runmat-hir`

Secondary crates:

- `runmat-builtins`
- `runmat-static-analysis`

Landing goal:

- make VM layout and call lowering a derived consequence of semantic HIR rather than a parallel identity system

Expected concrete outcomes:

- bytecode/user-function tables no longer use `String` as their primary semantic identity
- slot numbering is derived from explicit bindings/functions rather than standing in for them
- call lowering consumes semantic callee identity where available
- the old import-resolution heuristics in `runmat-vm` shrink toward a compatibility bridge rather than the semantic source of truth

Risk being retired in this slice:

- the VM stops being a second compiler architecture with its own function/import identity rules

Guardrail:

- do not attempt the full project/dependency model yet; this slice should consume the phase-1 resolved import bridge, not invent the final composition system locally in the VM

### Slice 6. Semantic Workspace Export and Session Integration

Primary crates:

- `runmat-core`
- `runmat-vm`

Secondary crates:

- `runmat-runtime`
- `runmat-snapshot`

Landing goal:

- make workspace export/import and desktop-visible variable state depend on semantic binding visibility rather than slots and assigned-name tracking

Expected concrete outcomes:

- `runmat-core` gains an explicit semantic export step after execution
- workspace-visible state is keyed by semantic bindings plus display names, not just `String -> slot`
- `ans`, globals, persistents, and top-level entrypoint-visible bindings follow one shared semantic contract

Risk being retired in this slice:

- the current workspace model stops depending on operational slot behavior
- interpreter/JIT/workspace reconciliation drift becomes easier to reason about and test

Guardrail:

- land this only after binding ownership and function ownership are stable enough that exports are not being redefined every slice

### Slice 7. Phase-1 Import and Module Resolution Bridge

Primary crates:

- `runmat-hir`
- `runmat-runtime`
- `runmat-vm`

Secondary crates:

- `runmat-config`
- `runmat-core`

Landing goal:

- unify the current split import-resolution story enough that HIR, VM, and runtime path search all agree on one temporary resolved-composition bridge

Expected concrete outcomes:

- a clearly defined phase-1 resolved import model
- HIR carries the resolution state the VM/compiler consumes
- runtime path search becomes an implementation source for discovery, not an independent semantic resolver

Risk being retired in this slice:

- import behavior stops being spread across three semi-independent systems
- composition work later can build on one agreed boundary rather than rewriting different resolution rules in place

Guardrail:

- do not combine this with the full package/dependency manifest rollout; first unify the existing resolution pipeline

### Slice 8. Config-Driven Composition and Entrypoints

Primary crates:

- `runmat-config`
- `runmat-cli`
- `runmat-core`

Secondary crates:

- `runmat-hir`
- `runmat-runtime`
- `runmat-kernel`
- `runmat-wasm`

Landing goal:

- move from host/path-centric execution to explicit project composition using `[sources]`, `[dependencies]`, and `[[entrypoints]]`

Expected concrete outcomes:

- `runmat.toml` manifest parsing and validation
- entrypoint selection becomes explicit
- source discovery and dependency availability are no longer implicit host behavior
- scripts remain source-style only and lower to synthetic entry functions internally

Risk being retired in this slice:

- composition stops depending on ad hoc file-path and runtime-path behavior
- hosts and tools gain a reproducible project model

Guardrail:

- this slice should consume the already-landed phase-1 import bridge rather than redefining import semantics at the same time

### Slice 9. LSP, Docs, and Metadata Consumers Move to the Shared Semantic Model

Primary crates:

- `runmat-lsp`
- `runmat-static-analysis`

Secondary crates:

- `runmat/bindings/ts`
- `runmat/crates/runmat-lsp/src/core/builtins_json.rs`

Landing goal:

- make editor/tooling/documentation surfaces consume the same semantic HIR, analysis facts, and nominal class metadata as the compiler/runtime

Expected concrete outcomes:

- LSP symbol and semantic analysis stop reconstructing the old model
- builtin/class metadata projections are fed from the unified metadata direction where possible
- object/class/property/method information becomes consistent across runtime, docs, and tooling

Risk being retired in this slice:

- tool-facing semantics stop drifting from compiler/runtime semantics

Guardrail:

- avoid building new tooling-only metadata shims if the runtime-facing metadata source can be reused

### Slice 10. Accelerate, GC, and Lifetime Hardening on Top of the Stable Ownership Model

Primary crates:

- `runmat-accelerate`
- `runmat-accelerate-api`
- `runmat-gc`

Secondary crates:

- `runmat-core`
- `runmat-vm`
- `runmat-runtime`

Landing goal:

- finish the descoping, allocation/free, and residency/lifetime cleanup only after ownership, bindings, workspace export, and nominal object/type behavior are stable

Expected concrete outcomes:

- accelerate/fusion consumers use the shared semantic fact model
- residency and GC lifetimes are tied to the stabilized ownership/export model rather than the old slot-centric assumptions
- object and accelerated-value lifetime rules become easier to reason about across execution modes

Risk being retired in this slice:

- the memory/lifetime pass no longer has to fight moving compiler/runtime identity semantics underneath it

Guardrail:

- do not pull this slice earlier just because it is urgent operationally; it will be far safer once the ownership model is settled

### Sequencing Rule

Each slice should end with an explicit readback of:

- which old compatibility layer still exists
- which crate now owns the new source of truth
- which known cross-crate risk was reduced
- which follow-on slice is now unblocked

If a slice cannot answer those four questions clearly, it is too broad or is trying to move too many ownership boundaries at once.

## Implementation Status

This section is intentionally stricter than casual progress language. It distinguishes:

- core slice completion
- downstream validation
- broader consumer migration

These are not the same thing and should not be conflated.

### Status Definitions

#### Core Slice Completion

A slice is core-complete when the primary crates named in that slice have landed the architectural change that slice was actually meant to deliver.

This means:

- the new source of truth exists in code
- the affected primary crates compile
- the slice's main semantic/runtime invariants are represented directly rather than only in notes

This does not automatically mean every downstream crate has already migrated.

#### Downstream Validation

A slice is downstream-validated when at least one real consumer outside the primary implementation crate(s) can use the new source of truth without immediately running into missing structure, missing metadata, or incompatible assumptions.

This is a confidence measure, not a redefinition of the slice boundary.

#### Broad Consumer Migration

This is the later phase where multiple downstream crates are moved over in earnest.

This should be treated as follow-on work after a slice is core-complete, even if small consumer probes begin earlier.

### Current Strict Status

#### Slice 1: Semantic ID and HIR Skeleton Freeze

Core status: complete.

Rationale:

- semantic IDs exist in `runmat-hir`
- the semantic HIR type surface exists in code
- the legacy HIR path remains available as the intended compatibility bridge

#### Slice 2: Nominal Type and Runtime Class Bridge

Core status: complete enough for the intended bridge.

Rationale:

- nominal class-aware `Type` cases exist
- runtime/VM class metadata access is centralized enough to act as a shared metadata layer
- HIR inference understands the nominal `DataDataset` / `DataArray` / `DataTransaction` bridge

Important nuance:

- this does not mean all domain-specific `Data*` special cases are gone
- it means the nominal bridge that Slice 2 was supposed to create is now real in code

#### Slice 3: Function Ownership, Capture Graph, and Binding Ownership in HIR

Core status: complete enough to serve as the semantic backbone.

Rationale:

- `runmat-hir` now emits `semantic_hir` alongside legacy lowering
- semantic modules, functions, classes, bindings, bodies, and capture relations are populated
- anonymous functions, class methods, global/persistent storage, and handle-class kind all have semantic representation
- targeted `runmat-hir` tests cover script entrypoints, class inventories, nested captures, anonymous functions, and storage modifiers

Important nuance:

- some later-phase refinements still remain, such as fuller call resolution and eventual replacement of legacy lowering paths
- those are not required to consider the Slice 3 semantic backbone itself landed

### Downstream Validation Status

The semantic-core work above is no longer only theoretical. It has active downstream validation.

Current validation status:

- `runmat-static-analysis::lints::data_api` is now substantially semantic-first
- `runmat-static-analysis::lints::shape` is now semantic-first for traversal

This means the semantic HIR is not just present in `runmat-hir`; it is already usable by real downstream consumers.

### Broad Consumer Migration Status

Broad consumer migration is in progress, not complete.

Current state:

- `runmat-static-analysis` has begun meaningful semantic-HIR adoption
- `runmat-vm`, `runmat-core`, and `runmat-lsp` still have substantial migration work remaining
- the full `FunctionId` / `BindingId`-first analysis/runtime/export model is not yet complete across the workspace

### Decision Rule Going Forward

From this point on, work should be described using the following language:

- "Slices 1-3 core-complete" when referring to the semantic foundation inside `runmat-hir` and `runmat-builtins`
- "downstream validation in progress" when referring to early consumer adoption
- "broad consumer migration" when referring to the remaining crate-by-crate rollout

This keeps the project honest about what is truly done versus what is merely enabled.

### Next Recommended Effort

The next strongest move is to continue the broad consumer migration in the order already laid out above, starting with `runmat-static-analysis` until its remaining legacy bridges are minimal, and then moving on to `runmat-lsp`, `runmat-vm`, and `runmat-core`.

Reason:

- `runmat-static-analysis` is already the most advanced semantic-HIR consumer
- continuing there will further pressure-test the new ownership/binding/capture model
- it reduces the risk of carrying two analysis architectures longer than necessary
