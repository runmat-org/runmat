## Language Compatibility Modes

Philosophy: RunMat's input language surface is compatible with MATLAB syntax so the language you already know works here — no new language to learn. The goal is not to replicate a legacy runtime function-for-function, but to give engineers a modern, high-performance environment for math and physics that feels immediately familiar. Where MATLAB conventions conflict with clarity or performance, RunMat offers explicit opt-in/opt-out controls via compatibility modes.

RunMat is a superset language. We want to run existing MATLAB/Octave-style code, while steadily steering toward clearer, explicit, and more strongly-typed patterns (similar in spirit to how TypeScript tightens JavaScript). Divergences from legacy behavior will be explicit, opt-in/opt-out via config, and documented here.

### Compat modes (configured in `.runmat`)

```toml
# .runmat
[language]
compat = "runmat"           # default
# compat = "matlab"         # opt-in: legacy-friendly
# compat = "strict"         # opt-in: explicit form only
```

#### `compat = "runmat"` (default)
- Identical to `compat = "matlab"` but with the default error namespace set to `RunMat` instead of `MATLAB`. E.g. an undefined function will raise an error with the identifier `RunMat:UndefinedFunction` instead of `MATLAB:UndefinedFunction`.

#### `compat = "matlab"` (legacy-friendly)
- Accepts a curated set of MATLAB command-style forms and rewrites them to regular calls internally so analysis and fusion planning stay well-typed.
- We discourage the implicit form because:
  - It introduces ambiguity/shadowing (e.g., `on`, `off`, `tight` can collide with user variables).
  - It weakens static analysis, refactoring, and type-checking—the call shape is implicit.
  - It blurs the line between data and function calls, making future extensions/options harder to reason about.
  - It is less consistent with the explicit, typed calling style we want to promote long term.

#### `compat = "strict"`
- Disables command-style implicit forms; authors must call functions with parentheses/strings (e.g., `hold("on")`).
- Recommended for codebases that want maximal clarity and strict control over implicit behavior.

### MATLAB implicit command syntax

RunMat supports command-style invocation for the verbs below when `compat = "matlab"` or `compat = "runmat"`; under `compat = "strict"` these must be called explicitly via the explicit form (e.g., `hold("on")` rather than `hold on`).

**Plotting/layout implicit command verbs**
- `hold on|off|all|reset`
- `grid on|off`
- `box on|off`
- `axis auto|manual|tight|equal|ij|xy` (numeric arg forms still supported via normal calls)
- `shading flat|interp|faceted`
- `colormap <name>`
- `colorbar on|off` (or no-arg add/remove)
- `figure` (command form)
- `subplot` (command form)
- `clf`, `cla`, `close` (command forms)
- (We may extend this list; additions will be documented here.)

**Explicit form**
- Use explicit calls with strings/args, e.g.:
  - `hold("on"); grid("on"); box("off");`
  - `axis("equal"); shading("interp"); colormap("parula");`
  - `figure(); subplot(2, 1, 1);`

**Fusion/analysis treatment**
- These verbs are treated as side-effect-only; they do not break compilation or fusion planning. Internally they are normalized to explicit calls for consistent typing and tooling.

### Implementation notes

- Compatibility is decided up front by the parser (`runmat_parser::parse_with_options`). Under `compat = "matlab"` or `compat = "runmat"` mode, the parser accepts the whitelisted command verbs; under `compat = "strict"` the same tokens produce targeted parse errors.
- The setting is read from `[language] compat` in `.runmat` (see `/docs/configuration`). CLI builds, the native runtime, and the WASM runtime all forward this value into `RunMatSession`.
- Hosts can override at runtime: the wasm bindings expose `session.setLanguageCompat("strict")` / `"matlab" / "runmat"`, and the LSP accepts `initializationOptions.language.compat`. When no explicit override is provided, the LSP will auto-discover the closest `.runmat*` file under the workspace root and mirror that value.
- Because the parser itself understands the compat mode, downstream passes (HIR lowering, fusion planning, Monaco semantic tokens, etc.) all see the same explicit AST without needing ad-hoc rewrites.

### Future compatibility surfaces
- New divergences or compat shims (e.g., potential Julia-inspired interop/syntax in the future) will be documented here as they are added.
