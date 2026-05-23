# Symptom Validation Closure (RM-369)

Last updated: 2026-05-23 (America/Los_Angeles)

## Purpose

Track RM-369 symptom-ticket closeout status across sessions with direct repro proof and explicit Linear actions.

## Scope and Method

- Parent effort: `RM-369` with children `RM-370..RM-377`
- Inventory source: related symptom tickets linked from RM-369 child relations in Linear
- Workflow mapping: Linear uses `In Review` as the practical "Ready for Review" state
- Validation source types:
  - focused regression tests
  - direct script repros run via `cargo run -p runmat -- --no-jit ...`
  - explicit per-ticket Linear comments documenting outcome

## Executive Snapshot

- Connected symptom tickets audited: `27`
- Tickets now in `In Review`: `14`
  - `RM-40`, `RM-44`, `RM-180`, `RM-255`, `RM-286`, `RM-290`, `RM-298`, `RM-303`, `RM-309`, `RM-312`, `RM-323`, `RM-325`, `RM-327`, `RM-355`
- Done/Canceled: `2`
  - `RM-304` (Done), `RM-273` (Canceled)
- Remaining non-backlog open tickets requiring additional work or verification: `7`
  - `RM-228`, `RM-231`, `RM-270`, `RM-272`, `RM-295`, `RM-302`, `RM-326`
- Remaining backlog tickets (not targeted for this closeout pass): `4`
  - `RM-338`, `RM-339`, `RM-343`, `RM-359`

## Per-Ticket Status Ledger

| Ticket | Current State | RM-369 Closeout Status | Evidence Summary | Action |
|---|---|---|---|---|
| RM-40 | In Review | Resolved/ready for review | Shared in/out semantics covered by VM regressions (`shared_input_output_name_updates_in_place`, `shared_input_output_name_multi_output_reads_original_input`) | Commented + moved to `In Review` |
| RM-44 | In Review | Resolved/ready for review | Added execution-time source-tree function loading + qualified-symbol aliasing; validated `import pkg.*; foo()` and `pkg.foo()` in regressions and CLI repros | Commented + moved to `In Review` |
| RM-180 | In Review | Resolved/ready for review | Empty-concat dynamic growth fixed (`y=[]; y=[y,1]`, `z=[]; z=[z;2]`) with runtime regressions for true-empty neutral semantics in `horzcat`/`vertcat` | Commented + moved to `In Review` |
| RM-228 | In Progress | Open | Umbrella runtime bug bucket remains active pending remaining children | Commented, no state change |
| RM-231 | In Progress | Open | Cross-file helper execution now validates via source-tree loading, but umbrella remains open for full module-graph closure scope | Commented, no state change |
| RM-255 | In Review | Resolved/ready for review | Parser regression for bracketed call-list expression statements + stmt dispatch guard fix | Commented + moved to `In Review` |
| RM-270 | In Progress | Open | >2GB GPU allocation splitting still open | Commented, no state change |
| RM-272 | In Progress | Open | GPU lifetime/GC leak class still open | Commented, no state change |
| RM-273 | Canceled | N/A | Already canceled | Commented, no state change |
| RM-286 | In Review | In review | Already in review prior to this closeout | No state change |
| RM-290 | In Review | Resolved/ready for review | Command syntax repro (`grid on`, `hold on`, `clear all`, `clc`) validated | Commented + moved to `In Review` |
| RM-295 | In Progress | Ready for review evidence complete | Added wasm-node + browser regressions (`impedance_loop_executes_without_runtime_error`) and reran successfully in both harnesses | No state change (Linear closeout comment/state update blocked by external-write policy in this environment) |
| RM-298 | In Review | Resolved/ready for review | Histogram `DisplayName` get/set support added with regression; direct repro returns expected name | Commented + moved to `In Review` |
| RM-302 | In Progress | Ready for review evidence complete | Added wasm-node + browser regressions (`slice_end_arithmetic_executes_without_runtime_error`) and reran successfully in both harnesses | No state change (Linear closeout comment/state update blocked by external-write policy in this environment) |
| RM-303 | In Review | Resolved/ready for review | Multi-output destructuring covered across user+builtin paths | Commented + moved to `In Review` |
| RM-304 | Done | Closed | Already done | No state change |
| RM-309 | In Review | Resolved/ready for review | Bare `nargin`/`nargout` plus optional-arg guard pattern validated (direct repro returns `4`) | Commented + moved to `In Review` |
| RM-312 | In Review | Resolved/ready for review | `varargin` / indexed `varargout{k}` semantics validated with focused tests + repro scripts | Commented + moved to `In Review` |
| RM-323 | In Review | Resolved/ready for review | Implicit indexed array creation now works (`x(3)=10`) | Commented + moved to `In Review` |
| RM-325 | In Review | Resolved/ready for review | Parser bug bundle children now all `In Review`/`Canceled` (`RM-255`, `RM-290`, `RM-309`, `RM-273`) | Commented + moved to `In Review` |
| RM-326 | In Progress | Open | GPU GC umbrella remains open with active children (`RM-270`, `RM-272`, `RM-295`, `RM-302`) | Commented, no state change |
| RM-327 | In Review | Resolved/ready for review | VM semantic bug bundle children now all `In Review` (`RM-40`, `RM-180`, `RM-303`, `RM-323`, `RM-355`) | Commented + moved to `In Review` |
| RM-338 | Backlog | Open | Backlog feature parity item | No state change |
| RM-339 | Backlog | Open | Backlog FFT/complex-shape issue | No state change |
| RM-343 | Backlog | Open | Backlog ComplexTensor parity issue | No state change |
| RM-355 | In Review | Resolved/ready for review | `get(h)` now returns populated property bag in direct repro (`fieldnames` count non-zero with expected keys) | Commented + moved to `In Review` |
| RM-359 | Backlog | Open | Backlog figure name-value support | No state change |

## Key Implementation Commits Used for Closure

- `579d8e0f`
  - Parser: bracketed call-list expression statement parsing fix + regression
- `35567774`
  - VM semantics: optional fixed-input behavior (`nargin` guards), `varargout{k}` indexed fill, implicit indexed array creation
- `90704285`
  - Runtime semantics: true-empty (`[]`) concat neutral handling for dynamic growth (`horzcat`/`vertcat`)
  - Plotting semantics: histogram `DisplayName` get/set parity and property-bag inclusion
- `82952ff1`
  - Runtime/compiler integration: preload project/path-resolved `.m` symbols into semantic function registry at execution time
  - Namespace/module behavior: enables package import call resolution and qualified package calls from source trees
  - Regression coverage: package wildcard import, qualified package call, and cross-file helper resolution tests in `runmat-core`
- `aba5fb19`
  - Lowering semantics: treat statement-form `load(...)` as zero-requested-output so it assigns loaded symbols into caller workspace
  - VM workspace semantics: reuse cleared lexical slot when a name is re-assigned in the same execution (fixes `clear x; load(...); y = x` flow)
  - Regression coverage: `execute_outcome_load_statement_assigns_workspace_bindings_*`, `execute_load_statement_assigns_workspace_bindings_with_semicolon`, and VM slot-reuse unit test
- `23eaedcf`
  - Wasm runtime/wire safety: precompute workspace payload before stream move in `ExecutionPayload::from_outcome`; include external/method/bound function handles in wasm JSON value encoding
  - Cross-target compile safety: gate non-wasm helper functions in `runmat-core/src/session/run.rs` to fix wasm32 `-D warnings` dead-code build failures
  - Regression coverage: new wasm-targeted integration tests in `runmat-wasm/tests/symptom_node_regressions.rs` for impedance loop and `end`-arithmetic slice assignment paths

## Focused Validation Commands (latest pass)

- `cargo test -p runmat-vm --test functions missing_fixed_input_can_be_guarded_with_nargin -- --nocapture`
- `cargo test -p runmat-vm --test functions bare_nargin_counts_varargin_inputs -- --nocapture`
- `cargo test -p runmat-vm --test functions varargout_indexed_fill_respects_nargout_loop -- --nocapture`
- `cargo test -p runmat-vm --test functions implicit_array_creation_from_linear_index_assignment -- --nocapture`
- `cargo test -p runmat-parser test_bracketed_call_list_expression_statement_parses -- --nocapture`
- `cargo test -p runmat-runtime true_empty_operand_is_neutral_for_dynamic_growth -- --nocapture`
- `cargo test -p runmat-runtime histogram_supports_displayname_property -- --nocapture`
- `cargo test -p runmat-core execute_outcome_resolves_ -- --nocapture`
- `cargo test -p runmat-core execute_outcome_load_statement_assigns_workspace_bindings -- --nocapture`
- `cargo test -p runmat-core execute_load_statement_assigns_workspace_bindings_with_semicolon -- --nocapture`
- `cargo test -p runmat-vm assignment_after_remove_reuses_previous_slot -- --nocapture`
- `RUNMAT_GENERATE_WASM_REGISTRY=1 cargo check -p runmat-runtime --target wasm32-unknown-unknown`
- `wasm-pack test --node --test symptom_node_regressions` (pass: `2 passed; 0 failed`)
- `wasm-pack test --chrome --headless --chromedriver /tmp/chromedriver-148/chromedriver-mac-arm64/chromedriver --test symptom_browser_regressions` (pass: `2 passed; 0 failed`)
- `scripts/test-wasm-regression-suite.sh symptom-closure` (pass: node + browser symptom closure in one command)

## Tracker Actions Recorded

- Added per-issue comments across symptom inventory.
- Moved resolved symptoms to `In Review`.
- Left open, unresolved, or umbrella items in active states.
- Posted RM-302 validation comment referencing commit `23eaedcf`.
- RM-295/RM-302 Linear closeout comment and state update writes are currently blocked by external-write policy enforcement in this environment.

## Remaining Direct-Work Queue

1. `RM-295` / `RM-302`: post Linear closeout comments and move state to `In Review` (evidence is complete; external-write policy currently blocks the update here).
2. `RM-270` / `RM-272` and parent `RM-326`: finish GPU allocation/lifetime/GC fixes and verification.
3. `RM-286`: confirm browser sandbox file-I/O closure proof and move to `In Review` when validated.
4. `RM-228`: close umbrella after remaining non-backlog children are in review (currently includes `RM-286` and `RM-326` chain).
