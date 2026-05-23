# Symptom Validation Closure

Last updated: 2026-05-23 (America/Los_Angeles)

## Purpose

Track symptom-ticket closeout status across sessions with direct repro proof and explicit tracker actions.

Tracker IDs are intentionally omitted from this repository document.

## Scope and Method

- Parent effort and child workstreams are tracked in Linear.
- Inventory source: related symptom issues linked from that effort.
- Workflow mapping: Linear uses `In Review` as the practical "Ready for Review" state.
- Validation source types:
  - focused regression tests
  - direct script repros run via `cargo run -p runmat -- --no-jit ...`
  - explicit per-issue Linear comments documenting outcome

## Executive Snapshot

- Connected symptom issues audited: `27`
- Issues now in `In Review`: `14`
- Done/Canceled: `2`
- Remaining non-backlog open issues requiring additional work or verification: `7`
- Remaining backlog issues (not targeted for this closeout pass): `4`

## Status Ledger (ID-Free)

| Symptom Area | Current State | Closeout Status | Evidence Summary | Action |
|---|---|---|---|---|
| Shared input/output variable aliasing | In Review | Resolved/ready for review | VM regressions cover in-place update and multi-output read semantics | Commented + moved to `In Review` |
| Package namespace resolution (`+PackageName`) | In Review | Resolved/ready for review | Added execution-time source-tree function loading with qualified-symbol aliasing; validated both `import pkg.*; foo()` and direct `pkg.foo()` in core regressions and CLI repros | Commented + moved to `In Review` |
| Empty-concat dynamic growth semantics | In Review | Resolved/ready for review | `y=[]; y=[y,1]` and `z=[]; z=[z;2]` fixed with runtime regressions for true-empty neutral behavior in `horzcat` and `vertcat` | Commented + moved to `In Review` |
| Runtime bug umbrella | In Progress | Open | Umbrella remains active pending remaining child closure | Commented, no state change |
| Module graph / function resolution umbrella | In Progress | Open | Cross-file helper function execution now validates via project source-tree loading, but broader umbrella remains open pending full end-to-end closure verification (including external data/workspace cases) | Commented, no state change |
| Bracketed call-list expression statement parsing | In Review | Resolved/ready for review | Parser regression + statement dispatch guard fix validated | Commented + moved to `In Review` |
| >2GB GPU allocation splitting | In Progress | Open | Still open | Commented, no state change |
| GPU lifetime / GC leak class | In Progress | Open | Still open | Commented, no state change |
| Canceled item | Canceled | N/A | Already canceled | Commented, no state change |
| Existing in-review item | In Review | In review | Was already in review before this closeout | No state change |
| Command syntax handling (`grid on`, `hold on`, `clear all`, `clc`) | In Review | Resolved/ready for review | Repros validated | Commented + moved to `In Review` |
| wasm unreachable crash symptom | In Progress | Open | wasm-only crash path still open | Commented, no state change |
| Histogram `DisplayName` property parity | In Review | Resolved/ready for review | Added get/set support + regression; direct `set/get` repro matches expected value | Commented + moved to `In Review` |
| Native-pass but wasm/browser-specific symptom | In Progress | Open | Native script passes; wasm/browser closure proof still pending | Commented, no state change |
| Multi-output destructuring | In Review | Resolved/ready for review | Covered across user and builtin call paths | Commented + moved to `In Review` |
| Already done item | Done | Closed | Already done | No state change |
| `nargin`/`nargout` optional-arg guard behavior | In Review | Resolved/ready for review | Bare `nargin`/`nargout` with optional-arg guard pattern validated | Commented + moved to `In Review` |
| `varargin` / indexed `varargout{k}` semantics | In Review | Resolved/ready for review | Focused tests + repro scripts validated indexed varargout fill semantics | Commented + moved to `In Review` |
| Implicit indexed array creation (`x(3)=10`) | In Review | Resolved/ready for review | Ticket repro initial conditions now pass | Commented + moved to `In Review` |
| Parser bug umbrella | In Review | Resolved/ready for review | Child parser symptoms are now all in review or canceled | Commented + moved to `In Review` |
| GPU GC umbrella | In Progress | Open | Umbrella remains open with active GPU/wasm-related children | Commented, no state change |
| VM semantic bug umbrella | In Review | Resolved/ready for review | Child VM semantic symptoms are all in review | Commented + moved to `In Review` |
| Backlog parity item A | Backlog | Open | Backlog scope | No state change |
| Backlog parity item B | Backlog | Open | Backlog scope | No state change |
| Backlog parity item C | Backlog | Open | Backlog scope | No state change |
| `get(h)` property-bag completeness | In Review | Resolved/ready for review | Direct repro returns populated property bag with expected keys | Commented + moved to `In Review` |
| Backlog figure name-value support | Backlog | Open | Backlog scope | No state change |

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

## Focused Validation Commands (latest pass)

- `cargo test -p runmat-vm --test functions missing_fixed_input_can_be_guarded_with_nargin -- --nocapture`
- `cargo test -p runmat-vm --test functions bare_nargin_counts_varargin_inputs -- --nocapture`
- `cargo test -p runmat-vm --test functions varargout_indexed_fill_respects_nargout_loop -- --nocapture`
- `cargo test -p runmat-vm --test functions implicit_array_creation_from_linear_index_assignment -- --nocapture`
- `cargo test -p runmat-parser test_bracketed_call_list_expression_statement_parses -- --nocapture`
- `cargo test -p runmat-runtime true_empty_operand_is_neutral_for_dynamic_growth -- --nocapture`
- `cargo test -p runmat-runtime histogram_supports_displayname_property -- --nocapture`
- `cargo test -p runmat-core execute_outcome_resolves_ -- --nocapture`

## Tracker Actions Recorded

- Added per-issue comments across symptom inventory.
- Moved resolved symptoms to `In Review`.
- Left open, unresolved, or umbrella items in active states.

## Remaining Direct-Work Queue

1. Obtain wasm/browser-specific closure proof for the native-pass wasm symptom.
2. Debug and close the wasm unreachable crash repro path.
3. Finish GPU allocation/lifetime/GC fixes and verification, then close the GPU umbrella.
4. Complete and verify remaining module graph umbrella scenarios.
5. Close runtime umbrella after remaining non-backlog children are in review.
