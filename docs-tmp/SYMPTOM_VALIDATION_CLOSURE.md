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
- Tickets now in `In Review`: `13`
  - `RM-40`, `RM-180`, `RM-255`, `RM-286`, `RM-290`, `RM-298`, `RM-303`, `RM-309`, `RM-312`, `RM-323`, `RM-325`, `RM-327`, `RM-355`
- Done/Canceled: `2`
  - `RM-304` (Done), `RM-273` (Canceled)
- Remaining non-backlog open tickets requiring additional work or verification: `8`
  - `RM-44`, `RM-228`, `RM-231`, `RM-270`, `RM-272`, `RM-295`, `RM-302`, `RM-326`
- Remaining backlog tickets (not targeted for this closeout pass): `4`
  - `RM-338`, `RM-339`, `RM-343`, `RM-359`

## Per-Ticket Status Ledger

| Ticket | Current State | RM-369 Closeout Status | Evidence Summary | Action |
|---|---|---|---|---|
| RM-40 | In Review | Resolved/ready for review | Shared in/out semantics covered by VM regressions (`shared_input_output_name_updates_in_place`, `shared_input_output_name_multi_output_reads_original_input`) | Commented + moved to `In Review` |
| RM-44 | In Progress | Open | `+PackageName` namespace behavior not fully signed off yet | Commented, no state change |
| RM-180 | In Review | Resolved/ready for review | Empty-concat dynamic growth fixed (`y=[]; y=[y,1]`, `z=[]; z=[z;2]`) with runtime regressions for true-empty neutral semantics in both `horzcat` and `vertcat` | Commented + moved to `In Review` |
| RM-228 | In Progress | Open | Umbrella runtime bug bucket still active | Commented, no state change |
| RM-231 | In Progress | Open | Broader module graph/resolution scope still active | Commented, no state change |
| RM-255 | In Review | Resolved/ready for review | Parser regression for bracketed call-list expression statements + stmt dispatch guard fix | Commented + moved to `In Review` |
| RM-270 | In Progress | Open | >2GB GPU allocation splitting still open | Commented, no state change |
| RM-272 | In Progress | Open | GPU lifetime/GC leak class still active | Commented, no state change |
| RM-273 | Canceled | N/A | Already canceled | Commented, no state change |
| RM-286 | In Review | In review | Already in review prior to this closeout | No state change |
| RM-290 | In Review | Resolved/ready for review | Command syntax repro (`grid on`, `hold on`, `clear all`, `clc`) validated | Commented + moved to `In Review` |
| RM-295 | In Progress | Open | wasm-only unreachable crash ticket still open | Commented, no state change |
| RM-298 | In Review | Resolved/ready for review | Added histogram `DisplayName` get/set support + regression test; direct repro `set/get` now returns expected name | Commented + moved to `In Review` |
| RM-302 | In Progress | Open | Exact script passes natively; wasm/browser-specific closure proof still pending | Commented, no state change |
| RM-303 | In Review | Resolved/ready for review | Multi-output destructuring covered across user+builtin paths | Commented + moved to `In Review` |
| RM-304 | Done | Closed | Already done | No state change |
| RM-309 | In Review | Resolved/ready for review | Bare `nargin`/`nargout` plus optional-arg guard pattern now passes; direct repro returns `4` | Commented + moved to `In Review` |
| RM-312 | In Review | Resolved/ready for review | `varargin` / indexed `varargout{k}` semantics validated with focused tests + repro scripts | Commented + moved to `In Review` |
| RM-323 | In Review | Resolved/ready for review | Implicit indexed array creation now works (`x(3)=10`, ticket script initial conditions) | Commented + moved to `In Review` |
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

## Focused Validation Commands (latest pass)

- `cargo test -p runmat-vm --test functions missing_fixed_input_can_be_guarded_with_nargin -- --nocapture`
- `cargo test -p runmat-vm --test functions bare_nargin_counts_varargin_inputs -- --nocapture`
- `cargo test -p runmat-vm --test functions varargout_indexed_fill_respects_nargout_loop -- --nocapture`
- `cargo test -p runmat-vm --test functions implicit_array_creation_from_linear_index_assignment -- --nocapture`
- `cargo test -p runmat-parser test_bracketed_call_list_expression_statement_parses -- --nocapture`
- `cargo test -p runmat-runtime true_empty_operand_is_neutral_for_dynamic_growth -- --nocapture`
- `cargo test -p runmat-runtime histogram_supports_displayname_property -- --nocapture`

## Linear Actions Recorded (newest entries)

- `RM-309` comment: `#comment-a9b19336`; moved to `In Review`
- `RM-312` comment: `#comment-6cf4d06a`; moved to `In Review`
- `RM-323` comment: `#comment-979db0a2`; moved to `In Review`
- `RM-180` comment: `#comment-e335ae5d`; moved to `In Review`
- `RM-298` comment: `#comment-5f29c7bf`; moved to `In Review`
- `RM-355` comment: `#comment-c27e9867`; moved to `In Review`
- `RM-325` comment: `#comment-b55fbc4d`; moved to `In Review`
- `RM-327` comment: `#comment-ba313990`; moved to `In Review`

## Remaining Direct-Work Queue

1. `RM-302`: obtain wasm/browser-specific closure proof (native repro already passes).
2. `RM-295`: debug/close wasm unreachable crash repro path.
3. `RM-270` / `RM-272` and parent `RM-326`: finish GPU allocation/lifetime/GC fixes and verification.
4. `RM-44` / `RM-231`: complete and verify module/package namespace behavior.
5. `RM-228`: close umbrella after remaining non-backlog children are in review.
