// None of these tests use #[wasm_bindgen_test], so they cannot run in the
// browser via wasm-pack. Excluding them from wasm32 avoids compiling a full
// runmat-runtime wasm binary per test file with zero executable tests.
#![cfg(not(target_arch = "wasm32"))]

use futures::executor::block_on;
use runmat_core::{ExecutionStreamKind, RunMatSession};
use runmat_gc::gc_test_context;

#[test]
fn close_all_command_form_executes_without_lowering_all_as_a_builtin() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("figure(1);")).unwrap();
    block_on(engine.execute("figure(2);")).unwrap();

    let result = block_on(engine.execute("close all;")).unwrap();
    assert!(result.error.is_none());
}

#[test]
fn clear_command_clears_workspace_state() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("x = 1;")).unwrap();

    let result = block_on(engine.execute("clear;")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert!(result.workspace.values.is_empty());

    let err = block_on(engine.execute("x")).unwrap_err();
    assert!(err.to_string().contains("Undefined variable: x"));
}

#[test]
fn clear_all_command_form_clears_workspace_state() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("y = 42;")).unwrap();

    let result = block_on(engine.execute("clear all;")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert!(result.workspace.values.is_empty());

    let err = block_on(engine.execute("y")).unwrap_err();
    assert!(err.to_string().contains("Undefined variable: y"));
}

#[test]
fn clear_named_variable_removes_only_that_binding() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("x = 1; y = 2;")).unwrap();

    let result = block_on(engine.execute("clear x;")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert_eq!(result.workspace.values.len(), 1);
    assert_eq!(result.workspace.values[0].name, "y");

    let err = block_on(engine.execute("x")).unwrap_err();
    assert!(err.to_string().contains("Undefined variable: x"));

    let y_value = block_on(engine.execute("y")).unwrap();
    assert!(y_value.error.is_none());
    assert_eq!(
        y_value.value.as_ref().map(|v| v.to_string()),
        Some("2".to_string())
    );
}

#[test]
fn clear_multiple_named_variables_accepts_multiple_inputs() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("a = 1; b = 2; c = 3;")).unwrap();

    let result = block_on(engine.execute("clear a b;")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert_eq!(result.workspace.values.len(), 1);
    assert_eq!(result.workspace.values[0].name, "c");
}

#[test]
fn clear_followed_by_assignments_shows_vars_in_workspace() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    // Variables assigned after clear() in the same execution block must appear
    // in the workspace snapshot – the bug was that StoreVar did not re-register
    // names into ws.idx_to_name after workspace_clear() wiped the map.
    let result = block_on(
        engine.execute("clear();\nXRange = -2:0.02:2;\nYRange = -2:0.02:2;\ndisp(YRange);"),
    )
    .unwrap();

    assert!(result.error.is_none());
    let names: Vec<&str> = result
        .workspace
        .values
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert!(
        names.contains(&"XRange"),
        "XRange should be in workspace after clear(); XRange = ...; got: {names:?}"
    );
    assert!(
        names.contains(&"YRange"),
        "YRange should be in workspace after clear(); YRange = ...; got: {names:?}"
    );
}

#[test]
fn clearvars_command_clears_workspace_state() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("x = 1; y = 2;")).unwrap();

    let result = block_on(engine.execute("clearvars;")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert!(result.workspace.values.is_empty());

    let err = block_on(engine.execute("x")).unwrap_err();
    assert!(err.to_string().contains("Undefined variable: x"));
}

#[test]
fn clearvars_named_variable_removes_only_that_binding() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("x = 1; y = 2;")).unwrap();

    let result = block_on(engine.execute("clearvars x;")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert_eq!(result.workspace.values.len(), 1);
    assert_eq!(result.workspace.values[0].name, "y");

    let err = block_on(engine.execute("x")).unwrap_err();
    assert!(err.to_string().contains("Undefined variable: x"));
}

#[test]
fn clearvars_multiple_named_variables_accepts_multiple_inputs() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("a = 1; b = 2; c = 3;")).unwrap();

    let result = block_on(engine.execute("clearvars a b;")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert_eq!(result.workspace.values.len(), 1);
    assert_eq!(result.workspace.values[0].name, "c");
}

#[test]
fn clearvars_except_keeps_named_bindings() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("drop1 = 1; keep = 2; drop2 = 3;")).unwrap();

    let result = block_on(engine.execute("clearvars -except keep;")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert_eq!(result.workspace.values.len(), 1);
    assert_eq!(result.workspace.values[0].name, "keep");

    let value = block_on(engine.execute("keep")).unwrap();
    assert!(value.error.is_none());
    assert_eq!(
        value.value.as_ref().map(|v| v.to_string()),
        Some("2".to_string())
    );
}

#[test]
fn clearvars_selected_names_with_except_preserves_exclusions() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("a = 1; b = 2; c = 3; untouched = 4;")).unwrap();

    let result = block_on(engine.execute("clearvars a b c -except b;")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);

    let names: Vec<&str> = result
        .workspace
        .values
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert!(!names.contains(&"a"), "a should have been cleared");
    assert!(names.contains(&"b"), "b should have been preserved");
    assert!(!names.contains(&"c"), "c should have been cleared");
    assert!(
        names.contains(&"untouched"),
        "variables outside the selected clear set should remain"
    );
}

#[test]
fn clearvars_function_call_selected_names_with_except_preserves_exclusions() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    block_on(engine.execute("a = 1; b = 2; c = 3; untouched = 4;")).unwrap();

    let result = block_on(engine.execute("clearvars('a', 'b', 'c', '-except', 'b');")).unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);

    let names: Vec<&str> = result
        .workspace
        .values
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert!(!names.contains(&"a"), "a should have been cleared");
    assert!(names.contains(&"b"), "b should have been preserved");
    assert!(!names.contains(&"c"), "c should have been cleared");
    assert!(
        names.contains(&"untouched"),
        "variables outside the selected clear set should remain"
    );
}

#[test]
fn clearvars_repro_with_clc_and_close_all_executes() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    let result = block_on(engine.execute("clearvars; clc; close all\nx = 5;\ndisp(x);")).unwrap();
    assert!(result.error.is_none());
    assert!(
        result
            .streams
            .iter()
            .any(|entry| entry.stream == ExecutionStreamKind::ClearScreen),
        "expected clc to emit a clear-screen control event"
    );
}

#[test]
fn clc_emits_clear_screen_control_stream() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    let result = block_on(engine.execute("clc;")).unwrap();
    assert!(result.error.is_none());
    assert!(
        result
            .streams
            .iter()
            .any(|entry| entry.stream == ExecutionStreamKind::ClearScreen),
        "expected clear-screen control event in execution streams"
    );
}
