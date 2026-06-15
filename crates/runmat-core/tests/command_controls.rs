// None of these tests use #[wasm_bindgen_test], so they cannot run in the
// browser via wasm-pack. Excluding them from wasm32 avoids compiling a full
// runmat-runtime wasm binary per test file with zero executable tests.
#![cfg(not(target_arch = "wasm32"))]

use runmat_core::{ExecutionStreamKind, RunError, RunMatSession};
use runmat_gc::gc_test_context;
use std::sync::Mutex;

static PLOT_TEST_LOCK: Mutex<()> = Mutex::new(());

struct PlotTestGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
    disable_previous: Option<std::ffi::OsString>,
    host_previous: Option<std::ffi::OsString>,
}

impl Drop for PlotTestGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(previous) = self.disable_previous.take() {
                std::env::set_var("RUNMAT_DISABLE_INTERACTIVE_PLOTS", previous);
            } else {
                std::env::remove_var("RUNMAT_DISABLE_INTERACTIVE_PLOTS");
            }
            if let Some(previous) = self.host_previous.take() {
                std::env::set_var("RUNMAT_HOST_MANAGED_PLOTS", previous);
            } else {
                std::env::remove_var("RUNMAT_HOST_MANAGED_PLOTS");
            }
        }
    }
}

fn isolate_plot_test() -> PlotTestGuard {
    let guard = PLOT_TEST_LOCK.lock().unwrap_or_else(|err| err.into_inner());
    let disable_previous = std::env::var_os("RUNMAT_DISABLE_INTERACTIVE_PLOTS");
    let host_previous = std::env::var_os("RUNMAT_HOST_MANAGED_PLOTS");
    unsafe {
        std::env::set_var("RUNMAT_DISABLE_INTERACTIVE_PLOTS", "1");
        std::env::set_var("RUNMAT_HOST_MANAGED_PLOTS", "1");
    }
    runmat_runtime::builtins::plotting::reset_plot_state();
    PlotTestGuard {
        _guard: guard,
        disable_previous,
        host_previous,
    }
}

#[test]
fn close_all_command_form_executes_without_lowering_all_as_a_builtin() {
    let _plot_guard = isolate_plot_test();
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "figure(1);").unwrap();
    runmat_core::execute_text_request_for_testing(&mut engine, "figure(2);").unwrap();

    let result = runmat_core::execute_text_request_for_testing(&mut engine, "close all;").unwrap();
    assert!(result.error.is_none());
}

#[test]
fn clear_command_clears_workspace_state() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "x = 1;").unwrap();

    let result = runmat_core::execute_text_request_for_testing(&mut engine, "clear;").unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert!(result.workspace.values.is_empty());

    let err = runmat_core::execute_text_request_for_testing(&mut engine, "x").unwrap_err();
    let RunError::Semantic(err) = err else {
        panic!("expected semantic undefined-variable error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn clear_all_command_form_clears_workspace_state() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "y = 42;").unwrap();

    let result = runmat_core::execute_text_request_for_testing(&mut engine, "clear all;").unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert!(result.workspace.values.is_empty());

    let err = runmat_core::execute_text_request_for_testing(&mut engine, "y").unwrap_err();
    let RunError::Semantic(err) = err else {
        panic!("expected semantic undefined-variable error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn clear_named_variable_removes_only_that_binding() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "x = 1; y = 2;").unwrap();

    let result = runmat_core::execute_text_request_for_testing(&mut engine, "clear x;").unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert_eq!(result.workspace.values.len(), 1);
    assert_eq!(result.workspace.values[0].name, "y");

    let err = runmat_core::execute_text_request_for_testing(&mut engine, "x").unwrap_err();
    let RunError::Semantic(err) = err else {
        panic!("expected semantic undefined-variable error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));

    let y_value = runmat_core::execute_text_request_for_testing(&mut engine, "y").unwrap();
    assert!(y_value.error.is_none());
    assert_eq!(
        y_value.value.as_ref().map(|v| v.to_string()),
        Some("2".to_string())
    );
}

#[test]
fn clear_multiple_named_variables_accepts_multiple_inputs() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "a = 1; b = 2; c = 3;").unwrap();

    let result = runmat_core::execute_text_request_for_testing(&mut engine, "clear a b;").unwrap();
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
    let result = runmat_core::execute_text_request_for_testing(
        &mut engine,
        "clear();\nXRange = -2:0.02:2;\nYRange = -2:0.02:2;\ndisp(YRange);",
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
fn untaken_branch_assignments_do_not_overwrite_existing_values() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "untaken = 1;").unwrap();

    let result = runmat_core::execute_text_request_for_testing(
        &mut engine,
        "if 0; untaken = 2; else; taken = 3; end;",
    )
    .unwrap();
    assert!(result.error.is_none());

    let taken = runmat_core::execute_text_request_for_testing(&mut engine, "taken").unwrap();
    assert_eq!(
        taken.value.as_ref().map(|v| v.to_string()),
        Some("3".to_string())
    );

    let untaken = runmat_core::execute_text_request_for_testing(&mut engine, "untaken").unwrap();
    assert_eq!(
        untaken.value.as_ref().map(|v| v.to_string()),
        Some("1".to_string()),
        "untaken branch should not overwrite prior workspace value"
    );
}

#[test]
fn jit_fallback_preserves_workspace_tracking_after_preflight_failure() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "textValue = \"hello\";").unwrap();
    for _ in 0..12 {
        let result =
            runmat_core::execute_text_request_for_testing(&mut engine, "jitFallbackTarget = 1;")
                .unwrap();
        assert!(result.error.is_none());
    }

    let clear_result =
        runmat_core::execute_text_request_for_testing(&mut engine, "clear jitFallbackTarget;")
            .unwrap();
    assert!(clear_result.error.is_none());

    let result =
        runmat_core::execute_text_request_for_testing(&mut engine, "jitFallbackTarget = 2;")
            .unwrap();
    assert!(result.error.is_none());

    let names: Vec<&str> = result
        .workspace
        .values
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert!(
        names.contains(&"jitFallbackTarget"),
        "interpreter fallback after JIT preflight failure should still publish assignments; got: {names:?}"
    );
}

#[test]
fn clearvars_command_clears_workspace_state() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "x = 1; y = 2;").unwrap();

    let result = runmat_core::execute_text_request_for_testing(&mut engine, "clearvars;").unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert!(result.workspace.values.is_empty());

    let err = runmat_core::execute_text_request_for_testing(&mut engine, "x").unwrap_err();
    let RunError::Semantic(err) = err else {
        panic!("expected semantic undefined-variable error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn clearvars_named_variable_removes_only_that_binding() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "x = 1; y = 2;").unwrap();

    let result =
        runmat_core::execute_text_request_for_testing(&mut engine, "clearvars x;").unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert_eq!(result.workspace.values.len(), 1);
    assert_eq!(result.workspace.values[0].name, "y");

    let err = runmat_core::execute_text_request_for_testing(&mut engine, "x").unwrap_err();
    let RunError::Semantic(err) = err else {
        panic!("expected semantic undefined-variable error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn clearvars_multiple_named_variables_accepts_multiple_inputs() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "a = 1; b = 2; c = 3;").unwrap();

    let result =
        runmat_core::execute_text_request_for_testing(&mut engine, "clearvars a b;").unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert_eq!(result.workspace.values.len(), 1);
    assert_eq!(result.workspace.values[0].name, "c");
}

#[test]
fn clearvars_except_keeps_named_bindings() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(&mut engine, "drop1 = 1; keep = 2; drop2 = 3;")
        .unwrap();

    let result =
        runmat_core::execute_text_request_for_testing(&mut engine, "clearvars -except keep;")
            .unwrap();
    assert!(result.error.is_none());
    assert!(result.workspace.full);
    assert_eq!(result.workspace.values.len(), 1);
    assert_eq!(result.workspace.values[0].name, "keep");

    let value = runmat_core::execute_text_request_for_testing(&mut engine, "keep").unwrap();
    assert!(value.error.is_none());
    assert_eq!(
        value.value.as_ref().map(|v| v.to_string()),
        Some("2".to_string())
    );
}

#[test]
fn clearvars_selected_names_with_except_preserves_exclusions() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    runmat_core::execute_text_request_for_testing(
        &mut engine,
        "a = 1; b = 2; c = 3; untouched = 4;",
    )
    .unwrap();

    let result =
        runmat_core::execute_text_request_for_testing(&mut engine, "clearvars a b c -except b;")
            .unwrap();
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

    runmat_core::execute_text_request_for_testing(
        &mut engine,
        "a = 1; b = 2; c = 3; untouched = 4;",
    )
    .unwrap();

    let result = runmat_core::execute_text_request_for_testing(
        &mut engine,
        "clearvars('a', 'b', 'c', '-except', 'b');",
    )
    .unwrap();
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
fn clearvars_named_variable_is_undefined_later_in_same_execution() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    let result =
        runmat_core::execute_text_request_for_testing(&mut engine, "x = 7; clearvars x; disp(x);")
            .unwrap();
    assert_eq!(
        result.error.as_ref().and_then(|err| err.identifier()),
        Some("RunMat:UndefinedVariable")
    );

    let err = runmat_core::execute_text_request_for_testing(&mut engine, "x").unwrap_err();
    let RunError::Semantic(err) = err else {
        panic!("expected semantic undefined-variable error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn clear_named_variable_is_undefined_later_in_same_execution() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    let result =
        runmat_core::execute_text_request_for_testing(&mut engine, "x = 7; clear x; disp(x);")
            .unwrap();
    assert_eq!(
        result.error.as_ref().and_then(|err| err.identifier()),
        Some("RunMat:UndefinedVariable")
    );

    let err = runmat_core::execute_text_request_for_testing(&mut engine, "x").unwrap_err();
    let RunError::Semantic(err) = err else {
        panic!("expected semantic undefined-variable error");
    };
    assert_eq!(err.identifier.as_deref(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn clearvars_repro_with_clc_and_close_all_executes() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    let result = runmat_core::execute_text_request_for_testing(
        &mut engine,
        "clearvars; clc; close all\nx = 5;\ndisp(x);",
    )
    .unwrap();
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

    let result = runmat_core::execute_text_request_for_testing(&mut engine, "clc;").unwrap();
    assert!(result.error.is_none());
    assert!(
        result
            .streams
            .iter()
            .any(|entry| entry.stream == ExecutionStreamKind::ClearScreen),
        "expected clear-screen control event in execution streams"
    );
}

#[test]
fn plotting_command_gap_repros_execute() {
    let _plot_guard = isolate_plot_test();
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    let result = runmat_core::execute_text_request_for_testing(
        &mut engine,
        "figure; plot([1 2], [3 4]); axis off; axis on; axis manual; axis ij; axis xy; drawnow;",
    )
    .unwrap();
    if let Some(error) = result.error {
        assert_eq!(error.identifier(), Some("RunMat:plot:EngineError"));
    }
}

#[test]
fn figure_name_property_pairs_execute_through_core() {
    let _plot_guard = isolate_plot_test();
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    let result = runmat_core::execute_text_request_for_testing(
        &mut engine,
        "f = figure('Name', 'demo', 'NumberTitle', 'off'); assert(strcmp(get(f, 'Name'), 'demo')); assert(~get(f, 'NumberTitle'));",
    )
    .unwrap();
    assert!(
        result.error.is_none(),
        "unexpected execution error: {:?}",
        result.error
    );
}

#[test]
fn warning_command_form_executes() {
    let mut engine = gc_test_context(RunMatSession::new).unwrap();

    let result = runmat_core::execute_text_request_for_testing(
        &mut engine,
        "warning off all; warning on all;",
    )
    .unwrap();
    assert!(result.error.is_none());
}
