use super::*;
use runmat_runtime::context::{IsolatedMemoryPreferenceService, RuntimeContext};
use runmat_runtime::execution::RuntimeExecutionService;
use std::collections::BTreeSet;
use std::rc::Rc;

#[test]
fn project_backed_test_discovery_and_execution_share_parent_driver() {
    let _guard = cwd_lock();
    let fixture = tempfile::TempDir::new().expect("project fixture");
    let _cwd = push_cwd(fixture.path());
    std::fs::create_dir(fixture.path().join("src")).expect("source root");
    std::fs::write(
        fixture.path().join("runmat.toml"),
        "[package]\nname = \"project-tests\"\n[sources]\nroots = [\"src\"]\n",
    )
    .expect("manifest");
    std::fs::write(
        fixture.path().join("src/testProject.m"),
        "assert(getpref('ProjectTest', 'Marker') == 29); assert(strcmp(userpath(), 'relative/Project'));",
    )
    .expect("test source");
    let frozen =
        runmat_package::build_frozen_project(&fixture.path().join("runmat.toml"), BTreeSet::new())
            .expect("frozen project");
    let mut parent = RunMatSession::with_options(false, false).expect("parent session");
    // The relative userpath fixture intentionally exercises RunMat's partial
    // path extension; MATLAB compatibility mode rejects that syntax.
    parent.set_compat_mode(runmat_parser::CompatMode::RunMat);
    parent
        .install_application_settings(Rc::new(IsolatedMemoryPreferenceService::new()))
        .expect("explicit application settings authority");
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("host reactor");
    let revision = host
        .block_on(parent.install_project_handoff(runmat_package::FrozenProjectHandoff::new(frozen)))
        .expect("install project before testing");

    let outcome = {
        let _entered = host.enter();
        execute_text_request(
            &mut parent,
            "setpref('ProjectTest', 'Marker', 29); userpath('relative/Project'); suite = testsuite('src/testProject.m'); results = runtests('src/testProject.m');",
        )
        .expect("project-backed testing executes on the active parent driver")
    };
    assert!(
        outcome.diagnostics.is_empty(),
        "project-backed testing diagnostics: {:?}",
        outcome.diagnostics
    );
    let suite = outcome_named_upsert_value(&outcome, "suite").expect("discovered suite");
    let runmat_value::Value::Object(suite) = suite else {
        panic!("expected scalar TestSuite, got {suite:?}");
    };
    assert!(suite.is_class(runmat_runtime::testing::TEST_SUITE_CLASS));
    let results = outcome_named_upsert_value(&outcome, "results").expect("test result");
    let runmat_value::Value::Object(result) = results else {
        panic!("expected scalar TestResult, got {results:?}");
    };
    assert!(result.is_class(runmat_runtime::testing::TEST_RESULT_CLASS));
    assert_eq!(
        result.properties().get("Passed"),
        Some(&runmat_value::Value::Bool(true))
    );
    assert_eq!(parent.project_revision(), Some(revision));
    assert!(futures::executor::block_on(
        parent.local_task_driver().participate_foreground(async {})
    )
    .is_ok());
}

#[test]
fn test_child_does_not_discover_settings_when_caller_has_no_authority() {
    let parent = RunMatSession::with_options(false, false).expect("parent driver");
    let bare_caller = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let mut child = RunMatSession::with_options_on_parent_driver(
        false,
        false,
        &parent.local_task_driver().child_scope_factory(),
        &bare_caller,
    )
    .expect("child construction preserves absence");
    let error = execute_text_request(&mut child, "value = 1;")
        .expect_err("source resolution must refuse absent caller user-path authority");
    assert!(
        error
            .to_string()
            .contains("application user-path settings service"),
        "unexpected authority error: {error}"
    );
    assert!(futures::executor::block_on(
        parent.local_task_driver().participate_foreground(async {})
    )
    .is_ok());
}
