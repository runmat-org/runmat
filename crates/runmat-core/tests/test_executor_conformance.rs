use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use runmat_core::{
    ExecutableSource, InvocationControl, ProcedureInvocation, RunError, RunMatSession,
};
use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
use runmat_test::event::TestEventPayload;
use runmat_test::result::TerminalDisposition;

fn conformance_snapshot() -> FrozenTestRunSnapshot {
    FrozenTestRunSnapshot::freeze(
        "sha256:conformance-graph",
        "sha256:conformance-sources",
        1,
        1,
        "sha256:conformance-config",
        vec![SavedRunSource {
            owner_identity: "path:conformance".into(),
            relative_path: "portableTest.m".into(),
            content: "function tests = portableTest()\n tests = functiontests(localfunctions);\nend\nfunction testPasses(testCase)\n testCase.verifyEqual(2 + 2, 4);\nend\nfunction testFails(testCase)\n testCase.verifyEqual(2 + 2, 5, 'portable failure');\nend\n".into(),
        }],
        Vec::new(),
    )
    .unwrap()
}

fn fixture_snapshot() -> FrozenTestRunSnapshot {
    FrozenTestRunSnapshot::freeze(
        "sha256:fixture-graph",
        "sha256:fixture-sources",
        1,
        1,
        "sha256:fixture-config",
        vec![SavedRunSource {
            owner_identity: "path:conformance".into(),
            relative_path: "fixtureTest.m".into(),
            content: "function tests = fixtureTest()\n tests = functiontests(localfunctions);\nend\nfunction setup(testCase)\n disp('setup');\n testCase.addTeardown(@cleanup);\nend\nfunction testBody(testCase)\n disp('body');\n testCase.verifyTrue(true);\nend\nfunction cleanup()\n disp('dynamic');\nend\nfunction teardown(testCase)\n disp('teardown');\nend\n".into(),
        }],
        Vec::new(),
    )
    .unwrap()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn core_executor_has_the_same_portable_lifecycle_contract() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let snapshot = conformance_snapshot();
    let discovery = session.discover_tests(&snapshot).unwrap();
    assert_eq!(discovery.suites.len(), 1, "{discovery:#?}");
    let run = session
        .run_test_snapshot(&snapshot, &TestSelector::default())
        .await
        .unwrap();

    assert_eq!(run.results.len(), 2, "{run:#?}");
    assert_eq!(
        run.results
            .iter()
            .map(|result| result.state.disposition)
            .collect::<Vec<_>>(),
        vec![TerminalDisposition::Passed, TerminalDisposition::Failed]
    );
    assert_eq!(
        run.events
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>(),
        (0..run.events.len() as u64).collect::<Vec<_>>()
    );
    assert!(matches!(
        run.events.first().map(|event| &event.payload),
        Some(TestEventPayload::RunStarted)
    ));
    assert!(matches!(
        run.events.last().map(|event| &event.payload),
        Some(TestEventPayload::RunFinished { .. })
    ));
    let failed = run
        .results
        .iter()
        .find(|result| result.state.failed)
        .expect("one conformance case fails");
    assert!(failed.attempts[0].diagnostics[0]
        .message
        .contains("portable failure"));
}

#[cfg(all(feature = "jit", not(target_arch = "wasm32")))]
#[tokio::test]
async fn jit_tiering_preserves_the_portable_lifecycle_contract() {
    let mut session = RunMatSession::with_options(true, false).unwrap();
    let snapshot = conformance_snapshot();

    for _ in 0..12 {
        let run = session
            .run_test_snapshot(&snapshot, &TestSelector::default())
            .await
            .unwrap();
        assert_eq!(run.results.len(), 2, "{run:#?}");
        assert_eq!(
            run.results[0].state.disposition,
            TerminalDisposition::Passed,
            "{run:#?}"
        );
        assert_eq!(
            run.results[1].state.disposition,
            TerminalDisposition::Failed,
            "{run:#?}"
        );
        assert_eq!(run.results[1].attempts[0].diagnostics.len(), 1, "{run:#?}");
    }

    assert!(
        session.stats().jit_compiled > 0,
        "the immutable procedure call frame never reached Turbine: {:?}",
        session.stats()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn executable_invocation_observes_cancellation_and_deadlines() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new(
                "path:conformance",
                "controlledTest.m",
                "function controlledTest()\nend\n",
            ),
            None,
        )
        .await
        .unwrap();
    let cancelled = Arc::new(AtomicBool::new(true));
    let cancellation_error = session
        .invoke_executable(
            &unit,
            ProcedureInvocation::function("controlledTest", Vec::new()),
            &InvocationControl::default().with_cancellation(cancelled.clone()),
        )
        .await
        .unwrap_err();
    assert_eq!(
        runtime_identifier(&cancellation_error),
        Some("RunMat:ExecutionCancelled")
    );

    cancelled.store(false, Ordering::Relaxed);
    let deadline_error = session
        .invoke_executable(
            &unit,
            ProcedureInvocation::function("controlledTest", Vec::new()),
            &InvocationControl::default().with_deadline_unix_ms(0),
        )
        .await
        .unwrap_err();
    assert_eq!(
        runtime_identifier(&deadline_error),
        Some("RunMat:ExecutionDeadline")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn declared_fixtures_and_dynamic_teardowns_have_portable_ordering() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let snapshot = fixture_snapshot();
    let discovery = session.discover_tests(&snapshot).unwrap();
    assert_eq!(discovery.suites.len(), 1, "{discovery:#?}");
    assert_eq!(discovery.suites[0].fixtures.len(), 1, "{discovery:#?}");

    let run = session
        .run_test_snapshot(&snapshot, &TestSelector::default())
        .await
        .unwrap();
    assert_eq!(run.results.len(), 1, "{run:#?}");
    assert_eq!(
        run.results[0].state.disposition,
        TerminalDisposition::Passed,
        "{run:#?}"
    );
    let output = &run.results[0].attempts[0].output;
    let setup = output.find("setup").expect("setup output");
    let body = output.find("body").expect("body output");
    let dynamic = output.find("dynamic").expect("dynamic teardown output");
    let teardown = output.find("teardown").expect("declared teardown output");
    assert!(
        setup < body && body < dynamic && dynamic < teardown,
        "{output:?}"
    );
}

fn runtime_identifier(error: &RunError) -> Option<&str> {
    match error {
        RunError::Runtime(error) => error.identifier(),
        _ => None,
    }
}
