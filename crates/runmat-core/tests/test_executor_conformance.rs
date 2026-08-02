use runmat_core::RunMatSession;
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
