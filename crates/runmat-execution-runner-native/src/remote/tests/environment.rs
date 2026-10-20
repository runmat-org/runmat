use super::*;

#[tokio::test]
async fn encrypted_driver_uses_exact_environment_after_source_teardown_without_host_mutation() {
    let name = "RUNMAT_BUNDLE_ENV132";
    let host = std::env::var_os(name);
    let source = "function tests = remoteTest()\n tests = functiontests(localfunctions);\nend\nfunction testEnvironment(testCase)\n testCase.verifyEqual(getenv('RUNMAT_BUNDLE_ENV132'), 'frozen');\n testCase.verifyEmpty(getenv('HOME'));\n setenv('RUNMAT_BUNDLE_ENV132', 'local-change');\n testCase.verifyEqual(getenv('RUNMAT_BUNDLE_ENV132'), 'local-change');\nend\n";
    let (root, project, revision) = project_fixture(source);
    let snapshot = FrozenTestRunSnapshot::freeze(
        revision.graph_digest().to_string(),
        revision.source_digest().to_string(),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        Digest::sha256(b"frozen-environment").to_string(),
        vec![SavedRunSource {
            owner_identity: "path:workspace".into(),
            relative_path: "src/remoteTest.m".into(),
            content: source.into(),
        }],
        Vec::new(),
    )
    .unwrap();
    let session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let prepared = session
        .prepare_tests(&snapshot, &TestSelector::default())
        .unwrap();
    let test_id = prepared.plan.tests().next().unwrap().id.clone();
    let workload = TestAttemptWorkload::new(
        RunSubmission::new(prepared.plan, snapshot).unwrap(),
        test_id,
        1,
    )
    .unwrap();
    let request = workload
        .program_request(runmat_types::InteropManifest::empty())
        .unwrap();
    let bundle = ExecutionBundleBuilder::native(&project, request.recipe.program_revision.clone())
        .unwrap()
        .with_materialized_program(
            request.recipe.clone(),
            ExecutableForm::TestAttemptV1,
            request.artifact.executable_bytes.clone(),
        )
        .with_portable_environment(vec![(name.into(), "frozen".into())])
        .unwrap()
        .build()
        .unwrap();
    root.close().unwrap();
    for run in [
        "run-frozen-environment-first",
        "run-frozen-environment-second",
    ] {
        let response = run_encrypted_remote_request(run, bundle.clone()).await;
        let ProgramExecutionResponse::Success { mut outputs } = response else {
            panic!("driver did not return its original test result: {response:?}")
        };
        assert_eq!(outputs.len(), 1, "one original semantic output");
        let value = outputs.remove(0);
        let execution = decode_execution(&value).unwrap();
        assert_eq!(
            execution.result.state.disposition,
            TerminalDisposition::Passed,
            "{execution:#?}"
        );
        assert_eq!(std::env::var_os(name), host);
    }
}
