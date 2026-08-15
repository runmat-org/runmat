use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use runmat_core::{
    ExecutableSource, InvocationControl, ProcedureInvocation, RunError, RunMatSession,
};
use runmat_test::coverage::CoverageMetric;
use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
use runmat_test::event::TestEventPayload;
use runmat_test::result::TerminalDisposition;

fn digest(value: &str) -> String {
    runmat_execution::Digest::sha256(value).to_string()
}

async fn execute_portable_envelope(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
) -> runmat_execution_artifact::ProgramExecutionResponse {
    let function = usize::try_from(envelope.manifest.identity.entrypoint_function.0).unwrap();
    let recipe = runmat_execution_artifact::ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: function.to_string(),
        outputs: runmat_execution::OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable(
            "portable-executable-unit-v3-test",
        ),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = runmat_execution_artifact::ProgramArtifact::materialize(
        &recipe,
        runmat_execution_artifact::ExecutableForm::ExecutableUnitV3,
        envelope.canonical_bytes().unwrap(),
    )
    .unwrap();
    runmat_vm::execute_program_request(runmat_execution_artifact::ProgramExecutionRequest {
        schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe,
        artifact,
        function,
        arguments: Vec::new(),
        requested_outputs: 1,
    })
    .await
}

fn conformance_snapshot() -> FrozenTestRunSnapshot {
    FrozenTestRunSnapshot::freeze(
        digest("conformance-graph"),
        "sha256:conformance-sources",
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        digest("conformance-config"),
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
        digest("fixture-graph"),
        "sha256:fixture-sources",
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        digest("fixture-config"),
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
        vec![TerminalDisposition::Passed, TerminalDisposition::Failed],
        "{run:#?}"
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

    for iteration in 0..12 {
        let run = session
            .run_test_snapshot(&snapshot, &TestSelector::default())
            .await
            .unwrap();
        assert_eq!(run.results.len(), 2, "iteration {iteration}: {run:#?}");
        assert_eq!(
            run.results[0].state.disposition,
            TerminalDisposition::Passed,
            "iteration {iteration}: {run:#?}"
        );
        assert_eq!(
            run.results[1].state.disposition,
            TerminalDisposition::Failed,
            "iteration {iteration}: {run:#?}"
        );
        assert_eq!(run.results[1].attempts[0].diagnostics.len(), 1, "{run:#?}");
    }

    assert!(
        session.stats().jit_compiled > 0,
        "the immutable procedure call frame never reached adaptive native execution: {:?}",
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
async fn executable_unit_retains_exact_program_point_analysis() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new(
                "path:analysis-store",
                "analyzed.m",
                "function y = analyzed(); y = (1 + 2) * 3; end\n",
            ),
            None,
        )
        .await
        .unwrap();

    assert!(!unit.analysis().program_points.is_empty());
    assert_eq!(unit.analysis().functions.len(), 1);
    assert!(!unit.analysis().dependencies.is_empty());
    assert!(!unit.analysis().regions.is_empty());
    assert_eq!(unit.mir().bodies.len(), 1);
    assert_eq!(unit.vm_layout().functions.len(), 1);
    assert!(!unit
        .revision()
        .program_revision
        .canonical_identity()
        .is_empty());
    let envelope = unit.portable_envelope().unwrap();
    let analysis_region_ids = unit
        .analysis()
        .regions
        .iter()
        .map(|region| region.contract.id)
        .collect::<Vec<_>>();
    assert_eq!(
        envelope
            .manifest
            .regions
            .iter()
            .map(|region| region.id)
            .collect::<Vec<_>>(),
        analysis_region_ids
    );
    assert_eq!(
        envelope.manifest.revisions.bytecode_schema,
        runmat_vm::BYTECODE_SCHEMA_VERSION
    );
    let bytecode: runmat_vm::Bytecode = serde_json::from_slice(
        &envelope
            .component(runmat_execution::ExecutableComponentKind::Bytecode)
            .unwrap()
            .bytes,
    )
    .unwrap();
    assert_eq!(
        bytecode
            .regions
            .iter()
            .map(|region| region.id)
            .collect::<Vec<_>>(),
        analysis_region_ids
    );
    for region in &bytecode.regions {
        assert_eq!(region.entry.point.function, region.id.function);
        assert!(region
            .exits
            .iter()
            .all(|exit| exit.point.function == region.id.function));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let binding_names = unit.binding_names();
        let native =
            runmat_native_codegen::lower_executable(runmat_native_codegen::NativeLoweringInput {
                mir: unit.mir(),
                analysis: unit.analysis(),
                manifest: &envelope.manifest,
                binding_names: Some(&binding_names),
                target: runmat_native_codegen::NativeTarget::current(),
            })
            .unwrap();
        assert_eq!(native.requirements.regions, envelope.manifest.regions);
        let native_region_ids = native
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.region_boundaries)
            .filter(|boundary| {
                matches!(
                    boundary.kind,
                    runmat_native_codegen::NativeRegionBoundaryKind::Entry
                )
            })
            .map(|boundary| boundary.region)
            .collect::<Vec<_>>();
        assert_eq!(native_region_ids, analysis_region_ids);
    }
    let bytes = envelope.canonical_bytes().unwrap();
    assert_eq!(
        runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(&bytes).unwrap(),
        envelope
    );
    let response = execute_portable_envelope(&envelope).await;
    assert!(matches!(
        response,
        runmat_execution_artifact::ProgramExecutionResponse::Success { .. }
    ));
    let mut duplicate_authority = envelope.clone();
    let registry: runmat_vm::FunctionRegistry = serde_json::from_slice(
        &duplicate_authority
            .component(runmat_execution::ExecutableComponentKind::FunctionRegistry)
            .unwrap()
            .bytes,
    )
    .unwrap();
    let bytecode = duplicate_authority
        .payloads
        .iter_mut()
        .find(|payload| payload.kind == runmat_execution::ExecutableComponentKind::Bytecode)
        .unwrap();
    let mut decoded: runmat_vm::Bytecode = serde_json::from_slice(&bytecode.bytes).unwrap();
    decoded.function_registry = registry;
    bytecode.bytes = serde_json::to_vec(&serde_json::to_value(decoded).unwrap()).unwrap();
    let descriptor = duplicate_authority
        .manifest
        .components
        .iter_mut()
        .find(|descriptor| descriptor.kind == runmat_execution::ExecutableComponentKind::Bytecode)
        .unwrap();
    let kind = descriptor.kind;
    let schema_version = descriptor.schema_version;
    *descriptor = runmat_execution::ExecutableComponentDescriptor::from_payload(
        kind,
        schema_version,
        &bytecode.bytes,
    )
    .unwrap();
    assert!(matches!(
        execute_portable_envelope(&duplicate_authority).await,
        runmat_execution_artifact::ProgramExecutionResponse::Failure { message }
            if message.contains("duplicate component authorities")
    ));
    let source_map = envelope
        .payloads
        .iter()
        .find(|payload| payload.kind == runmat_execution::ExecutableComponentKind::SourceMap)
        .unwrap();
    let source_map = String::from_utf8(source_map.bytes.clone()).unwrap();
    assert!(!source_map.contains("full_path"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn complete_portable_script_executes_after_component_reconstruction() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new("path:portable-script", "script.m", "answer = 42;\n"),
            None,
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope().unwrap();
    assert_eq!(
        envelope.manifest.identity.entrypoint_kind,
        runmat_execution::ExecutableEntrypointKind::Script
    );
    assert!(matches!(
        execute_portable_envelope(&envelope).await,
        runmat_execution_artifact::ProgramExecutionResponse::Success { .. }
    ));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn loose_executable_revision_is_stable_and_source_sensitive() {
    let source = ExecutableSource::new(
        "path:loose",
        "stable.m",
        "function y = stable(); y = 1; end\n",
    );
    let mut first = RunMatSession::with_options(false, false).unwrap();
    let first = first
        .compile_executable_unit(source.clone(), None)
        .await
        .unwrap();
    let mut second = RunMatSession::with_options(false, false).unwrap();
    let second = second.compile_executable_unit(source, None).await.unwrap();
    assert_eq!(first.revision(), second.revision());

    let environment = second.revision().program_revision.environment();
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let changed = session
        .compile_executable_unit(
            ExecutableSource::new(
                "path:loose",
                "stable.m",
                "function y = stable(); y = 2; end\n",
            ),
            None,
        )
        .await
        .unwrap();
    assert_ne!(first.revision(), changed.revision());
    assert_eq!(
        changed.revision().program_revision.environment(),
        environment
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn portable_product_retains_names_for_lexical_and_session_bindings() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new(
                "path:bindings",
                "bindings.m",
                "function y = bindings(x)\n global shared\n y = x + shared;\nend\n",
            ),
            None,
        )
        .await
        .unwrap();
    let names = unit.binding_names();
    assert!(names.values().any(|name| name == "x"));
    assert!(names.values().any(|name| name == "y"));
    assert!(names.values().any(|name| name == "shared"));

    let envelope = unit.portable_envelope_for(Some("bindings")).unwrap();
    assert_eq!(envelope.manifest.revisions.vm_layout_schema, 3);
    assert_eq!(
        envelope.manifest.revisions.function_registry_schema,
        runmat_vm::FUNCTION_REGISTRY_SCHEMA_VERSION
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
async fn coverage_sites_are_source_stable_and_record_missed_statements() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new(
                "path:conformance",
                "covered.m",
                "function y = covered(x)\n y = 1;\n if x > 0\n  y = 2;\n else\n  y = 3;\n end\nend\n",
            ),
            None,
        )
        .await
        .unwrap();
    let (_, fragment) = session
        .invoke_executable_with_coverage(
            &unit,
            ProcedureInvocation::function("covered", vec![runmat_value::Value::Num(1.0)]),
            &InvocationControl::default(),
        )
        .await
        .unwrap();

    assert!(fragment
        .sites
        .iter()
        .any(|site| site.metric == CoverageMetric::Function
            && fragment.counts.get(&site.counter_key).copied().unwrap_or(0) == 1));
    let statements = fragment
        .sites
        .iter()
        .filter(|site| site.metric == CoverageMetric::Statement && site.instrumented)
        .collect::<Vec<_>>();
    assert!(statements.len() >= 3, "{statements:#?}");
    assert!(statements.iter().any(|site| fragment
        .counts
        .get(&site.counter_key)
        .copied()
        .unwrap_or(0)
        > 0));
    assert!(statements.iter().any(|site| fragment
        .counts
        .get(&site.counter_key)
        .copied()
        .unwrap_or(0)
        == 0));
    assert!(fragment
        .sites
        .iter()
        .all(|site| site.relative_path == "covered.m"
            && site.start_line > 0
            && site.start_column > 0));
}

#[cfg(all(feature = "jit", not(target_arch = "wasm32")))]
#[tokio::test]
async fn jit_and_interpreter_hit_the_same_coverage_sites() {
    let source = ExecutableSource::new(
        "path:conformance",
        "coveredJit.m",
        "function y = coveredJit(x)\n y = x + 1;\n if y > 0\n  y = y * 2;\n end\nend\n",
    );
    let mut interpreter = RunMatSession::with_options(false, false).unwrap();
    let interpreter_unit = interpreter
        .compile_executable_unit(source.clone(), None)
        .await
        .unwrap();
    let (_, expected) = interpreter
        .invoke_executable_with_coverage(
            &interpreter_unit,
            ProcedureInvocation::function("coveredJit", vec![runmat_value::Value::Num(1.0)]),
            &InvocationControl::default(),
        )
        .await
        .unwrap();

    let mut jit = RunMatSession::with_options(true, false).unwrap();
    let jit_unit = jit.compile_executable_unit(source, None).await.unwrap();
    let mut actual = None;
    for _ in 0..12 {
        actual = Some(
            jit.invoke_executable_with_coverage(
                &jit_unit,
                ProcedureInvocation::function("coveredJit", vec![runmat_value::Value::Num(1.0)]),
                &InvocationControl::default(),
            )
            .await
            .unwrap()
            .1,
        );
    }
    assert!(jit.stats().jit_compiled > 0, "{:?}", jit.stats());
    assert_eq!(actual.unwrap(), expected);
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
