//! Original compiler products through real native task admission and worker sessions.
use super::*;
use runmat_execution::identity::ArtifactId;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{OutputContract, TaskId};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest, ProgramTarget,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_execution_runner::{DriverSnapshot, TaskSubmission};
use runmat_execution_runner_native::NativeProgramSession;

pub(super) async fn request(
    text: &str,
    selected: Option<&str>,
    outputs: u16,
) -> ProgramExecutionRequest {
    let directory = tempfile::tempdir().unwrap();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let source = runmat_core::ExecutableSource::frozen(
        "original-portable-worker",
        "outer.m",
        directory.path().join("outer.m").to_string_lossy(),
        text,
    );
    let unit = session.compile_executable_unit(source, None).await.unwrap();
    let envelope = unit.portable_envelope_for(selected).unwrap();
    let recipe = ProgramBuildRecipe {
        foreign_policy: None,
        accelerators: Vec::new(),
        interop: runmat_types::InteropManifest::empty(),
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: envelope.manifest.identity.entrypoint.clone(),
        outputs: OutputContract {
            owned_work: None,
            requested_outputs: outputs,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("original-session-worker"),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: vec![],
        expected_artifact_id: None,
    };
    ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        callable: runmat_execution::ProgramCallable::semantic(
            envelope
                .manifest
                .identity
                .entrypoint_function
                .expect("selected semantic entrypoint"),
            None,
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        artifact: ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::ExecutableUnitV4,
            envelope.canonical_bytes().unwrap(),
        )
        .unwrap(),
        recipe,
        arguments: vec![],
        requested_outputs: outputs,
    }
}

async fn execute(
    config: &NativeExecutionConfig,
    request: ProgramExecutionRequest,
    expected: Vec<Value>,
) {
    let session = NativeProgramSession::new(config.clone()).unwrap();
    let original = request.clone();
    let request = serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
    let task_id = TaskId::derive(&[original.canonical_digest().unwrap().bytes()]);
    let task = session
        .submit(
            request,
            TaskSubmission {
                request: TaskRequest {
                    schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
                    console: None,
                    environment: original.environment.clone(),
                    id: task_id,
                    scope_id: session.scope_id(),
                    pool_id: session.pool_id(),
                    executable: runmat_execution::task::TaskExecutableIdentity::ProgramArtifact(
                        ArtifactId::derive(&[original.artifact.id.0.bytes()]),
                    ),
                    callable: Callable::for_program("original-portable-worker", &original.callable),
                    host: original
                        .artifact
                        .execution_host_requirement(
                            &original.recipe,
                            BTreeSet::from([
                                runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                            ]),
                        )
                        .unwrap(),
                    invocation_context: original.context.clone(),
                    inputs: original.arguments.clone(),
                    outputs: original.recipe.outputs.clone(),
                    resources: ResourceRequest {
                        cpu_millicores: config.worker_cpu_millicores,
                        memory_bytes: config.worker_memory_bytes,
                        scratch_bytes: config.worker_scratch_bytes,
                        max_wall_millis: 30_000,
                        max_artifact_bytes: 1024 * 1024,
                        max_egress_bytes: 0,
                        max_relay_bytes: 0,
                        accelerators: vec![],
                        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
                    },
                    retry: RetryPolicy::Never,
                    deadline_unix_millis: None,
                },
                dependencies: BTreeSet::new(),
                priority: 0,
            },
        )
        .unwrap();
    let completion = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Some(completion) = task.try_completion() {
                break completion;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("original portable worker did not complete");
    let result = completion.delivery_result().unwrap();
    assert_eq!(
        result
            .values()
            .expect("portable value result")
            .0
            .iter()
            .map(
                |value| runmat_runtime::execution::value_codec::decode_inline_value(value).unwrap()
            )
            .collect::<Vec<_>>(),
        expected
    );
    let evidence = completion.execution_evidence().unwrap();
    evidence.validate().unwrap();
    assert_eq!(
        evidence.task.submission.request.environment,
        original.environment
    );
    assert_eq!(evidence.task.submission.request.inputs, original.arguments);
    assert_eq!(
        evidence.task.submission.request.outputs,
        original.recipe.outputs
    );
    assert_eq!(
        evidence
            .task
            .committed
            .as_ref()
            .unwrap()
            .result
            .values()
            .expect("committed value result")
            .0,
        result.values().expect("portable value result").0
    );
    let root = config.store_root.join(session.scope_id().to_string());
    let cold: DriverSnapshot =
        serde_json::from_slice(&std::fs::read(root.join("checkpoints/driver.json")).unwrap())
            .unwrap();
    cold.validate().unwrap();
    assert_eq!(cold.task_execution_evidence(task_id).unwrap(), *evidence);
}

async fn transitive_request() -> ProgramExecutionRequest {
    let mut producer = runmat_core::RunMatSession::with_options(false, false).unwrap();
    for (name, source) in [
        ("leaf.m", "function y = originalLeaf(x); y = x + 7; end"),
        (
            "middle.m",
            "function y = originalMiddle(x); y = originalLeaf(x) * 2; end",
        ),
    ] {
        let request = runmat_core::abi::ExecutionRequest::for_source(
            runmat_core::abi::SourceInput::Text {
                name: name.into(),
                text: source.into(),
            },
            Default::default(),
            runmat_core::abi::HostExecutionPolicy::default(),
            runmat_core::abi::WorkspaceHandle(uuid::Uuid::nil()),
        );
        producer.execute_request(request).await.result.unwrap();
    }
    let unit = producer
        .compile_executable_unit(
            runmat_core::ExecutableSource::new(
                "original-portable-worker",
                "consumer.m",
                "function y = consumer(x); y = originalMiddle(x) + 3; end",
            ),
            None,
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope_for(Some("consumer")).unwrap();
    assert_eq!(envelope.dependency_products.len(), 2);
    drop(producer);
    let (recipe, artifact) = ProgramArtifact::materialize_executable_unit(
        &envelope,
        OutputContract {
            owned_work: None,
            requested_outputs: 1,
        },
        ProgramTarget::portable("original-session-worker"),
    )
    .unwrap();
    ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        callable: runmat_execution::ProgramCallable::semantic(
            envelope
                .manifest
                .identity
                .entrypoint_function
                .expect("selected semantic entrypoint"),
            None,
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: vec![
            runmat_runtime::execution::value_codec::encode_inline_value(&Value::Num(5.0)).unwrap(),
        ],
        requested_outputs: 1,
        recipe,
        artifact,
    }
}

pub(super) async fn verify(config: &NativeExecutionConfig) {
    // The source session is already gone when this request reaches the actual worker.
    // Its snapshot, including absent HOME, survives task storage and cold receipts.
    let mut environment = request(
        "function [value, absent] = originalEnvironment(); value = getenv('RUNMAT_ORIGINAL_WORKER'); absent = getenv('HOME'); setenv('RUNMAT_ORIGINAL_WORKER','worker-local'); end",
        Some("originalEnvironment"), 2,
    ).await;
    environment.environment = runmat_execution::EnvironmentSnapshot::new(vec![(
        "RUNMAT_ORIGINAL_WORKER".into(),
        "captured".into(),
    )])
    .unwrap();
    let host = std::env::var_os("RUNMAT_ORIGINAL_WORKER");
    execute(
        config,
        environment,
        vec![
            Value::CharArray(runmat_value::CharArray::new_row("captured")),
            Value::CharArray(runmat_value::CharArray::new_row("")),
        ],
    )
    .await;
    assert_eq!(std::env::var_os("RUNMAT_ORIGINAL_WORKER"), host);

    execute(config, transitive_request().await, vec![Value::Num(27.0)]).await;
    for outputs in [0, 1, 2] {
        let mut original = request(
            concat!(
                "function y = outer(); y = 999; end\n",
                "function [y,z] = selected(x)\ng = @(t) helper(t) + x;\n",
                "y = g(3); z = helper(x); end\n",
                "function y = helper(x); y = 2*x; end\n",
            ),
            Some("selected"),
            outputs,
        )
        .await;
        assert_ne!(
            original
                .callable
                .semantic_function()
                .expect("selected semantic function")
                .0,
            0
        );
        original.arguments =
            vec![
                runmat_runtime::execution::value_codec::encode_inline_value(&Value::Num(10.0))
                    .unwrap(),
            ];
        execute(
            config,
            original,
            vec![Value::Num(16.0), Value::Num(20.0)][..usize::from(outputs)].to_vec(),
        )
        .await;
    }
    execute(
        config,
        request("ans = 42; unrelated = 999;", None, 1).await,
        vec![Value::Num(42.0)],
    )
    .await;
    execute(config, request("unrelated = 999;", None, 0).await, vec![]).await;
    println!("original V4 captures, exact semantic functions and script output authority passed real native tasks and cold checkpoints");
}
