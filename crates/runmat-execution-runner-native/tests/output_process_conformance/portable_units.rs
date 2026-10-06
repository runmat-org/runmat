//! Original compiler products through real native task admission and worker sessions.
use super::*;
use runmat_execution::identity::ArtifactId;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{Digest, OutputContract, TaskId};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest, ProgramTarget,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner::{DriverSnapshot, TaskSubmission};
use runmat_execution_runner_native::NativeProgramSession;

async fn request(text: &str, selected: Option<&str>, outputs: u16) -> ProgramExecutionRequest {
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
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: envelope.manifest.identity.entrypoint_function.0.to_string(),
        outputs: OutputContract {
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
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        function: envelope.manifest.identity.entrypoint_function.0 as usize,
        artifact: ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::ExecutableUnitV3,
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
                    id: task_id,
                    scope_id: session.scope_id(),
                    pool_id: session.pool_id(),
                    program_artifact_id: ArtifactId::derive(&[original.artifact.id.0.bytes()]),
                    callable: Callable {
                        owner_identity: "original-portable-worker".into(),
                        qualified_name: original.recipe.entrypoint.clone(),
                        entrypoint_digest: Digest::sha256(original.function.to_be_bytes()),
                    },
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
            .outputs
            .iter()
            .map(
                |value| runmat_runtime::execution::value_codec::decode_inline_value(value).unwrap()
            )
            .collect::<Vec<_>>(),
        expected
    );
    let evidence = completion.execution_evidence().unwrap();
    evidence.validate().unwrap();
    assert_eq!(evidence.task.submission.request.inputs, original.arguments);
    assert_eq!(
        evidence.task.submission.request.outputs,
        original.recipe.outputs
    );
    assert_eq!(
        evidence.task.committed.as_ref().unwrap().outputs,
        result.outputs
    );
    let root = config.store_root.join(session.scope_id().to_string());
    let cold: DriverSnapshot =
        serde_json::from_slice(&std::fs::read(root.join("checkpoints/driver.json")).unwrap())
            .unwrap();
    cold.validate().unwrap();
    assert_eq!(cold.task_execution_evidence(task_id).unwrap(), *evidence);
}

pub(super) async fn verify(config: &NativeExecutionConfig) {
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
        assert_ne!(original.function, 0);
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
    println!("original V3 captures, exact semantic functions and script output authority passed real native tasks and cold checkpoints");
}
