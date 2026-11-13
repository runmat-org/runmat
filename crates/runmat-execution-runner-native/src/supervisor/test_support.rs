//! Platform-independent exact durable program fixture; no shell driver or process launch.
use super::{ProgramBatchSubmission, MIN_RETENTION_MILLIS};
use runmat_execution::resource::ResourceRequest;
use runmat_execution::{
    Digest, OutputContract, ProgramCallable, ProgramEnvironment, ProgramFunctionId,
    ProgramInvocationContext, ProgramRevision,
};
use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};

fn program_resources() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1024 * 1024,
        scratch_bytes: 1024 * 1024,
        max_wall_millis: 60_000,
        max_artifact_bytes: 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: Default::default(),
    }
}

pub(super) fn original_program_submission(
    context: super::DurableExecutionContext,
) -> ProgramBatchSubmission {
    let revision = ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
            runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: "7".into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("test-interpreter-bytecode-v2"),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV2,
        b"exact-program".to_vec(),
    )
    .unwrap();
    let callable = ProgramCallable::semantic(ProgramFunctionId(7), None);
    let task = runmat_execution_runner::TaskSubmission {
        request: runmat_execution::task::TaskRequest {
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: runmat_execution::TaskId::derive(&[b"original-durable-program"]),
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                runmat_execution::identity::ArtifactId::derive(&[artifact.id.0.bytes()]),
            ),
            callable: runmat_execution::task::Callable::for_program("test", &callable),
            host: runmat_execution::host::ExecutionHostRequirement::portable(
                runmat_core::program_environment(runmat_core::CompatMode::Matlab),
                runmat_types::CapabilitySet(std::collections::BTreeSet::from([
                    runmat_types::CapabilityRequirement::HostRuntime,
                ])),
                runmat_types::InteropManifest::empty(),
                std::collections::BTreeSet::from([
                    runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                ]),
            )
            .unwrap(),
            invocation_context: ProgramInvocationContext::Direct,
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: 1,
                owned_work: None,
            },
            resources: program_resources(),
            retry: runmat_execution::task::RetryPolicy::Never,
            deadline_unix_millis: None,
        },
        dependencies: Default::default(),
        priority: 0,
    };
    ProgramBatchSubmission {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        recipe,
        artifact,
        callable,
        invocation_context: ProgramInvocationContext::Direct,
        arguments: Vec::new(),
        requested_outputs: 1,
        task,
        idempotency_key: Some("program-key".into()),
        retention_millis: MIN_RETENTION_MILLIS,
    }
}
