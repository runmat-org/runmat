use std::collections::BTreeSet;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use runmat_analysis_execution::{
    build_analysis_task_submission, import_structural_linear_result,
    prepare_structural_linear_program, AnalysisExecutionContext,
    StructuralLinearProgramPreparation,
};
use runmat_analysis_fea::execution::{StructuralLinearWorkload, DEFAULT_ANALYSIS_CHUNK_BYTES};
use runmat_analysis_fea::FeaRunResult;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValuePayload;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ProgramExecutionResponse};
use runmat_execution_runner_native::{NativeExecutionConfig, NativeProgramSession};
use runmat_meshing_core::SolverMeshArtifact;
use runmat_runtime::analysis::{
    analysis_study_content_digest, AnalysisStudySpec, StructuralLinearExecutor,
};

const POLL_INTERVAL: Duration = Duration::from_millis(10);

pub(super) struct LocalProcessStructuralExecutor {
    study_digest: runmat_execution::Digest,
    program_revision: ProgramRevision,
}

impl LocalProcessStructuralExecutor {
    pub(super) fn new(study: &AnalysisStudySpec, config: &RunMatRuntimeConfig) -> Result<Self> {
        let study_digest = analysis_study_content_digest(study);
        let program_revision = ProgramRevision::new(
            study_digest,
            study_digest,
            runmat_core::program_environment(crate::diagnostics::parser_compat(
                config.language.compat,
            )),
        )?;
        Ok(Self {
            study_digest,
            program_revision,
        })
    }

    fn access(&self) -> ArtifactAccess {
        ArtifactAccess {
            authorization_scope: format!("study:{}", self.study_digest),
            encryption_context: runmat_execution::Digest::sha256(
                [
                    b"runmat-local-analysis-access/v1\0".as_slice(),
                    self.study_digest.bytes(),
                ]
                .concat(),
            ),
        }
    }
}

impl StructuralLinearExecutor for LocalProcessStructuralExecutor {
    fn execute(
        &self,
        workload: &StructuralLinearWorkload,
        mesh: &SolverMeshArtifact,
    ) -> Result<FeaRunResult, String> {
        let limits = ObjectInventoryLimits::default();
        let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
            authored_study_digest: self.study_digest,
            frozen_project_digest: None,
            program_revision: self.program_revision.clone(),
            artifact_access: self.access(),
            workload: workload.clone(),
            mesh: mesh.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: limits,
            required_capabilities: BTreeSet::new(),
        })
        .map_err(|error| error.to_string())?;

        let mut task_capabilities = prepared
            .host
            .workload
            .generic_capabilities()
            .map_err(|error| error.to_string())?;
        task_capabilities.insert(Capability::ProcessIsolation);
        let mut native_config =
            NativeExecutionConfig::for_current_executable().map_err(|error| error.to_string())?;
        native_config
            .worker_capabilities
            .extend(task_capabilities.iter().cloned());
        let session =
            NativeProgramSession::new(native_config).map_err(|error| error.to_string())?;
        let mut store = session.object_store();
        prepared
            .program
            .write_to(&mut store)
            .map_err(|error| error.to_string())?;
        let roots = prepared
            .program
            .request
            .arguments
            .iter()
            .map(|argument| match argument {
                ValuePayload::Object(root) => Ok((**root).clone()),
                ValuePayload::Inline(_) => {
                    Err("portable analysis input was unexpectedly inline".to_string())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let maximum_wall_millis = workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_wall_time_milliseconds;
        let submission = build_analysis_task_submission(
            &prepared.host,
            roots,
            Vec::new(),
            BTreeSet::new(),
            AnalysisExecutionContext {
                scope_id: session.scope_id(),
                pool_id: session.pool_id(),
                program_artifact_id: ArtifactId::derive(&[prepared
                    .program
                    .request
                    .artifact
                    .id
                    .0
                    .bytes()]),
                resources: ResourceRequest {
                    cpu_millicores: 1_000,
                    memory_bytes: 1024 * 1024 * 1024,
                    scratch_bytes: 1024 * 1024 * 1024,
                    max_wall_millis: maximum_wall_millis,
                    max_artifact_bytes: limits.max_total_bytes,
                    max_egress_bytes: 0,
                    max_relay_bytes: 0,
                    accelerators: Vec::new(),
                    required_capabilities: task_capabilities,
                },
                retry: RetryPolicy::IdempotentInfrastructure,
                deadline_unix_millis: None,
                priority: 0,
            },
        )
        .map_err(|error| error.to_string())?;
        let task = session
            .submit(prepared.program.request.clone(), submission)
            .map_err(|error| error.to_string())?;
        let started = Instant::now();
        let completed = loop {
            if let Some(result) = task.try_result() {
                break result?;
            }
            if started.elapsed() >= Duration::from_millis(maximum_wall_millis) {
                session.cancel(runmat_execution::CancellationReason::Deadline);
                return Err(
                    "local structural analysis exceeded its authored wall-time limit".into(),
                );
            }
            thread::sleep(POLL_INTERVAL);
        };
        let response = ProgramExecutionResponse::ExternalizedSuccess {
            outputs: completed.outputs,
            result_objects: completed.result_objects,
        };
        import_structural_linear_result(&store, response, &prepared, limits)
            .map(|result| result.run)
            .map_err(|error| error.to_string())
    }
}
