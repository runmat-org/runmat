use std::collections::BTreeSet;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use runmat_analysis_execution::{
    build_analysis_task_submission, import_structural_linear_result,
    prepare_structural_linear_program, AnalysisExecutionContext, PreparedStructuralLinearProgram,
    StructuralLinearProgramPreparation,
};
use runmat_analysis_fea::execution::{StructuralLinearWorkload, DEFAULT_ANALYSIS_CHUNK_BYTES};
use runmat_analysis_fea::FeaRunResult;
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ProgramExecutionResponse};
use runmat_execution_runner_native::{
    FilesystemObjectStore, NativeExecutionConfig, NativeProgramSession, NativeProgramTask,
};
use runmat_meshing_core::SolverMeshArtifact;
use runmat_runtime::analysis::{
    analysis_study_content_digest, AnalysisStudySpec, PreparedStructuralStudy,
    StructuralLinearExecutor,
};

const POLL_INTERVAL: Duration = Duration::from_millis(10);

struct PreparedLocalProgram {
    prepared: PreparedStructuralLinearProgram,
    task_capabilities: BTreeSet<Capability>,
    maximum_wall_millis: u64,
}

pub(super) struct LocalSweepEntry<'a> {
    pub(super) study: &'a AnalysisStudySpec,
    pub(super) prepared: &'a PreparedStructuralStudy,
}

pub(super) enum LocalSweepOutcome {
    Completed(Result<FeaRunResult, String>),
    NotRun,
}

pub(super) struct LocalProcessStructuralExecutor {
    study_digest: Digest,
    program_revision: ProgramRevision,
    native_config: NativeExecutionConfig,
}

impl LocalProcessStructuralExecutor {
    pub(super) fn new(study: &AnalysisStudySpec, config: &RunMatRuntimeConfig) -> Result<Self> {
        let study_digest = analysis_study_content_digest(study);
        Ok(Self {
            study_digest,
            program_revision: program_revision(study_digest, config)?,
            native_config: NativeExecutionConfig::for_current_executable()?,
        })
    }
}

impl StructuralLinearExecutor for LocalProcessStructuralExecutor {
    fn execute(
        &self,
        workload: &StructuralLinearWorkload,
        mesh: &SolverMeshArtifact,
    ) -> Result<FeaRunResult, String> {
        let program = prepare_local_program(
            self.study_digest,
            self.program_revision.clone(),
            workload,
            mesh,
        )?;
        match execute_local_programs(vec![program], self.native_config.clone(), false)?
            .pop()
            .ok_or_else(|| "local structural execution returned no result".to_string())?
        {
            LocalSweepOutcome::Completed(result) => result,
            LocalSweepOutcome::NotRun => Err("local structural execution was not run".to_string()),
        }
    }
}

pub(super) fn execute_structural_sweep(
    entries: &[LocalSweepEntry<'_>],
    config: &RunMatRuntimeConfig,
    fail_fast: bool,
) -> Result<Vec<LocalSweepOutcome>, String> {
    if entries.is_empty() {
        return Err("local structural sweep must contain at least one prepared study".into());
    }
    let programs = entries
        .iter()
        .map(|entry| {
            let study_digest = analysis_study_content_digest(entry.study);
            prepare_local_program(
                study_digest,
                program_revision(study_digest, config).map_err(|error| error.to_string())?,
                entry.prepared.workload(),
                entry.prepared.solver_mesh(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let native_config =
        NativeExecutionConfig::for_current_executable().map_err(|error| error.to_string())?;
    execute_local_programs(programs, native_config, fail_fast)
}

fn program_revision(study_digest: Digest, config: &RunMatRuntimeConfig) -> Result<ProgramRevision> {
    Ok(ProgramRevision::new(
        study_digest,
        study_digest,
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?)
}

fn artifact_access(study_digest: Digest) -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: format!("study:{study_digest}"),
        encryption_context: Digest::sha256(
            [
                b"runmat-local-analysis-access/v1\0".as_slice(),
                study_digest.bytes(),
            ]
            .concat(),
        ),
    }
}

fn prepare_local_program(
    study_digest: Digest,
    program_revision: ProgramRevision,
    workload: &StructuralLinearWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<PreparedLocalProgram, String> {
    let limits = ObjectInventoryLimits::default();
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: study_digest,
        frozen_project_digest: None,
        program_revision,
        artifact_access: artifact_access(study_digest),
        workload: workload.clone(),
        mesh: mesh.clone(),
        initial_guess: None,
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
    Ok(PreparedLocalProgram {
        maximum_wall_millis: workload
            .procedure_step()
            .map_err(|error| error.to_string())?
            .accuracy
            .resources
            .maximum_wall_time_milliseconds,
        prepared,
        task_capabilities,
    })
}

fn execute_local_programs(
    programs: Vec<PreparedLocalProgram>,
    mut native_config: NativeExecutionConfig,
    fail_fast: bool,
) -> Result<Vec<LocalSweepOutcome>, String> {
    if programs.is_empty() {
        return Err("local structural program graph must not be empty".into());
    }
    for program in &programs {
        native_config
            .worker_capabilities
            .extend(program.task_capabilities.iter().cloned());
    }
    let session = NativeProgramSession::new(native_config).map_err(|error| error.to_string())?;
    let mut store = session.object_store();
    for program in &programs {
        program
            .prepared
            .program
            .write_to(&mut store)
            .map_err(|error| error.to_string())?;
    }
    let submissions = programs
        .iter()
        .map(|program| {
            let request = &program.prepared.program.request;
            let roots = request
                .arguments
                .iter()
                .map(|argument| match argument {
                    ValuePayload::Object(root) => Ok((**root).clone()),
                    ValuePayload::Inline(_) => {
                        Err("portable analysis input was unexpectedly inline".to_string())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let submission = build_analysis_task_submission(
                &program.prepared.host,
                roots,
                Vec::new(),
                BTreeSet::new(),
                AnalysisExecutionContext {
                    scope_id: session.scope_id(),
                    pool_id: session.pool_id(),
                    program_artifact_id: ArtifactId::derive(&[request.artifact.id.0.bytes()]),
                    resources: ResourceRequest {
                        cpu_millicores: 1_000,
                        memory_bytes: 1024 * 1024 * 1024,
                        scratch_bytes: 1024 * 1024 * 1024,
                        max_wall_millis: program.maximum_wall_millis,
                        max_artifact_bytes: ObjectInventoryLimits::default().max_total_bytes,
                        max_egress_bytes: 0,
                        max_relay_bytes: 0,
                        accelerators: Vec::new(),
                        required_capabilities: program.task_capabilities.clone(),
                    },
                    retry: RetryPolicy::IdempotentInfrastructure,
                    deadline_unix_millis: None,
                    priority: 0,
                },
            )
            .map_err(|error| error.to_string())?;
            Ok((request.clone(), submission))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let tasks = session
        .submit_graph(submissions)
        .map_err(|error| error.to_string())?;
    wait_for_programs(&session, &store, &programs, &tasks, fail_fast)
}

fn wait_for_programs(
    session: &NativeProgramSession,
    store: &FilesystemObjectStore,
    programs: &[PreparedLocalProgram],
    tasks: &[NativeProgramTask],
    fail_fast: bool,
) -> Result<Vec<LocalSweepOutcome>, String> {
    let limits = ObjectInventoryLimits::default();
    let started = Instant::now();
    let mut outcomes = (0..tasks.len()).map(|_| None).collect::<Vec<_>>();
    while outcomes.iter().any(Option::is_none) {
        let elapsed = started.elapsed();
        let mut timed_out = false;
        let mut failed = false;
        for (index, task) in tasks.iter().enumerate() {
            if outcomes[index].is_some() {
                continue;
            }
            if let Some(completed) = task.try_result() {
                let completed = completed.and_then(|success| {
                    let response = ProgramExecutionResponse::ExternalizedSuccess {
                        outputs: success.outputs,
                        result_objects: success.result_objects,
                    };
                    import_structural_linear_result(
                        store,
                        response,
                        &programs[index].prepared,
                        limits,
                    )
                    .map(|result| result.run)
                    .map_err(|error| error.to_string())
                });
                failed |= completed.is_err();
                outcomes[index] = Some(LocalSweepOutcome::Completed(completed));
            } else if elapsed >= Duration::from_millis(programs[index].maximum_wall_millis) {
                outcomes[index] = Some(LocalSweepOutcome::Completed(Err(
                    "local structural analysis exceeded its authored wall-time limit".into(),
                )));
                timed_out = true;
            }
        }
        if timed_out {
            session.cancel(runmat_execution::CancellationReason::Deadline);
            for outcome in &mut outcomes {
                if outcome.is_none() {
                    *outcome = Some(LocalSweepOutcome::NotRun);
                }
            }
        } else if fail_fast && failed {
            session.cancel(runmat_execution::CancellationReason::ParentScope);
            for outcome in &mut outcomes {
                if outcome.is_none() {
                    *outcome = Some(LocalSweepOutcome::NotRun);
                }
            }
        } else if outcomes.iter().any(Option::is_none) {
            thread::sleep(POLL_INTERVAL);
        }
    }
    outcomes
        .into_iter()
        .map(|outcome| outcome.ok_or_else(|| "local structural result was not resolved".into()))
        .collect()
}
