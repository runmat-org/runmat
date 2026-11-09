//! Publication of a solved cross-section: its line parameters as observation results, on the
//! canonical triangle mesh the potential solves ran on. No re-solve and no proxy values.
use crate::analysis::{AnalysisLineParametersRunOptions, AnalysisStudyCompletionIdentity};
use crate::analysis::*;
use crate::operations::{OperationContext, OperationEnvelope, OperationErrorEnvelope};
use runmat_analysis_fea::{
    contracts::{ExecutionBackend, SolverMethod, SolverPreconditioner},
    physics::electrical::cross_section::LineParameters,
    results::observation::ObservationResult,
    solve::backend::kind::LinearAlgebraBackendKind,
    study::CompiledAnalysisStudy,
    ComputeBackend, FeaRunResult,
};
use runmat_execution::Digest;
use runmat_meshing_core::{MeshingEvidence, SolverMeshArtifact, StableDigest};

/// A solved line-parameter study, ready to publish.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LineParametersStudyCompletion {
    pub compiled: CompiledAnalysisStudy,
    /// The section digest the mesh's geometry identity was derived from.
    pub section_digest: StableDigest,
    pub parameters: LineParameters,
    /// One result per authored output of the step, in authored order.
    pub observation_results: Vec<ObservationResult>,
    pub frozen_project_digest: Option<Digest>,
}

/// What a line-parameter run did: the section, its step and the controls it ran under.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
struct LineParametersWorkload {
    step_id: runmat_analysis_core::ProcedureStepId,
    section_digest: StableDigest,
    options: AnalysisLineParametersRunOptions,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
struct LineParametersNumericalEvidence {
    parameters: LineParameters,
    relative_residual: f64,
}

pub fn prepare_line_parameters_study_publication(
    completed: &LineParametersStudyCompletion,
    mesh: &SolverMeshArtifact,
    meshing_evidence: &MeshingEvidence,
    options: &AnalysisLineParametersRunOptions,
    context: OperationContext,
    publication_identity: AnalysisStudyCompletionIdentity,
) -> Result<crate::analysis::PreparedStudyPublication, OperationErrorEnvelope> {
    let study_id = completed.compiled.study_id.clone();
    let invalid = |message: String| publication_error(&context, &study_id, message);
    let study = completed
        .compiled
        .definition()
        .map_err(|error| invalid(error.to_string()))?;
    let definition = study.digest();
    let [step] = completed.compiled.model.sequence.steps.as_slice() else {
        return Err(invalid(
            "a line-parameter study publishes exactly one line-parameter step".into(),
        ));
    };
    let preparation =
        crate::analysis::study_publication::preparation::PublicationPreparation::capture(
            definition,
            &(completed, mesh, meshing_evidence, options),
        )
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
    {
        let _source_phase = preparation.staging.begin_source_phase().map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
        validate(completed, step, mesh, meshing_evidence, options).map_err(invalid)?;
    }
    let sequence =
        runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
            &completed.compiled.model.sequence,
        )
        .map_err(|error| invalid(error.to_string()))?;
    let step_identity =
        runmat_analysis_fea::execution::StepIdentity::from_solver_contract(sequence, step)
            .map_err(|error| invalid(error.to_string()))?;
    let numerical = completed
        .parameters
        .numerical_digest()
        .map_err(|error| invalid(error.to_string()))?;
    let solver_result_identity =
        runmat_analysis_fea::execution::SolverResultIdentity::from_step_result(
            step_identity,
            numerical,
        )
        .map_err(|error| invalid(error.to_string()))?;
    // The step's declared states hold the solved parameters; they name the same solver result.
    let states = step
        .state_outputs
        .iter()
        .map(|declaration| {
            runmat_analysis_fea::execution::CanonicalSemanticState::from_line_parameters(
                step_identity,
                declaration,
                mesh,
                &completed.parameters,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid(error.to_string()))?;
    let run_id = publication_identity.local_run_id(
        runmat_analysis_core::PhysicsFamily::Electromagnetic,
        definition,
        &step.step_id,
    );
    let run = materialize(completed, mesh, options, &run_id, &context).map_err(invalid)?;
    let persisted = preparation
        .persist_mesh(mesh, meshing_evidence)
        .map_err(|error| {
            crate::analysis::prepared_study_publication::publication_failure(
                error, &context, invalid,
            )
        })?;
    let execution = crate::analysis::AnalysisExecutionStep {
        step_id: step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        accelerator_admission: Default::default(),
        solver_options: crate::analysis::AnalysisSolverOptions::LineParameters(options.clone()),
    };
    let prepared = crate::analysis::structural_study::standalone_completion(
        context.clone(),
        study,
        execution,
        persisted,
    )
    .map_err(invalid)?;
    let completed_step = crate::analysis::structural_study::CompletedStudyStep {
        step_id: step.step_id.clone(),
        solver_result_identity,
        run: crate::analysis::structural_study::CompletedStudyStepRun::Standard(Box::new(run)),
        resolved_solver_options: crate::analysis::AnalysisSolverOptions::LineParameters(options.clone()),
    };
    let mut published =
        crate::analysis::structural_study::publication::complete_study_run_prepared(
            prepared,
            vec![completed_step],
            states,
            completed.frozen_project_digest,
            preparation,
        )?;
    let workload = LineParametersWorkload {
        step_id: step.step_id.clone(),
        section_digest: completed.section_digest,
        options: options.clone(),
    };
    let workload_digest = Digest::sha256(
        serde_json::to_vec(&workload).map_err(|error| invalid(error.to_string()))?,
    );
    let manifest = crate::analysis::study_result_manifest::StudyResultManifest::new(
        study,
        workload_digest,
        &workload,
        mesh,
        meshing_evidence,
        LineParametersNumericalEvidence {
            parameters: completed.parameters.clone(),
            relative_residual: completed.parameters.relative_residual,
        },
        &mut published,
    )
    .map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(error, &context, invalid)
    })?;
    crate::analysis::PreparedStudyPublication::new(published, manifest).map_err(|error| {
        crate::analysis::prepared_study_publication::publication_failure(error, &context, invalid)
    })
}

/// The mesh is the compiled study's mesh, the evidence closes over it, the solve met its
/// residual target, and every output has exactly one result under the step's plan.
fn validate(
    completed: &LineParametersStudyCompletion,
    step: &runmat_analysis_core::ProcedureStep,
    mesh: &SolverMeshArtifact,
    meshing_evidence: &MeshingEvidence,
    options: &AnalysisLineParametersRunOptions,
) -> Result<(), String> {
    if completed.compiled.solver_mesh_digest != mesh.canonical_digest
        || mesh.geometry
            != runmat_analysis_fea::physics::electrical::cross_section::cross_section_geometry(
                completed.section_digest,
            )
    {
        return Err("line-parameter mesh differs from the compiled section".into());
    }
    meshing_evidence
        .validate(mesh)
        .map_err(|error| error.to_string())?;
    if options.precision_mode != PrecisionMode::Fp64 {
        return Err("line parameters are solved in f64".into());
    }
    if !(completed.parameters.relative_residual <= options.residual_target) {
        return Err(format!(
            "the section's potential solves reached a relative residual of {:.3e}, above the target {:.3e}",
            completed.parameters.relative_residual, options.residual_target
        ));
    }
    runmat_analysis_fea::results::observation::validate_line_parameter_observations(
        step,
        mesh,
        &completed.observation_results,
    )
    .map_err(|error| error.to_string())
}

fn materialize(
    completed: &LineParametersStudyCompletion,
    mesh: &SolverMeshArtifact,
    options: &AnalysisLineParametersRunOptions,
    run_id: &str,
    context: &OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, String> {
    let identity = runmat_analysis_fea::admission::admit_cross_section_line_parameters()
        .map_err(|error| error.to_string())?;
    let run = FeaRunResult {
        identity,
        requested_backend: ComputeBackend::Cpu,
        execution_backend: ExecutionBackend::Cpu,
        solver_backend: LinearAlgebraBackendKind::CpuReference,
        solver_device_apply_k_ratio: 0.0,
        solver_method: SolverMethod::SparseDirectCholesky,
        preconditioner: SolverPreconditioner::None,
        solver_host_sync_count: 0,
        accelerator_evidence: Vec::new(),
        diagnostics: Vec::new(),
        spatial_projection_evidence: Vec::new(),
        structural_static_convergence: None,
        structural_static_balance: None,
        structural_constraint_result: None,
        structural_connector_result: None,
        observations: completed.observation_results.clone(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: Vec::new(),
    };
    let provenance = RunProvenance {
        requested_backend: run.requested_backend,
        execution_backend: run.execution_backend,
        solver_backend: run.solver_backend,
        solver_device_apply_k_ratio: run.solver_device_apply_k_ratio,
        solver_host_sync_count: run.solver_host_sync_count,
        precision_mode: contracts::format_precision_mode(options.precision_mode),
        deterministic_mode: true,
        solver_method: run.solver_method,
        preconditioner: run.preconditioner,
        quality_policy: contracts::format_quality_policy(QualityPolicy::Strict),
        fallback_events: Vec::new(),
        effective_memory_ceiling_bytes: None,
        static_solve_path: None,
    };
    let result = AnalysisRunResult {
        observation_scalars: Vec::new(),
        run_id: run_id.into(),
        run,
        render_topology: render_topology_from_solver_mesh(Some(mesh)),
        modal_results: None,
        buckling_results: None,
        thermal_results: None,
        transient_results: None,
        nonlinear_results: None,
        model_validity: QualityGate::Pass,
        solver_convergence: QualityGate::Pass,
        result_quality: QualityGate::Pass,
        run_status: RunStatus::Completed,
        numerically_acceptable: true,
        quality_reasons: Vec::new(),
        provenance,
    };
    Ok(OperationEnvelope::new(
        crate::analysis::execution_dispatch::AnalysisOperation::LineParameters.name(),
        crate::analysis::execution_dispatch::AnalysisOperation::LineParameters.version(),
        context,
        result,
    ))
}

fn publication_error(
    context: &OperationContext,
    study_id: &str,
    message: String,
) -> OperationErrorEnvelope {
    crate::operations::operation_error(
        crate::analysis::ANALYSIS_RUN_STUDY_OPERATION,
        crate::analysis::ANALYSIS_RUN_STUDY_OP_VERSION,
        context,
        crate::operations::OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.INVALID_LINE_PARAMETERS_RESULT",
            error_type: crate::operations::OperationErrorType::Contract,
            retryable: false,
            severity: crate::operations::OperationErrorSeverity::Fatal,
        },
        message,
        std::collections::BTreeMap::from([("study_id".into(), study_id.into())]),
    )
}
