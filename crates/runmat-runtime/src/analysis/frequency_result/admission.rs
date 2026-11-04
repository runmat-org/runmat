use std::sync::Arc;

use runmat_execution::value::{PublishedObjectCatalog, PublishedObjectSource};

use super::StructuralStudyFrequencyNumericalResult;
use crate::analysis::contracts::{
    self, AnalysisStructuralFrequencyRunOptions, AnalysisStudyStepNumericalResultData,
    FallbackEvent, QualityGate, QualityReason, QualityReasonCode, RunProvenance, RunStatus,
    StructuralFrequencyResultsData,
};
use crate::analysis::result_policy::classify_run_status;
use crate::analysis::{AnalysisRunIdentity, StructuralFrequencyArtifactRootIndex};

#[derive(Clone)]
pub(in crate::analysis) struct StructuralFrequencyPersistence {
    pub object_catalog: PublishedObjectCatalog,
    pub artifact_roots: StructuralFrequencyArtifactRootIndex,
    pub source: Arc<dyn PublishedObjectSource>,
}

pub(in crate::analysis) struct CompletedStructuralFrequencyStep {
    pub identity: AnalysisRunIdentity,
    pub run_id: String,
    pub run_status: RunStatus,
    pub numerically_acceptable: bool,
    pub solver_convergence: QualityGate,
    pub result_quality: QualityGate,
    pub quality_reasons: Vec<QualityReason>,
    pub provenance: RunProvenance,
    pub numerical_result: AnalysisStudyStepNumericalResultData,
    pub persistence: StructuralFrequencyPersistence,
}

pub(in crate::analysis) fn admit_structural_frequency_result(
    numerical: &StructuralStudyFrequencyNumericalResult,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisStructuralFrequencyRunOptions,
    assigned_run_id: String,
) -> Result<CompletedStructuralFrequencyStep, String> {
    admit_structural_frequency_result_with_state(
        numerical,
        solver_mesh,
        options,
        assigned_run_id,
        None,
        None,
    )
}

pub(in crate::analysis) fn admit_prestressed_structural_frequency_result(
    numerical: &StructuralStudyFrequencyNumericalResult,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisStructuralFrequencyRunOptions,
    assigned_run_id: String,
    prestress_state: &runmat_analysis_fea::execution::CanonicalSemanticState,
) -> Result<CompletedStructuralFrequencyStep, String> {
    admit_structural_frequency_result_with_state(
        numerical,
        solver_mesh,
        options,
        assigned_run_id,
        Some(prestress_state),
        None,
    )
}

pub(in crate::analysis) fn admit_modal_structural_frequency_result(
    numerical: &StructuralStudyFrequencyNumericalResult,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisStructuralFrequencyRunOptions,
    assigned_run_id: String,
    modal_state: &runmat_analysis_fea::execution::CanonicalSemanticState,
) -> Result<CompletedStructuralFrequencyStep, String> {
    admit_structural_frequency_result_with_state(
        numerical,
        solver_mesh,
        options,
        assigned_run_id,
        None,
        Some(modal_state),
    )
}

pub(in crate::analysis) fn admit_prestressed_modal_structural_frequency_result(
    numerical: &StructuralStudyFrequencyNumericalResult,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisStructuralFrequencyRunOptions,
    assigned_run_id: String,
    prestress_state: &runmat_analysis_fea::execution::CanonicalSemanticState,
    modal_state: &runmat_analysis_fea::execution::CanonicalSemanticState,
) -> Result<CompletedStructuralFrequencyStep, String> {
    admit_structural_frequency_result_with_state(
        numerical,
        solver_mesh,
        options,
        assigned_run_id,
        Some(prestress_state),
        Some(modal_state),
    )
}

fn admit_structural_frequency_result_with_state(
    numerical: &StructuralStudyFrequencyNumericalResult,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    options: &AnalysisStructuralFrequencyRunOptions,
    assigned_run_id: String,
    prestress_state: Option<&runmat_analysis_fea::execution::CanonicalSemanticState>,
    modal_state: Option<&runmat_analysis_fea::execution::CanonicalSemanticState>,
) -> Result<CompletedStructuralFrequencyStep, String> {
    let identity = match (prestress_state, modal_state) {
        (Some(prestress), Some(modal)) => runmat_analysis_fea::pipeline::structural_frequency::admit_prestressed_modal_structural_frequency_publication(
            &numerical.workload,
            solver_mesh,
            prestress,
            modal,
            &numerical.result,
            &numerical.publication,
        ),
        (Some(state), None) => runmat_analysis_fea::pipeline::structural_frequency::admit_prestressed_structural_frequency_publication(
            &numerical.workload,
            solver_mesh,
            state,
            &numerical.result,
            &numerical.publication,
        ),
        (None, None) => runmat_analysis_fea::pipeline::structural_frequency::admit_structural_frequency_publication(
            &numerical.workload,
            solver_mesh,
            &numerical.result,
            &numerical.publication,
        ),
        (None, Some(modal)) => runmat_analysis_fea::pipeline::structural_frequency::admit_modal_structural_frequency_publication(
            &numerical.workload,
            solver_mesh,
            modal,
            &numerical.result,
            &numerical.publication,
        ),
    }
    .map_err(|error| error.to_string())?;
    match (&options.solver, numerical.workload.solver()) {
        (
            crate::analysis::AnalysisStructuralFrequencySolver::Direct { policy: authored },
            runmat_analysis_fea::execution::StructuralFrequencySolverPlan::Direct { policy },
        ) if authored == policy => authored.validate().map_err(|error| error.to_string())?,
        (
            crate::analysis::AnalysisStructuralFrequencySolver::ModalReduced,
            runmat_analysis_fea::execution::StructuralFrequencySolverPlan::ModalReduced { .. },
        ) => {}
        _ => {
            return Err(
                "structural frequency publication solver differs from its typed workload".into(),
            )
        }
    }

    let receipt = numerical.publication.receipt();
    let evidence = receipt.samples();
    let first = evidence
        .first()
        .ok_or_else(|| "structural frequency receipt contains no sample evidence".to_string())?;
    if evidence.iter().any(|sample| {
        sample.solver_method() != first.solver_method()
            || sample.solver_backend() != first.solver_backend()
            || sample.preconditioner() != first.preconditioner()
    }) {
        return Err(
            "structural frequency samples use inconsistent solver provenance within one run".into(),
        );
    }
    if first.solver_backend()
        != runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind::CpuReference
        || numerical.workload.backend() != runmat_analysis_fea::ComputeBackend::Cpu
    {
        return Err(
            "current structural frequency runtime publication admits only governed CPU execution"
                .into(),
        );
    }

    let relative_tolerance = authored_frequency_relative_tolerance(&numerical.workload)?;
    let maximum_solver_residual = evidence
        .iter()
        .map(|sample| sample.solver_relative_residual())
        .fold(0.0_f64, f64::max);
    let maximum_balance_error = evidence
        .iter()
        .map(|sample| sample.physical_relative_balance_error())
        .fold(0.0_f64, f64::max);
    let solver_convergence = if maximum_solver_residual <= relative_tolerance {
        QualityGate::Pass
    } else {
        QualityGate::Fail
    };
    let result_quality = if maximum_balance_error <= relative_tolerance {
        QualityGate::Pass
    } else {
        QualityGate::Fail
    };
    let mut quality_reasons = Vec::new();
    if solver_convergence != QualityGate::Pass {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::StructuralFrequencyResidualExceeded,
            detail: format!(
                "maximum structural frequency relative solver residual {maximum_solver_residual} exceeds {relative_tolerance}"
            ),
        });
    }
    if result_quality != QualityGate::Pass {
        quality_reasons.push(QualityReason {
            code: QualityReasonCode::StructuralFrequencyBalanceExceeded,
            detail: format!(
                "maximum structural frequency physical balance error {maximum_balance_error} exceeds {relative_tolerance}"
            ),
        });
    }
    let numerically_acceptable =
        solver_convergence == QualityGate::Pass && result_quality == QualityGate::Pass;
    let run_status = classify_run_status(result_quality, numerically_acceptable);

    Ok(CompletedStructuralFrequencyStep {
        identity,
        run_id: assigned_run_id,
        run_status,
        numerically_acceptable,
        solver_convergence,
        result_quality,
        quality_reasons,
        provenance: RunProvenance {
            requested_backend: numerical.workload.backend(),
            execution_backend: runmat_analysis_fea::ExecutionBackend::Cpu,
            solver_backend: first.solver_backend(),
            solver_device_apply_k_ratio: 0.0,
            solver_host_sync_count: 0,
            precision_mode: contracts::format_precision_mode(options.precision_mode),
            deterministic_mode: options.deterministic_mode,
            solver_method: first.solver_method(),
            preconditioner: first.preconditioner(),
            quality_policy: contracts::format_quality_policy(options.quality_policy),
            fallback_events: Vec::<FallbackEvent>::new(),
            effective_memory_ceiling_bytes: None,
            static_solve_path: None,
        },
        numerical_result: AnalysisStudyStepNumericalResultData::StructuralFrequency(Box::new(
            StructuralFrequencyResultsData {
                publication: numerical.publication.clone(),
            },
        )),
        persistence: numerical.persistence(),
    })
}

fn authored_frequency_relative_tolerance(
    workload: &runmat_analysis_fea::execution::StructuralFrequencyWorkload,
) -> Result<f64, String> {
    let step = workload
        .model()
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == *workload.step_id())
        .ok_or_else(|| "structural frequency workload omits its authored step".to_string())?;
    step.accuracy
        .criteria
        .iter()
        .filter(|criterion| {
            criterion.mandatory
                && criterion.contribution
                    == runmat_analysis_core::ErrorContributionKind::Algebraic
                && matches!(
                    criterion.evidence,
                    runmat_analysis_core::RequiredAccuracyEvidence::ResidualBound
                        | runmat_analysis_core::RequiredAccuracyEvidence::ConservationBalance
                )
        })
        .filter_map(|criterion| match criterion.tolerance {
            runmat_analysis_core::AccuracyTolerance::Relative(tolerance) => Some(tolerance),
            runmat_analysis_core::AccuracyTolerance::Absolute { .. } => None,
        })
        .min_by(f64::total_cmp)
        .ok_or_else(|| {
            "structural frequency publication requires a mandatory relative algebraic residual or conservation criterion"
                .to_string()
        })
}
