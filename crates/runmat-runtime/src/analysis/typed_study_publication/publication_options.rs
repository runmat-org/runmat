//! Exact projection from portable typed workloads into the shared runtime publication contract.
//!
//! This module does not resolve defaults or analysis intent. Every runtime option is derived
//! from an already-compiled workload and exists only to admit its completed numerical result.

use runmat_analysis_core::{ProcedureStep, ProcedureStepId};
use runmat_analysis_fea::execution::{
    StructuralFrequencySolverPlan, StructuralStepWorkload, StructuralStudyWorkloads,
};
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;

use super::super::structural_study::PreparedStructuralStudyIntentStep;
use super::super::{
    AnalysisBucklingRunOptions, AnalysisExecutionStep, AnalysisModalRunOptions,
    AnalysisRandomResponseRunOptions, AnalysisResponseSpectrumRunOptions, AnalysisRunOptions,
    AnalysisSolverOptions, AnalysisStructuralFrequencyRunOptions,
    AnalysisStructuralFrequencySolver, AnalysisStructuralTimeStepPolicy,
    AnalysisTransientRunOptions, PrecisionMode, PreconditionerMode, QualityPolicy,
};

pub(super) fn publication_steps(
    workloads: &StructuralStudyWorkloads,
    authored_steps: &[ProcedureStep],
) -> Result<Vec<PreparedStructuralStudyIntentStep>, String> {
    authored_steps
        .iter()
        .map(|step| {
            Ok(PreparedStructuralStudyIntentStep {
                authored_execution: execution_step(workloads, step)?,
                procedure_step: step.clone(),
            })
        })
        .collect()
}

fn execution_step(
    workloads: &StructuralStudyWorkloads,
    step: &ProcedureStep,
) -> Result<AnalysisExecutionStep, String> {
    let (backend, solver_options) = match workloads {
        StructuralStudyWorkloads::LinearSequence { workloads } => {
            let workload = workload_for_step(workloads, &step.step_id)?;
            (workload.backend, linear_options(workload.preconditioner))
        }
        StructuralStudyWorkloads::StructuralTransient {
            workload,
            block_plan,
        } if workload.step_id == step.step_id => (
            workload.backend,
            transient_options(
                workload.time_step_policy.interval_count(),
                workload.residual_target,
                block_plan.maximum_intervals_per_block(),
            ),
        ),
        StructuralStudyWorkloads::StructuralFrequency { workload }
            if workload.step_id() == &step.step_id =>
        {
            let StructuralFrequencySolverPlan::Direct { policy } = workload.solver() else {
                return Err(
                    "standalone typed frequency publication requires a direct solver".into(),
                );
            };
            (
                workload.backend(),
                frequency_options(AnalysisStructuralFrequencySolver::Direct { policy: *policy }),
            )
        }
        StructuralStudyWorkloads::StructuralModal { workload }
            if workload.step_id() == &step.step_id =>
        {
            (
                workload.backend(),
                modal_options(
                    workload.truncation(),
                    workload.convergence(),
                    workload.residual_vector_directions().to_vec(),
                ),
            )
        }
        StructuralStudyWorkloads::HeterogeneousSequence { workloads } => {
            heterogeneous_options(workload_for_step(workloads, &step.step_id)?)?
        }
        _ => {
            return Err(format!(
                "typed workload inventory has no exact entry for procedure step `{}`",
                step.step_id.as_str()
            ))
        }
    };
    Ok(AnalysisExecutionStep {
        step_id: step.step_id.clone(),
        backend,
        accelerator_admission: Default::default(),
        solver_options,
    })
}

fn heterogeneous_options(
    workload: &StructuralStepWorkload,
) -> Result<(runmat_analysis_fea::ComputeBackend, AnalysisSolverOptions), String> {
    Ok(match workload {
        StructuralStepWorkload::Thermal { workload } => (
            workload.backend,
            AnalysisSolverOptions::ThermalDiffusion(super::super::AnalysisThermalRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Strict,
                snapshot_count: match workload.procedure {
                    runmat_analysis_fea::ThermalProcedureControl::Steady => 1,
                    runmat_analysis_fea::ThermalProcedureControl::BackwardEuler {
                        snapshot_count,
                    } => snapshot_count,
                },
                residual_warn_threshold: workload.residual_target,
                solver_mesh_artifact_path: None,
            }),
        ),
        StructuralStepWorkload::Linear { workload } => {
            (workload.backend, linear_options(workload.preconditioner))
        }
        StructuralStepWorkload::ModalExtraction { workload } => (
            workload.backend(),
            modal_options(
                workload.truncation(),
                workload.convergence(),
                workload.residual_vector_directions().to_vec(),
            ),
        ),
        StructuralStepWorkload::ModalTransient {
            template,
            block_plan,
        } => (
            template.backend,
            transient_options(
                template.time_step_policy.interval_count(),
                template.residual_target,
                block_plan.maximum_intervals_per_block(),
            ),
        ),
        StructuralStepWorkload::ModalResponseSpectrum { template } => (
            template.backend,
            AnalysisSolverOptions::StructuralResponseSpectrum(AnalysisResponseSpectrumRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Strict,
                solver_mesh_artifact_path: None,
            }),
        ),
        StructuralStepWorkload::RandomResponse { template } => (
            template.backend,
            AnalysisSolverOptions::StructuralRandomResponse(AnalysisRandomResponseRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Strict,
                integration: template.integration_policy,
                maximum_intervals_per_task: template.maximum_intervals_per_task,
                solver_mesh_artifact_path: None,
            }),
        ),
        StructuralStepWorkload::PrestressedModal { template } => (
            template.backend,
            modal_options(
                template.truncation,
                template.convergence,
                template.residual_vector_directions.clone(),
            ),
        ),
        StructuralStepWorkload::Buckling { template } => (
            template.backend,
            AnalysisSolverOptions::StructuralBuckling(AnalysisBucklingRunOptions {
                deterministic_mode: true,
                precision_mode: PrecisionMode::Fp64,
                quality_policy: QualityPolicy::Strict,
                solver: template.solver,
                solver_mesh_artifact_path: None,
            }),
        ),
        StructuralStepWorkload::PrestressedFrequency { template } => {
            let solver = match template.solver {
                runmat_analysis_fea::execution::StructuralPrestressedFrequencySolverTemplate::Direct {
                    policy,
                } => AnalysisStructuralFrequencySolver::Direct { policy },
                runmat_analysis_fea::execution::StructuralPrestressedFrequencySolverTemplate::ModalReduced => {
                    AnalysisStructuralFrequencySolver::ModalReduced
                }
            };
            (template.backend, frequency_options(solver))
        }
        StructuralStepWorkload::ModalFrequency { template } => (
            template.backend,
            frequency_options(AnalysisStructuralFrequencySolver::ModalReduced),
        ),
    })
}

fn workload_for_step<'a, T>(
    workloads: &'a std::collections::BTreeMap<ProcedureStepId, T>,
    step_id: &ProcedureStepId,
) -> Result<&'a T, String> {
    workloads.get(step_id).ok_or_else(|| {
        format!(
            "typed workload inventory is missing procedure step `{}`",
            step_id.as_str()
        )
    })
}

fn linear_options(preconditioner: SpdPreconditionerKind) -> AnalysisSolverOptions {
    AnalysisSolverOptions::StructuralLinear(AnalysisRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        preconditioner_mode: match preconditioner {
            SpdPreconditionerKind::Auto | SpdPreconditionerKind::Direct => PreconditionerMode::Auto,
            SpdPreconditionerKind::Amg => PreconditionerMode::Amg,
            SpdPreconditionerKind::Jacobi => PreconditionerMode::Jacobi,
            SpdPreconditionerKind::Ilu0 => PreconditionerMode::Ilu,
        },
        quality_policy: QualityPolicy::Strict,
        solver_mesh_artifact_path: None,
    })
}

fn transient_options(
    interval_count: usize,
    residual_target: f64,
    maximum_intervals_per_block: u64,
) -> AnalysisSolverOptions {
    AnalysisSolverOptions::StructuralTransient(AnalysisTransientRunOptions {
        deterministic_mode: true,
        quality_policy: QualityPolicy::Strict,
        time_step_policy: AnalysisStructuralTimeStepPolicy::Fixed { interval_count },
        residual_target,
        maximum_intervals_per_block,
        solver_mesh_artifact_path: None,
    })
}

fn frequency_options(solver: AnalysisStructuralFrequencySolver) -> AnalysisSolverOptions {
    AnalysisSolverOptions::StructuralFrequency(AnalysisStructuralFrequencyRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        quality_policy: QualityPolicy::Strict,
        solver,
        solver_mesh_artifact_path: None,
    })
}

fn modal_options(
    truncation: runmat_analysis_fea::ModalTruncationPolicy,
    convergence: runmat_analysis_fea::ModalConvergencePolicy,
    residual_vector_directions: Vec<runmat_analysis_core::SpectrumDirection>,
) -> AnalysisSolverOptions {
    AnalysisSolverOptions::StructuralModes(AnalysisModalRunOptions {
        deterministic_mode: true,
        precision_mode: PrecisionMode::Fp64,
        quality_policy: QualityPolicy::Strict,
        truncation,
        convergence,
        residual_vector_directions,
        solver_mesh_artifact_path: None,
    })
}
