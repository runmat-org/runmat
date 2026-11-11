use std::collections::BTreeMap;

use runmat_analysis_core::{
    AnalysisProcedure, ErrorContributionKind, IndependentDomain, ModalTruncationCorrection,
    ProcedureSettings, ProcedureStep, SolutionSpace,
};

use crate::execution::{
    StructuralBucklingSolverPolicy, StructuralBucklingTemplate, StructuralFrequencySolverPlan,
    StructuralFrequencyWorkload, StructuralFrequencyWorkloadInput,
    StructuralModalFrequencyTemplate, StructuralModalResponseSpectrumTemplate,
    StructuralModalTransientTemplate, StructuralModalWorkload, StructuralModalWorkloadInput,
    StructuralPrestressedFrequencySolverTemplate, StructuralPrestressedFrequencyTemplate,
    StructuralPrestressedModalTemplate, StructuralRandomResponseTemplate, StructuralStepWorkload,
    StructuralStudyWorkloads, StructuralTransientBlockPlan, StructuralTransientSolverPlan,
    StructuralTransientWorkload,
};
use crate::frequency::FrequencyResponsePlan;
use crate::pipeline::modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy};
use crate::results::observation::ObservationExecutionPolicy;
use crate::solve::dynamics::{DirectHarmonicPolicy, RandomResponseIntegrationPolicy};
use crate::solve::transient::StructuralTimeStepPolicy;

use super::{
    algebraic_residual_target, linear_workload, relative_accuracy_target, unsupported, CompiledAnalysisStudy,
    StructuralStudyPlanningError, StructuralStudyPlanningPolicy,
};

pub(super) fn plan(
    study: &CompiledAnalysisStudy,
    policy: StructuralStudyPlanningPolicy,
    fields: &[crate::execution::InitialFieldBinding],
) -> Result<StructuralStudyWorkloads, StructuralStudyPlanningError> {
    match study.model.sequence.steps.as_slice() {
        [step]
            if step.procedure == AnalysisProcedure::TransientResponse
                && step.solution_space == SolutionSpace::Direct =>
        {
            plan_direct_transient(study, step, policy, fields)
        }
        [step]
            if step.procedure == AnalysisProcedure::HarmonicResponse
                && step.solution_space == SolutionSpace::Direct =>
        {
            plan_direct_frequency(study, step, policy)
        }
        [step] if step.procedure == AnalysisProcedure::ModalExtraction => {
            let workload = modal_workload(study, step, policy, Vec::new())?;
            let planned = StructuralStudyWorkloads::StructuralModal {
                workload: Box::new(workload),
            };
            planned.validate()?;
            Ok(planned)
        }
        _ => plan_heterogeneous(study, policy, fields),
    }
}

fn plan_direct_transient(
    study: &CompiledAnalysisStudy,
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
    fields: &[crate::execution::InitialFieldBinding],
) -> Result<StructuralStudyWorkloads, StructuralStudyPlanningError> {
    let (interval_count, block_plan) = transient_schedule(step, policy)?;
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: super::initial_fields::select(&study.model, step, fields)?,
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        solver: StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
        residual_target: relative_accuracy_target(step, ErrorContributionKind::Algebraic)?,
    };
    let planned = StructuralStudyWorkloads::StructuralTransient {
        workload: Box::new(workload),
        block_plan,
    };
    planned.validate()?;
    Ok(planned)
}

fn plan_direct_frequency(
    study: &CompiledAnalysisStudy,
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
) -> Result<StructuralStudyWorkloads, StructuralStudyPlanningError> {
    let solver = StructuralFrequencySolverPlan::Direct {
        policy: harmonic_policy(step, policy)?,
    };
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: frequency_plan(step)?,
        solver,
        prestress_transition: None,
    })?;
    let planned = StructuralStudyWorkloads::StructuralFrequency {
        workload: Box::new(workload),
    };
    planned.validate()?;
    Ok(planned)
}

fn plan_heterogeneous(
    study: &CompiledAnalysisStudy,
    policy: StructuralStudyPlanningPolicy,
    fields: &[crate::execution::InitialFieldBinding],
) -> Result<StructuralStudyWorkloads, StructuralStudyPlanningError> {
    let residual_directions = modal_residual_directions(study)?;
    let mut workloads = BTreeMap::new();
    for step in &study.model.sequence.steps {
        if step.family == runmat_analysis_core::PhysicsFamily::Thermal {
            let workload = super::super::plan_thermal_step_with_fields(
                study,
                &step.step_id.0,
                super::super::ThermalStudyPlanningPolicy {
                    backend: policy.backend,
                    element_heat_source: None,
                },
                super::initial_fields::select(&study.model, step, fields)?,
            )
            .map_err(|error| {
                crate::execution::AnalysisExecutionError::Invalid(error.to_string())
            })?;
            workloads.insert(
                step.step_id.clone(),
                StructuralStepWorkload::Thermal {
                    workload: Box::new(workload),
                },
            );
            continue;
        }
        let workload = match (step.procedure, step.solution_space) {
            (AnalysisProcedure::Static, SolutionSpace::Direct) => StructuralStepWorkload::Linear {
                workload: Box::new(linear_workload(study, step, policy)?),
            },
            (AnalysisProcedure::ModalExtraction, SolutionSpace::Direct) => {
                StructuralStepWorkload::ModalExtraction {
                    workload: Box::new(modal_workload(
                        study,
                        step,
                        policy,
                        residual_directions.clone(),
                    )?),
                }
            }
            (AnalysisProcedure::PrestressedModalExtraction, SolutionSpace::Direct) => {
                StructuralStepWorkload::PrestressedModal {
                    template: Box::new(prestressed_modal_template(
                        study,
                        step,
                        policy,
                        residual_directions.clone(),
                    )?),
                }
            }
            (AnalysisProcedure::TransientResponse, SolutionSpace::ModalReduced) => {
                let (interval_count, block_plan) = transient_schedule(step, policy)?;
                StructuralStepWorkload::ModalTransient {
                    template: Box::new(StructuralModalTransientTemplate {
                        schema_version: 2,
                        initial_fields: super::initial_fields::select(&study.model, step, fields)?,
                        model: study.model.clone(),
                        step_id: step.step_id.clone(),
                        backend: policy.backend,
                        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
                        residual_target: relative_accuracy_target(
                            step,
                            ErrorContributionKind::Algebraic,
                        )?,
                    }),
                    block_plan,
                }
            }
            (AnalysisProcedure::HarmonicResponse, SolutionSpace::ModalReduced) => {
                StructuralStepWorkload::ModalFrequency {
                    template: Box::new(StructuralModalFrequencyTemplate {
                        schema_version: 1,
                        model: study.model.clone(),
                        step_id: step.step_id.clone(),
                        backend: policy.backend,
                        observation_execution: ObservationExecutionPolicy::scalable(),
                        plan: frequency_plan(step)?,
                    }),
                }
            }
            (AnalysisProcedure::ModalResponseSpectrum, SolutionSpace::ModalReduced) => {
                StructuralStepWorkload::ModalResponseSpectrum {
                    template: Box::new(StructuralModalResponseSpectrumTemplate {
                        schema_version: 1,
                        model: study.model.clone(),
                        step_id: step.step_id.clone(),
                        backend: policy.backend,
                        observation_execution: ObservationExecutionPolicy::scalable(),
                    }),
                }
            }
            (AnalysisProcedure::RandomResponse, SolutionSpace::ModalReduced) => {
                StructuralStepWorkload::RandomResponse {
                    template: Box::new(random_response_template(study, step, policy)?),
                }
            }
            (AnalysisProcedure::EigenvalueBuckling, SolutionSpace::Direct) => {
                StructuralStepWorkload::Buckling {
                    template: Box::new(buckling_template(study, step, policy)?),
                }
            }
            (AnalysisProcedure::PrestressedHarmonicResponse, solution_space) => {
                StructuralStepWorkload::PrestressedFrequency {
                    template: Box::new(prestressed_frequency_template(
                        study,
                        step,
                        policy,
                        solution_space,
                    )?),
                }
            }
            _ => return Err(unsupported(step)),
        };
        workload.validate()?;
        workloads.insert(step.step_id.clone(), workload);
    }
    let planned = StructuralStudyWorkloads::HeterogeneousSequence { workloads };
    planned.validate()?;
    Ok(planned)
}

fn modal_workload(
    study: &CompiledAnalysisStudy,
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
    residual_vector_directions: Vec<runmat_analysis_core::SpectrumDirection>,
) -> Result<StructuralModalWorkload, StructuralStudyPlanningError> {
    StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: modal_truncation(step, policy)?,
        convergence: modal_convergence(step, policy)?,
        residual_vector_directions,
        prestress_transition: None,
    })
    .map_err(Into::into)
}

fn prestressed_modal_template(
    study: &CompiledAnalysisStudy,
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
    residual_vector_directions: Vec<runmat_analysis_core::SpectrumDirection>,
) -> Result<StructuralPrestressedModalTemplate, StructuralStudyPlanningError> {
    let template = StructuralPrestressedModalTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: modal_truncation(step, policy)?,
        convergence: modal_convergence(step, policy)?,
        residual_vector_directions,
    };
    template.validate()?;
    Ok(template)
}

fn modal_truncation(
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
) -> Result<ModalTruncationPolicy, StructuralStudyPlanningError> {
    let IndependentDomain::Eigenvalue { requested_count } = step.independent_domain else {
        return Err(invalid_domain(step));
    };
    Ok(ModalTruncationPolicy::requested(requested_count)
        .with_eigenspace_relative_tolerance(policy.numerics.modal_eigenspace_relative_tolerance))
}

fn modal_convergence(
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
) -> Result<ModalConvergencePolicy, StructuralStudyPlanningError> {
    let (measure, tolerance) = algebraic_residual_target(step)?;
    Ok(ModalConvergencePolicy::production()
        .with_algebraic_accuracy(measure, tolerance)
        .with_maximum_subspace_iterations(policy.numerics.maximum_modal_iterations)
        .with_maximum_subspace_dimension(policy.numerics.maximum_modal_subspace_dimension))
}

fn transient_schedule(
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
) -> Result<(usize, StructuralTransientBlockPlan), StructuralStudyPlanningError> {
    let IndependentDomain::TimeSeconds { interval_count, .. } = step.independent_domain else {
        return Err(invalid_domain(step));
    };
    let interval_count_u64 = u64::try_from(interval_count).map_err(|_| {
        StructuralStudyPlanningError::TransientIntervalCountOverflow {
            step_id: step.step_id.0.clone(),
            interval_count,
        }
    })?;
    let maximum = interval_count_u64.min(policy.numerics.maximum_transient_intervals_per_block);
    Ok((interval_count, StructuralTransientBlockPlan::new(maximum)?))
}

fn frequency_plan(
    step: &ProcedureStep,
) -> Result<FrequencyResponsePlan, StructuralStudyPlanningError> {
    let IndependentDomain::FrequencyHertz(frequencies) = &step.independent_domain else {
        return Err(invalid_domain(step));
    };
    let (&reference, requested) = frequencies
        .split_first()
        .ok_or_else(|| invalid_domain(step))?;
    FrequencyResponsePlan::new(reference, requested, &[])
        .map_err(|error| StructuralStudyPlanningError::Frequency(error.to_string()))
}

fn harmonic_policy(
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
) -> Result<DirectHarmonicPolicy, StructuralStudyPlanningError> {
    Ok(DirectHarmonicPolicy {
        relative_tolerance: relative_accuracy_target(step, ErrorContributionKind::Algebraic)?,
        restart: policy.numerics.harmonic_restart,
        maximum_iterations: policy.numerics.maximum_harmonic_iterations,
    })
}

fn prestressed_frequency_template(
    study: &CompiledAnalysisStudy,
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
    solution_space: SolutionSpace,
) -> Result<StructuralPrestressedFrequencyTemplate, StructuralStudyPlanningError> {
    let solver = match solution_space {
        SolutionSpace::Direct => StructuralPrestressedFrequencySolverTemplate::Direct {
            policy: harmonic_policy(step, policy)?,
        },
        SolutionSpace::ModalReduced => StructuralPrestressedFrequencySolverTemplate::ModalReduced,
    };
    let template = StructuralPrestressedFrequencyTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: frequency_plan(step)?,
        solver,
    };
    template.validate()?;
    Ok(template)
}

fn buckling_template(
    study: &CompiledAnalysisStudy,
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
) -> Result<StructuralBucklingTemplate, StructuralStudyPlanningError> {
    let IndependentDomain::Eigenvalue { requested_count } = step.independent_domain else {
        return Err(invalid_domain(step));
    };
    let template = StructuralBucklingTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        observation_execution: ObservationExecutionPolicy::scalable(),
        solver: StructuralBucklingSolverPolicy::new(
            requested_count,
            relative_accuracy_target(step, ErrorContributionKind::Algebraic)?,
            policy.numerics.maximum_buckling_iterations,
        )?,
    };
    template.validate()?;
    Ok(template)
}

fn random_response_template(
    study: &CompiledAnalysisStudy,
    step: &ProcedureStep,
    policy: StructuralStudyPlanningPolicy,
) -> Result<StructuralRandomResponseTemplate, StructuralStudyPlanningError> {
    let ProcedureSettings::StructuralRandomResponse(settings) = step.settings else {
        return Err(StructuralStudyPlanningError::InvalidProcedureSettings {
            step_id: step.step_id.0.clone(),
        });
    };
    let maximum_refinement_levels = u8::try_from(step.accuracy.resources.maximum_refinement_cycles)
        .map_err(|_| StructuralStudyPlanningError::RefinementLimitOverflow {
            step_id: step.step_id.0.clone(),
        })?;
    let template = StructuralRandomResponseTemplate {
        schema_version: 1,
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        backend: policy.backend,
        observation_execution: ObservationExecutionPolicy::scalable(),
        integration_policy: RandomResponseIntegrationPolicy {
            modal_damping_ratio: settings.modal_damping_ratio,
            minimum_cumulative_mass_ratio: settings.minimum_cumulative_mass_ratio,
            relative_variance_tolerance: relative_accuracy_target(
                step,
                ErrorContributionKind::FrequencySampling,
            )?,
            maximum_refinement_levels,
        },
        maximum_intervals_per_task: policy.numerics.maximum_frequency_intervals_per_task,
    };
    template.validate()?;
    Ok(template)
}

/// Residual-vector directions of the study's modal basis: the authored modal-step directions,
/// then any response-spectrum direction that requests a residual-mode correction.
fn modal_residual_directions(
    study: &CompiledAnalysisStudy,
) -> Result<Vec<runmat_analysis_core::SpectrumDirection>, StructuralStudyPlanningError> {
    let mut directions = Vec::new();
    for step in &study.model.sequence.steps {
        for direction in step.modal_residual_vector_directions() {
            if !directions.contains(direction) {
                directions.push(direction.clone());
            }
        }
    }
    for step in &study.model.sequence.steps {
        let IndependentDomain::ResponseSpectra { spectrum_ids } = &step.independent_domain else {
            continue;
        };
        for spectrum_id in spectrum_ids {
            let spectrum = study
                .model
                .response_spectra
                .iter()
                .find(|candidate| candidate.spectrum_id == *spectrum_id)
                .ok_or_else(|| invalid_domain(step))?;
            if spectrum.truncation.correction == ModalTruncationCorrection::ResidualMode
                && !directions.contains(&spectrum.direction)
            {
                directions.push(spectrum.direction.clone());
            }
        }
    }
    Ok(directions)
}

fn invalid_domain(step: &ProcedureStep) -> StructuralStudyPlanningError {
    StructuralStudyPlanningError::InvalidIndependentDomain {
        step_id: step.step_id.0.clone(),
    }
}
