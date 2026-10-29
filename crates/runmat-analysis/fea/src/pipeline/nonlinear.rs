use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, AnalysisProcedure, IndependentDomain,
    ObservationEvent, ObservationRequest, SamplingPolicy,
};

use crate::{
    assembly::{
        assemble_structural_system_borrowed, LinearAssemblyError, StructuralFormulationRequirement,
    },
    contracts::{
        fea_nonlinear_displacement_field_id, fea_nonlinear_equivalent_plastic_strain_field_id,
        fea_nonlinear_load_factor_field_id, fea_nonlinear_plastic_strain_field_id,
        fea_nonlinear_residual_norm_field_id, fea_nonlinear_rotation_field_id,
        fea_nonlinear_von_mises_field_id, ComputeBackend, FeaNonlinearRunResult, FeaRunError,
        FeaRunResult, FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    execution::{SequenceIdentity, SolverResultIdentity, StepIdentity},
    pipeline::observation_history::{
        evaluate_field_histories, required_load_factor_history_samples, HistoryFieldSeries,
        HistoryTopology,
    },
    pipeline::structural_nodal::{
        nodal_vector_shape, project_structural_nodal_snapshot, project_structural_nodal_snapshots,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::nonlinear::{
        detect_completed_nonlinear_events, nonlinear_solver_content_digest,
        solve_nonlinear_system_at_boundaries, LoadIncrementSchedule, NonlinearSolveOptions,
        NonlinearSolveResult,
    },
};

const TENSOR_COMPONENT_COUNT: usize = 6;
const NONLINEAR_DISPLACEMENT_HISTORY_SOURCE: &str = "structural.nonlinear.displacement";
const NONLINEAR_ROTATION_HISTORY_SOURCE: &str = "structural.nonlinear.rotation";
mod blocks;
mod contact;
mod observations;
mod preflight;
mod preparation;
mod recovery;
pub use blocks::{run_nonlinear_block, NonlinearBlockInput, NonlinearBlockOutput};
pub use observations::require_nonlinear_terminal_field_outputs;
pub use recovery::recover_original_nonlinear_block;

#[cfg(test)]
mod qualification;

pub fn run_nonlinear(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    run_nonlinear_with_options(model, backend, NonlinearSolveOptions::default())
}

pub fn run_nonlinear_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: NonlinearSolveOptions,
) -> Result<FeaNonlinearRunResult, FeaRunError> {
    let admission = crate::contracts::accelerator_admission::admit_accelerator(
        crate::contracts::accelerator_admission::AcceleratorFamily::StructuralNonlinear,
        backend,
    )?;
    let prepared = preparation::prepare(model, admission.effective, &options)?;
    let solver_admission = preflight::tangent_admission(&prepared.resource_plan)?;
    let maximum_resident_bytes = prepared
        .execution_step
        .accuracy
        .resources
        .resident_byte_budget();
    let contact_inputs = match options.solver_mesh.as_ref() {
        Some(mesh) => contact::prepare_inputs(
            model,
            mesh,
            &prepared.summary,
            options.contact_enforcement.clone().unwrap_or_default(),
            maximum_resident_bytes,
        )?,
        None => None,
    };
    let contact_operator = contact_inputs
        .as_ref()
        .map(|inputs| contact::prepare_operator(model, inputs, prepared.summary.dof_count))
        .transpose()?;
    let contact_execution = contact_operator
        .as_ref()
        .map(|operator| contact::execution(operator, &options, maximum_resident_bytes));

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving nonlinear increments",
        Some(2),
        Some(5),
    );
    let nonlinear = solve_nonlinear_system_at_boundaries(
        model,
        &prepared.summary,
        &options,
        crate::solve::nonlinear::NonlinearExecution {
            backend: admission.effective,
            contact: contact_execution,
            solver_admission: Some(solver_admission),
        },
        prepared.load_factor_start,
        prepared.load_factor_end,
        &prepared.increment_schedule,
    )?;
    let mut nonlinear = nonlinear;
    nonlinear.diagnostics.extend(admission.fallback);
    recovery::recover(model, backend, &options, prepared, nonlinear, 0)
}

fn validate_nonlinear_event_requests(
    step: &runmat_analysis_core::ProcedureStep,
) -> Result<(), FeaRunError> {
    for request in &step.observations {
        let ObservationRequest::History(spec) = request else {
            continue;
        };
        let SamplingPolicy::Event(event) = &spec.sampling else {
            continue;
        };
        if !matches!(*event, ObservationEvent::Solver(_)) {
            return Err(FeaRunError::Observation(
                "nonlinear execution currently owns solver events; derived-series and accuracy events require their source result or report"
                    .into(),
            ));
        }
    }
    Ok(())
}

fn nonlinear_event_attributions(
    model: &AnalysisModel,
    step: &runmat_analysis_core::ProcedureStep,
    result: &NonlinearSolveResult,
    resource_plan: &crate::planning::resource::FeaResourcePlan,
) -> Result<Vec<crate::results::observation::ObservationEventAttribution>, FeaRunError> {
    if !step.observations.iter().any(|request| {
        matches!(
            request,
            ObservationRequest::History(runmat_analysis_core::ObservationSpec {
                sampling: SamplingPolicy::Event(ObservationEvent::Solver(_)),
                ..
            })
        )
    }) {
        return Ok(Vec::new());
    }
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let producer = StepIdentity::from_solver_contract(sequence, step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let history_limits = match &resource_plan.input.demand {
        crate::planning::resource::FeaResourceDemand::StructuralNonlinearPipeline(demand) => {
            demand.history_limits
        }
        crate::planning::resource::FeaResourceDemand::StructuralNonlinearRecovery(demand) => {
            demand.history_limits
        }
        _ => {
            return Err(FeaRunError::Observation(
                "nonlinear event attribution requires a nonlinear resource plan".into(),
            ))
        }
    };
    let content = nonlinear_solver_content_digest(result, history_limits)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let source = SolverResultIdentity::from_step_result(producer, content)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    Ok(detect_completed_nonlinear_events(result)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?
        .into_iter()
        .map(|event| {
            crate::results::observation::ObservationEventAttribution::Solver(
                crate::results::observation::SolverEventAttribution {
                    event: event.event,
                    sample: crate::results::field::RecoveredFieldSample::LoadFactor(
                        event.load_factor,
                    ),
                    occurrence_index: event.occurrence_index,
                    source: crate::results::observation::SolverEventSource::CompletedResult {
                        result: source,
                    },
                },
            )
        })
        .collect())
}
