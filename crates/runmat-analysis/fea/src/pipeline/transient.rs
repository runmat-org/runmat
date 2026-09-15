use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, AnalysisProcedure, ProcedureStep,
};

use crate::assembly::AssemblySummary;
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    contracts::{
        fea_transient_damping_dissipation_field_id, fea_transient_energy_balance_residual_field_id,
        fea_transient_external_work_field_id, fea_transient_kinetic_energy_field_id,
        fea_transient_relative_energy_balance_error_field_id, fea_transient_residual_norm_field_id,
        fea_transient_strain_energy_field_id, fea_transient_support_work_field_id, ComputeBackend,
        FeaRunError, FeaRunResult, FeaTransientRunResult, StructuralInitialStateEvidence,
        FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
    excitation::{
        realize_step_excitations, time_discontinuities, ExcitationSamplePoint, StructuralMotionSet,
    },
    execution::{
        StructuralTransientBlockRequest, StructuralTransientCheckpoint,
        StructuralTransientCheckpointScope, StructuralTransientCheckpointScopeInput,
        StructuralTransientObservationCursor, StructuralTransientSegment,
        StructuralTransientWorkload,
    },
    physics::structural::dynamics::initial_state::{
        realize as realize_structural_initial_state, RealizedStructuralInitialState,
        StructuralInitialConstraintView, StructuralInitialStateRealizationBasis,
    },
    pipeline::electro_thermal::recover_realized_electro_thermal_fields,
    pipeline::observation_history::{
        evaluate_field_histories, required_time_history_samples, HistoryFieldSeries,
        HistoryTopology,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::transient::{
        canonical_transient_step_boundaries, mechanical_energy_history_diagnostic,
        solve_structural_modal_transient_block_with_assembly,
        solve_structural_transient_block_with_assembly, solve_transient_system_with_assembly,
        StructuralModalTransientBlockSolveInput, StructuralTransientBlockSolveInput,
        StructuralTransientInitialConditions, TransientSolveError, TransientSolveOptions,
        TransientTimeDomain,
    },
};

mod constraint_admission;
mod finalization;
mod motion;
mod reconstruction;
mod recovery;

use finalization::{finish_transient_run, TransientRunFinalization};

pub use reconstruction::{recover_completed_transient_run, CompletedStructuralTransientRunInput};

const TRANSIENT_DISPLACEMENT_HISTORY_SOURCE: &str = "structural.transient.displacement";
const TRANSIENT_VELOCITY_HISTORY_SOURCE: &str = "structural.transient.velocity";
const TRANSIENT_ACCELERATION_HISTORY_SOURCE: &str = "structural.transient.acceleration";
const TRANSIENT_ROTATION_HISTORY_SOURCE: &str = "structural.transient.rotation";
const TRANSIENT_STRAIN_HISTORY_SOURCE: &str = "structural.transient.strain";
const TRANSIENT_STRESS_HISTORY_SOURCE: &str = "structural.transient.stress";
const TRANSIENT_STRAIN_ENERGY_DENSITY_HISTORY_SOURCE: &str =
    "structural.transient.strain_energy_density";

pub fn run_transient(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaTransientRunResult, FeaRunError> {
    run_transient_with_options(model, backend, TransientSolveOptions::default())
}

pub struct StructuralTransientRunBlockInput<'a> {
    pub workload: &'a StructuralTransientWorkload,
    pub solver_mesh: &'a runmat_meshing_core::SolverMeshArtifact,
    pub request: &'a StructuralTransientBlockRequest,
    pub predecessor: Option<&'a StructuralTransientCheckpoint>,
    pub modal_state: Option<&'a crate::execution::CanonicalSemanticState>,
    pub observation_cursors: Vec<StructuralTransientObservationCursor>,
}

struct PreparedTransientBlockBasis {
    execution_step: ProcedureStep,
    active_model: AnalysisModel,
    motions: StructuralMotionSet,
    initial_assembly: AssemblySummary,
    schedule: crate::execution::StructuralTransientSchedule,
}

fn realize_transient_initial_state(
    active_model: &AnalysisModel,
    execution_step: &ProcedureStep,
    initial_assembly: &AssemblySummary,
    solver_mesh: Option<&runmat_meshing_core::SolverMeshArtifact>,
) -> Result<RealizedStructuralInitialState, FeaRunError> {
    let initial_state_basis = match initial_assembly.structural_formulations.as_slice() {
        [realization]
            if realization.operator
                == runmat_analysis_core::StructuralOperatorImplementation::SyntheticDiagonalChain =>
        {
            StructuralInitialStateRealizationBasis::OperatorBoundZeroDefault {
                dof_count: initial_assembly.operator.dof_count(),
                operator_identity: initial_assembly
                    .operator
                    .operators()
                    .stiffness()
                    .numerical_identity()
                    .map_err(|error| FeaRunError::Assembly(error.to_string()))?,
            }
        }
        _ => StructuralInitialStateRealizationBasis::CanonicalTopology,
    };
    realize_structural_initial_state(
        active_model,
        solver_mesh,
        &initial_assembly.structural_dof_layout,
        initial_state_basis,
        StructuralInitialConstraintView {
            constraint_matrix: initial_assembly.constraint_matrix.as_ref(),
            affine_reduction: initial_assembly.affine_constraint_reduction.as_ref(),
            direct_constrained_dofs: initial_assembly.operator.constrained_dofs(),
            direct_prescribed_displacement: initial_assembly.operator.load(),
        },
        execution_step.step_id.as_str(),
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))
}

fn prepare_transient_block_basis(
    model: &AnalysisModel,
    options: &TransientSolveOptions,
) -> Result<PreparedTransientBlockBasis, FeaRunError> {
    let execution_step = super::admission::require_structural(
        model,
        AnalysisProcedure::TransientResponse,
        options.formulation_requirement,
    )?
    .clone();
    validate_model(model).map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let mut required_history_samples = required_time_history_samples(&execution_step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let time_grid = crate::excitation::ProcedureTimeGrid::from_interval_count(
        &execution_step,
        options.time_step_policy.interval_count(),
    )
    .map_err(FeaRunError::InvalidModel)?;
    let start_time_s = time_grid.start_s();
    let end_time_s = time_grid.end_s();
    let active_model = super::activation::model_with_active_constraints(model, &execution_step);
    constraint_admission::validate(&active_model)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let motions =
        StructuralMotionSet::materialize(&active_model, &execution_step, start_time_s, end_time_s)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    required_history_samples.extend(
        motions
            .breakpoints_s()
            .iter()
            .copied()
            .filter(|time| *time >= start_time_s && *time <= end_time_s),
    );
    required_history_samples.sort_by(f64::total_cmp);
    required_history_samples.dedup_by(|left, right| left.to_bits() == right.to_bits());
    let discontinuities =
        time_discontinuities(&active_model, &execution_step, start_time_s, end_time_s)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let step_boundaries = canonical_transient_step_boundaries(
        &discontinuities,
        &required_history_samples,
        start_time_s,
        end_time_s,
    )
    .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let schedule = crate::solve::transient::canonical_transient_schedule(
        options.time_step_policy.interval_count(),
        start_time_s,
        end_time_s,
        &step_boundaries,
    )
    .map_err(FeaRunError::Solve)?;
    let excitations = realize_step_excitations(
        &active_model,
        &execution_step,
        ExcitationSamplePoint::time(start_time_s),
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let initial_assembly = assemble_structural_system(
        &active_model,
        &excitations,
        options.solver_mesh.clone(),
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    Ok(PreparedTransientBlockBasis {
        execution_step,
        active_model,
        motions,
        initial_assembly,
        schedule,
    })
}

/// Derives the immutable restart scope from the same authored preparation authority used to run
/// each block. Callers cannot manufacture operator, excitation, schedule, or constraint identities.
pub fn prepare_transient_checkpoint_scope(
    workload: &StructuralTransientWorkload,
    solver_mesh: &runmat_meshing_core::SolverMeshArtifact,
    modal_state: Option<&crate::execution::CanonicalSemanticState>,
) -> Result<StructuralTransientCheckpointScope, FeaRunError> {
    workload
        .validate()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let options = workload.solve_options(solver_mesh.clone());
    let basis = prepare_transient_block_basis(&workload.model, &options)?;
    StructuralTransientCheckpointScope::from_prepared_workload(
        StructuralTransientCheckpointScopeInput {
            workload,
            solver_mesh,
            initial_assembly: &basis.initial_assembly,
            modal_state,
            schedule: basis.schedule,
        },
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))
}

pub fn run_transient_block(
    input: StructuralTransientRunBlockInput<'_>,
) -> Result<StructuralTransientSegment, FeaRunError> {
    input
        .workload
        .validate()
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let options = input.workload.solve_options(input.solver_mesh.clone());
    let basis = prepare_transient_block_basis(&input.workload.model, &options)?;
    if input.request.scope.schedule != basis.schedule {
        return Err(FeaRunError::Solve(
            "transient block request schedule differs from the authored model schedule".into(),
        ));
    }
    let expected_scope = StructuralTransientCheckpointScope::from_prepared_workload(
        StructuralTransientCheckpointScopeInput {
            workload: input.workload,
            solver_mesh: input.solver_mesh,
            initial_assembly: &basis.initial_assembly,
            modal_state: input.modal_state,
            schedule: basis.schedule.clone(),
        },
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    if input.request.scope != expected_scope {
        return Err(FeaRunError::Solve(
            "transient block request scope differs from the actual prepared workload".into(),
        ));
    }
    let initial_state = realize_transient_initial_state(
        &basis.active_model,
        &basis.execution_step,
        &basis.initial_assembly,
        Some(input.solver_mesh),
    )?;
    let initial_conditions =
        input
            .predecessor
            .is_none()
            .then_some(StructuralTransientInitialConditions {
                displacement: initial_state.displacement,
                velocity: initial_state.velocity,
            });
    let mut assemble_at = |time_s, approach| {
        let excitations = realize_step_excitations(
            &basis.active_model,
            &basis.execution_step,
            ExcitationSamplePoint::TimeSeconds {
                value: time_s,
                approach,
            },
        )
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        let assembly = assemble_structural_system(
            &basis.active_model,
            &excitations,
            Some(input.solver_mesh.clone()),
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        if assembly.dof_count != basis.initial_assembly.dof_count {
            return Err(FeaRunError::Assembly(format!(
                "transient reassembly changed structural DOF count from {} to {}",
                basis.initial_assembly.dof_count, assembly.dof_count
            )));
        }
        Ok(assembly)
    };
    let block_input = StructuralTransientBlockSolveInput {
        request: input.request,
        predecessor: input.predecessor,
        initial_summary: &basis.initial_assembly,
        options: &options,
        backend: input.workload.backend,
        initial_conditions,
        observation_cursors: input.observation_cursors,
    };
    let result = match &input.workload.solver {
        crate::execution::StructuralTransientSolverPlan::Direct if basis.motions.is_empty() => {
            solve_structural_transient_block_with_assembly(block_input, &mut assemble_at)
        }
        crate::execution::StructuralTransientSolverPlan::Direct => {
            motion::solve_prescribed_motion_block(
                motion::StructuralMotionBlockSolveInput {
                    summary: &basis.initial_assembly,
                    motions: &basis.motions,
                    options: &options,
                    backend: input.workload.backend,
                    request: input.request,
                    predecessor: input.predecessor,
                    initial_conditions: block_input.initial_conditions,
                    observation_cursors: block_input.observation_cursors,
                },
                &mut assemble_at,
            )
        }
        crate::execution::StructuralTransientSolverPlan::ModalReduced { basis_transition } => {
            if !basis.motions.is_empty() {
                return Err(FeaRunError::InvalidModel(
                    "modal transient execution with prescribed support motion requires the governed affine modal projection"
                        .into(),
                ));
            }
            let modal_state = input.modal_state.ok_or_else(|| {
                FeaRunError::InvalidModel(
                    "modal transient execution is missing its admitted modal-basis state".into(),
                )
            })?;
            let modal_basis = modal_state.payload.modal_basis().ok_or_else(|| {
                FeaRunError::InvalidModel(
                    "modal transient semantic state does not contain a modal basis".into(),
                )
            })?;
            solve_structural_modal_transient_block_with_assembly(
                StructuralModalTransientBlockSolveInput {
                    block: block_input,
                    basis_state: basis_transition.source_state,
                    basis: modal_basis,
                },
                &mut assemble_at,
            )
        }
    };
    result.map_err(|error| match error {
        TransientSolveError::Assembly(error) => error,
        other => FeaRunError::Solve(other.to_string()),
    })
}

pub fn run_transient_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: TransientSolveOptions,
) -> Result<FeaTransientRunResult, FeaRunError> {
    let execution_step = super::admission::require_structural(
        model,
        AnalysisProcedure::TransientResponse,
        options.formulation_requirement,
    )?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating transient FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    let mut required_history_samples = required_time_history_samples(execution_step)
        .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "transient model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    let time_grid = crate::excitation::ProcedureTimeGrid::from_interval_count(
        execution_step,
        options.time_step_policy.interval_count(),
    )
    .map_err(FeaRunError::InvalidModel)?;
    let start_time_s = time_grid.start_s();
    let end_time_s = time_grid.end_s();
    let active_model = super::activation::model_with_active_constraints(model, execution_step);
    constraint_admission::validate(&active_model)
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let structural_motions =
        StructuralMotionSet::materialize(&active_model, execution_step, start_time_s, end_time_s)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let excitations = realize_step_excitations(
        &active_model,
        execution_step,
        ExcitationSamplePoint::time(start_time_s),
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling transient system",
        Some(1),
        Some(5),
    );
    let summary = assemble_structural_system(
        &active_model,
        &excitations,
        options.solver_mesh.clone(),
        StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .map_err(|err| FeaRunError::Assembly(err.to_string()))?;
    let identity = super::admission::admit_structural(
        &summary,
        execution_step,
        options.formulation_requirement,
    )?;
    super::validate_rotational_dof_targets(model, &summary)?;
    let initial_state = realize_transient_initial_state(
        &active_model,
        execution_step,
        &summary,
        options.solver_mesh.as_ref(),
    )?;
    let initial_state_evidence = initial_state.evidence.clone();
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "transient system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving transient steps",
        Some(2),
        Some(5),
    );
    required_history_samples.extend(
        structural_motions
            .breakpoints_s()
            .iter()
            .copied()
            .filter(|time| *time >= start_time_s && *time <= end_time_s),
    );
    required_history_samples.sort_by(f64::total_cmp);
    required_history_samples.dedup_by(|left, right| left.to_bits() == right.to_bits());
    let discontinuities =
        time_discontinuities(&active_model, execution_step, start_time_s, end_time_s)
            .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
    let step_boundaries = canonical_transient_step_boundaries(
        &discontinuities,
        &required_history_samples,
        start_time_s,
        end_time_s,
    )
    .map_err(|error| FeaRunError::Observation(error.to_string()))?;
    let assemble_at = |time_s, approach| {
        let excitations = realize_step_excitations(
            &active_model,
            execution_step,
            ExcitationSamplePoint::TimeSeconds {
                value: time_s,
                approach,
            },
        )
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
        let assembly = assemble_structural_system(
            &active_model,
            &excitations,
            options.solver_mesh.clone(),
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        if assembly.dof_count != summary.dof_count {
            return Err(FeaRunError::Assembly(format!(
                "transient reassembly changed structural DOF count from {} to {}",
                summary.dof_count, assembly.dof_count
            )));
        }
        Ok::<AssemblySummary, FeaRunError>(assembly)
    };
    let transient = if structural_motions.is_empty() {
        solve_transient_system_with_assembly(
            &summary,
            options.clone(),
            backend,
            TransientTimeDomain {
                start_time_s,
                end_time_s,
                step_boundaries: &step_boundaries,
            },
            initial_state.displacement,
            initial_state.velocity,
            assemble_at,
        )
    } else {
        motion::solve_prescribed_motion(
            motion::StructuralMotionSolveInput {
                summary: &summary,
                motions: &structural_motions,
                options: options.clone(),
                backend,
                time_domain: TransientTimeDomain {
                    start_time_s,
                    end_time_s,
                    step_boundaries: &step_boundaries,
                },
                initial_displacement: initial_state.displacement,
                initial_velocity: initial_state.velocity,
            },
            assemble_at,
        )
    }
    .map_err(|error| match error {
        TransientSolveError::Assembly(error) => error,
        other => FeaRunError::Solve(other.to_string()),
    })?;
    finish_transient_run(
        TransientRunFinalization {
            model,
            execution_step,
            active_model: &active_model,
            backend,
            options: &options,
            identity,
            initial_state: initial_state_evidence,
        },
        transient,
    )
}
