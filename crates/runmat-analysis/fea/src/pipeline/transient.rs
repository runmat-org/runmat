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
        FeaRunError, FeaRunResult, FeaTransientRunResult, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
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
        realize as realize_structural_initial_state, StructuralInitialConstraintView,
        StructuralInitialStateRealizationBasis,
    },
    pipeline::electro_thermal::recover_realized_electro_thermal_fields,
    pipeline::observation_history::{
        evaluate_field_histories, required_time_history_samples, HistoryFieldSeries,
        HistoryTopology,
    },
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::transient::{
        canonical_transient_step_boundaries, mechanical_energy_history_diagnostic,
        solve_structural_transient_block_with_assembly, solve_transient_system_with_assembly,
        StructuralTransientBlockSolveInput, StructuralTransientInitialConditions,
        TransientSolveError, TransientSolveOptions, TransientTimeDomain,
    },
};

mod constraint_admission;
mod motion;
mod recovery;

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
    pub model: &'a AnalysisModel,
    pub backend: ComputeBackend,
    pub options: &'a TransientSolveOptions,
    pub request: &'a StructuralTransientBlockRequest,
    pub predecessor: Option<&'a StructuralTransientCheckpoint>,
    pub observation_cursors: Vec<StructuralTransientObservationCursor>,
}

struct PreparedTransientBlockBasis {
    execution_step: ProcedureStep,
    active_model: AnalysisModel,
    motions: StructuralMotionSet,
    initial_assembly: AssemblySummary,
    schedule: crate::execution::StructuralTransientSchedule,
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
            schedule: basis.schedule,
        },
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))
}

pub fn run_transient_block(
    input: StructuralTransientRunBlockInput<'_>,
) -> Result<StructuralTransientSegment, FeaRunError> {
    let basis = prepare_transient_block_basis(input.model, input.options)?;
    if input.request.scope.schedule != basis.schedule {
        return Err(FeaRunError::Solve(
            "transient block request schedule differs from the authored model schedule".into(),
        ));
    }
    let solver_mesh = input.options.solver_mesh.as_ref().ok_or_else(|| {
        FeaRunError::InvalidModel(
            "portable structural transient block execution requires a canonical solver mesh".into(),
        )
    })?;
    let workload = StructuralTransientWorkload {
        schema_version: 1,
        model: input.model.clone(),
        step_id: basis.execution_step.step_id.clone(),
        backend: input.backend,
        formulation_requirement: input.options.formulation_requirement,
        time_step_policy: input.options.time_step_policy,
        residual_target: input.options.residual_target,
    };
    let expected_scope = StructuralTransientCheckpointScope::from_prepared_workload(
        StructuralTransientCheckpointScopeInput {
            workload: &workload,
            solver_mesh,
            initial_assembly: &basis.initial_assembly,
            schedule: basis.schedule.clone(),
        },
    )
    .map_err(|error| FeaRunError::Solve(error.to_string()))?;
    if input.request.scope != expected_scope {
        return Err(FeaRunError::Solve(
            "transient block request scope differs from the actual prepared workload".into(),
        ));
    }
    let initial_state_basis = match basis.initial_assembly.structural_formulations.as_slice() {
        [realization]
            if realization.operator
                == runmat_analysis_core::StructuralOperatorImplementation::SyntheticDiagonalChain =>
        {
            StructuralInitialStateRealizationBasis::OperatorBoundZeroDefault {
                dof_count: basis.initial_assembly.operator.dof_count(),
                operator_identity: basis
                    .initial_assembly
                    .operator
                    .operators()
                    .stiffness()
                    .numerical_identity()
                    .map_err(|error| FeaRunError::Assembly(error.to_string()))?,
            }
        }
        _ => StructuralInitialStateRealizationBasis::CanonicalTopology,
    };
    let initial_state = realize_structural_initial_state(
        &basis.active_model,
        input.options.solver_mesh.as_ref(),
        &basis.initial_assembly.structural_dof_layout,
        initial_state_basis,
        StructuralInitialConstraintView {
            constraint_matrix: basis.initial_assembly.constraint_matrix.as_ref(),
            affine_reduction: basis.initial_assembly.affine_constraint_reduction.as_ref(),
            direct_constrained_dofs: basis.initial_assembly.operator.constrained_dofs(),
            direct_prescribed_displacement: basis.initial_assembly.operator.load(),
        },
        basis.execution_step.step_id.as_str(),
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
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
            input.options.solver_mesh.clone(),
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
    let result = if basis.motions.is_empty() {
        solve_structural_transient_block_with_assembly(
            StructuralTransientBlockSolveInput {
                request: input.request,
                predecessor: input.predecessor,
                initial_summary: &basis.initial_assembly,
                options: input.options,
                backend: input.backend,
                initial_conditions,
                observation_cursors: input.observation_cursors,
            },
            &mut assemble_at,
        )
    } else {
        motion::solve_prescribed_motion_block(
            motion::StructuralMotionBlockSolveInput {
                summary: &basis.initial_assembly,
                motions: &basis.motions,
                options: input.options,
                backend: input.backend,
                request: input.request,
                predecessor: input.predecessor,
                initial_conditions,
                observation_cursors: input.observation_cursors,
            },
            &mut assemble_at,
        )
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
    let initial_state_basis = match summary.structural_formulations.as_slice() {
        [realization]
            if realization.operator
                == runmat_analysis_core::StructuralOperatorImplementation::SyntheticDiagonalChain =>
        {
            StructuralInitialStateRealizationBasis::OperatorBoundZeroDefault {
                dof_count: summary.operator.dof_count(),
                operator_identity: summary
                    .operator
                    .operators()
                    .stiffness()
                    .numerical_identity()
                    .map_err(|error| FeaRunError::Assembly(error.to_string()))?,
            }
        }
        _ => StructuralInitialStateRealizationBasis::CanonicalTopology,
    };
    let initial_state = realize_structural_initial_state(
        &active_model,
        options.solver_mesh.as_ref(),
        &summary.structural_dof_layout,
        initial_state_basis,
        StructuralInitialConstraintView {
            constraint_matrix: summary.constraint_matrix.as_ref(),
            affine_reduction: summary.affine_constraint_reduction.as_ref(),
            direct_constrained_dofs: summary.operator.constrained_dofs(),
            direct_prescribed_displacement: summary.operator.load(),
        },
        execution_step.step_id.as_str(),
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
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
    let reporting_summary = transient
        .assembly_snapshots
        .last()
        .cloned()
        .unwrap_or_else(|| summary.clone());
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "transient solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering transient fields",
        Some(3),
        Some(5),
    );
    let mut diagnostics = transient.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &reporting_summary,
        },
    );

    let recovered = recovery::recover_transient_fields(&transient)?;
    let kinetic_energy_values = transient.kinetic_energy_snapshots.clone();
    let strain_energy_values = transient.strain_energy_snapshots.clone();
    recovery::require_snapshot_count(
        transient.displacement_snapshots.len(),
        "residual norm",
        transient.residual_norms.len(),
    )?;
    let residual_norm_values = transient.residual_norms.clone();
    let mut run = FeaRunResult {
        identity,
        requested_backend: backend,
        execution_backend: crate::contracts::ExecutionBackend::from_solver_backend(
            transient.solver_backend,
        ),
        solver_backend: transient.solver_backend,
        solver_device_apply_k_ratio: if transient.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            transient.device_apply_k_count as f64 / transient.device_apply_k_attempt_count as f64
        },
        solver_method: transient.solver_method,
        preconditioner: transient.preconditioner,
        solver_host_sync_count: transient.solver_host_sync_count,
        accelerator_evidence: transient.accelerator_evidence.clone(),
        diagnostics,
        spatial_projection_evidence: transient
            .assembly_snapshots
            .iter()
            .flat_map(|summary| summary.spatial_projection_evidence.iter().cloned())
            .collect(),
        observations: Vec::new(),
        observation_histories: Vec::new(),
        series_reductions: Vec::new(),
        fields: recovered.final_deformation_fields.clone(),
    };
    run.diagnostics.push(mechanical_energy_history_diagnostic(
        &kinetic_energy_values,
        &strain_energy_values,
        &transient.external_work_j,
        &transient.support_work_j,
        &transient.damping_dissipation_j,
        &transient.energy_balance_residual_j,
        &transient.relative_energy_balance_error,
    ));

    let displacement_snapshots = recovered.displacement_snapshots;
    let rotation_snapshots = recovered.rotation_snapshots;
    let velocity_snapshots = recovered.velocity_snapshots;
    let angular_velocity_snapshots = recovered.angular_velocity_snapshots;
    let acceleration_snapshots = recovered.acceleration_snapshots;
    let angular_acceleration_snapshots = recovered.angular_acceleration_snapshots;
    let reaction_force_snapshots = recovered.reaction_force_snapshots;
    let reaction_moment_snapshots = recovered.reaction_moment_snapshots;
    let strain_snapshots = recovered.strain_snapshots;
    let stress_snapshots = recovered.stress_snapshots;
    let von_mises_snapshots = recovered.von_mises_snapshots;
    let strain_energy_density_snapshots = recovered.strain_energy_density_snapshots;
    let kinetic_energy_snapshots = kinetic_energy_values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_kinetic_energy_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let strain_energy_snapshots = strain_energy_values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_strain_energy_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let residual_norm_snapshots = residual_norm_values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_residual_norm_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let external_work_snapshots = scalar_snapshots(
        &transient.external_work_j,
        fea_transient_external_work_field_id,
    );
    let support_work_snapshots = scalar_snapshots(
        &transient.support_work_j,
        fea_transient_support_work_field_id,
    );
    let damping_dissipation_snapshots = scalar_snapshots(
        &transient.damping_dissipation_j,
        fea_transient_damping_dissipation_field_id,
    );
    let energy_balance_residual_snapshots = scalar_snapshots(
        &transient.energy_balance_residual_j,
        fea_transient_energy_balance_residual_field_id,
    );
    let relative_energy_balance_error_snapshots = scalar_snapshots(
        &transient.relative_energy_balance_error,
        fea_transient_relative_energy_balance_error_field_id,
    );
    if let Some(field) = reaction_force_snapshots.last() {
        let mut field = field.clone();
        field.field_id = FEA_FIELD_STRUCTURAL_REACTION_FORCE.into();
        run.fields.push(field);
    }
    if let Some(field) = reaction_moment_snapshots.last() {
        let mut field = field.clone();
        field.field_id = FEA_FIELD_STRUCTURAL_REACTION_MOMENT.into();
        run.fields.push(field);
    }
    let electro_thermal_summaries = transient
        .assembly_snapshots
        .iter()
        .map(|summary| summary.electro_thermal.clone())
        .collect::<Vec<_>>();
    let electro_thermal_fields = recover_realized_electro_thermal_fields(
        &electro_thermal_summaries,
        &residual_norm_values,
        reporting_summary.dof_count,
        options.solver_mesh.as_ref(),
    );
    run.diagnostics.extend(electro_thermal_fields.diagnostics);
    run.fields.extend(electro_thermal_fields.static_fields);
    let evaluated_histories = if execution_step.observations.iter().any(|request| {
        matches!(
            request,
            runmat_analysis_core::ObservationRequest::History(_)
        )
    }) {
        let coordinates = transient
            .time_points_s
            .iter()
            .copied()
            .map(crate::results::field::RecoveredFieldSample::TimeSeconds)
            .collect::<Vec<_>>();
        let mut series = vec![
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Displacement,
                source_field_id: TRANSIENT_DISPLACEMENT_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &displacement_snapshots,
            },
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Velocity,
                source_field_id: TRANSIENT_VELOCITY_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &velocity_snapshots,
            },
            HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Acceleration,
                source_field_id: TRANSIENT_ACCELERATION_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &acceleration_snapshots,
            },
        ];
        if rotation_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Rotation,
                source_field_id: TRANSIENT_ROTATION_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Node,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::StructuralNodes,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &rotation_snapshots,
            });
        }
        if strain_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Strain,
                source_field_id: TRANSIENT_STRAIN_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Element,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                value_shape: runmat_analysis_core::ObservationValueShape::SymmetricTensor3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &strain_snapshots,
            });
        }
        if stress_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::Stress,
                source_field_id: TRANSIENT_STRESS_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Element,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                value_shape: runmat_analysis_core::ObservationValueShape::SymmetricTensor3,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &stress_snapshots,
            });
        }
        if strain_energy_density_snapshots.len() == coordinates.len() {
            series.push(HistoryFieldSeries {
                quantity: runmat_analysis_core::ObservationQuantity::StrainEnergy,
                source_field_id: TRANSIENT_STRAIN_ENERGY_DENSITY_HISTORY_SOURCE,
                location: runmat_analysis_core::FieldLocation::Element,
                entity_domain: crate::results::field::RecoveredFieldEntityDomain::SolidElements,
                value_shape: runmat_analysis_core::ObservationValueShape::Scalar,
                numeric_representation:
                    crate::results::field::RecoveredFieldNumericRepresentation::Real,
                snapshots: &strain_energy_density_snapshots,
            });
        }
        evaluate_field_histories(
            execution_step,
            HistoryTopology::Structural {
                model: &active_model,
                mesh: options.solver_mesh.as_ref(),
            },
            &coordinates,
            None,
            &series,
        )
        .map_err(|error| FeaRunError::Observation(error.to_string()))?
    } else {
        Default::default()
    };
    run.observation_histories = evaluated_histories.histories;
    run.series_reductions = evaluated_histories.series_reductions;

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "transient result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA transient run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaTransientRunResult {
        run,
        initial_state: initial_state.evidence,
        time_points_s: transient.time_points_s,
        displacement_snapshots,
        rotation_snapshots,
        velocity_snapshots,
        angular_velocity_snapshots,
        acceleration_snapshots,
        angular_acceleration_snapshots,
        strain_snapshots,
        stress_snapshots,
        von_mises_snapshots,
        strain_energy_density_snapshots,
        kinetic_energy_snapshots,
        strain_energy_snapshots,
        residual_norm_snapshots,
        reaction_force_snapshots,
        reaction_moment_snapshots,
        external_work_snapshots,
        support_work_snapshots,
        damping_dissipation_snapshots,
        energy_balance_residual_snapshots,
        relative_energy_balance_error_snapshots,
        dynamic_balance_snapshots: transient.dynamic_balance_snapshots,
        electro_thermal_temperature_snapshots: electro_thermal_fields.temperature_snapshots,
        electro_thermal_thermal_residual_snapshots: electro_thermal_fields
            .thermal_residual_snapshots,
        residual_norms: transient.residual_norms,
    })
}

fn scalar_snapshots(values: &[f64], field_id: impl Fn(usize) -> String) -> Vec<AnalysisField> {
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| AnalysisField::host_f64(field_id(index), vec![1], vec![value]))
        .collect()
}
