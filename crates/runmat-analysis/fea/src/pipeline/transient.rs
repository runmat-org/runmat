use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        fea_transient_acceleration_field_id, fea_transient_displacement_field_id,
        fea_transient_kinetic_energy_field_id, fea_transient_residual_norm_field_id,
        fea_transient_strain_energy_field_id, fea_transient_velocity_field_id,
        fea_transient_von_mises_field_id, ComputeBackend, FeaRunError, FeaRunResult,
        FeaTransientRunResult, FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::{
        builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
        FeaDiagnostic, FeaDiagnosticSeverity,
    },
    operator::{apply_k_unconstrained, apply_m},
    pipeline::electro_thermal::recover_electro_thermal_fields,
    pipeline::thermo_mechanical::recover_thermo_mechanical_snapshots,
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::transient::{solve_transient_system, TransientSolveOptions},
};

const VECTOR_COMPONENT_COUNT: usize = 3;

pub fn run_transient(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaTransientRunResult, FeaRunError> {
    run_transient_with_options(model, backend, TransientSolveOptions::default())
}

pub fn run_transient_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: TransientSolveOptions,
) -> Result<FeaTransientRunResult, FeaRunError> {
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
    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "transient model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_transient")?;
    let prep_context = options.prep_context;
    let thermo_context = options.thermo_mechanical_context.clone();
    let electro_context = options.electro_thermal_context.clone();

    emit_phase(
        "fea.run_transient",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling transient system",
        Some(1),
        Some(5),
    );
    let summary = assemble_linear_system(model, prep_context, thermo_context, electro_context);
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
    let transient = solve_transient_system(&summary, options.clone(), backend);
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
            summary: &summary,
            prep_context,
            iteration_metric: transient.converged_steps as f64,
            residual_metric: transient
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            requested_preconditioner: "auto",
            effective_preconditioner: &transient.preconditioner,
        },
    );

    let displacement = transient
        .displacement_snapshots
        .last()
        .cloned()
        .unwrap_or_else(|| vec![0.0; summary.dof_count.max(3)]);
    let von_mises = displacement
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        * 1.0e11;

    let mut run = FeaRunResult {
        backend,
        solver_backend: transient.solver_backend,
        solver_device_apply_k_ratio: if transient.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            transient.device_apply_k_count as f64 / transient.device_apply_k_attempt_count as f64
        },
        solver_method: transient.solver_method,
        preconditioner: transient.preconditioner,
        solver_host_sync_count: transient.solver_host_sync_count,
        diagnostics,
        fields: vec![
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_DISPLACEMENT,
                vec![displacement.len()],
                displacement,
            ),
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![1], vec![von_mises]),
        ],
    };

    let velocity_values = recover_velocity_snapshots(
        &transient.displacement_snapshots,
        &transient.time_points_s,
        summary.dof_count,
    );
    let acceleration_values = recover_acceleration_snapshots(
        &velocity_values,
        &transient.time_points_s,
        summary.dof_count,
    );
    let von_mises_values =
        recover_von_mises_snapshots(&transient.displacement_snapshots, summary.dof_count);
    let kinetic_energy_values = recover_kinetic_energy_snapshots(&summary, &velocity_values);
    let strain_energy_values =
        recover_strain_energy_snapshots(&summary, &transient.displacement_snapshots);
    run.diagnostics.push(transient_energy_balance_diagnostic(
        &kinetic_energy_values,
        &strain_energy_values,
    ));
    let residual_norm_values = recover_residual_norm_snapshots(
        transient.displacement_snapshots.len(),
        &transient.residual_norms,
    );

    let displacement_snapshots = transient
        .displacement_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_displacement_field_id(index),
                vector_shape(summary.dof_count.max(snapshot.len())),
                padded_vector_values(snapshot, summary.dof_count),
            )
        })
        .collect::<Vec<_>>();
    let velocity_snapshots = velocity_values
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_velocity_field_id(index),
                vector_shape(summary.dof_count.max(snapshot.len())),
                padded_vector_values(snapshot, summary.dof_count),
            )
        })
        .collect::<Vec<_>>();
    let acceleration_snapshots = acceleration_values
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_transient_acceleration_field_id(index),
                vector_shape(summary.dof_count.max(snapshot.len())),
                padded_vector_values(snapshot, summary.dof_count),
            )
        })
        .collect::<Vec<_>>();
    let von_mises_snapshots = von_mises_values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_transient_von_mises_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
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
    let thermo_mechanical_fields = recover_thermo_mechanical_snapshots(
        summary.thermo_mechanical.as_ref(),
        &normalized_time_factors(&transient.time_points_s, displacement_snapshots.len()),
        &displacement_snapshots,
        &von_mises_snapshots,
        &residual_norm_values,
        element_count_for_dofs(summary.dof_count),
    );
    let electro_thermal_fields = recover_electro_thermal_fields(
        summary.electro_thermal.as_ref(),
        &normalized_time_factors(&transient.time_points_s, displacement_snapshots.len()),
        &residual_norm_values,
        summary.dof_count,
    );
    run.fields.extend(electro_thermal_fields.static_fields);

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
        time_points_s: transient.time_points_s,
        displacement_snapshots,
        velocity_snapshots,
        acceleration_snapshots,
        von_mises_snapshots,
        kinetic_energy_snapshots,
        strain_energy_snapshots,
        residual_norm_snapshots,
        thermo_mechanical_temperature_snapshots: thermo_mechanical_fields.temperature_snapshots,
        thermo_mechanical_thermal_strain_snapshots: thermo_mechanical_fields
            .thermal_strain_snapshots,
        thermo_mechanical_thermal_stress_snapshots: thermo_mechanical_fields
            .thermal_stress_snapshots,
        thermo_mechanical_displacement_snapshots: thermo_mechanical_fields.displacement_snapshots,
        thermo_mechanical_von_mises_snapshots: thermo_mechanical_fields.von_mises_snapshots,
        thermo_mechanical_coupling_residual_snapshots: thermo_mechanical_fields
            .coupling_residual_snapshots,
        electro_thermal_temperature_snapshots: electro_thermal_fields.temperature_snapshots,
        electro_thermal_thermal_residual_snapshots: electro_thermal_fields
            .thermal_residual_snapshots,
        residual_norms: transient.residual_norms,
    })
}

fn normalized_time_factors(time_points_s: &[f64], snapshot_count: usize) -> Vec<f64> {
    if snapshot_count == 0 {
        return Vec::new();
    }
    let end_time = time_points_s
        .last()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or((snapshot_count.saturating_sub(1)).max(1) as f64);
    (0..snapshot_count)
        .map(|index| {
            time_points_s
                .get(index)
                .copied()
                .unwrap_or(index as f64)
                .clamp(0.0, end_time)
                / end_time
        })
        .collect()
}

fn recover_velocity_snapshots(
    displacement_snapshots: &[Vec<f64>],
    time_points_s: &[f64],
    dof_count: usize,
) -> Vec<Vec<f64>> {
    let mut velocities = Vec::with_capacity(displacement_snapshots.len());
    for (index, snapshot) in displacement_snapshots.iter().enumerate() {
        if index == 0 {
            velocities.push(vec![0.0; dof_count.max(snapshot.len())]);
            continue;
        }
        let previous = &displacement_snapshots[index - 1];
        let dt = (time_points_s.get(index).copied().unwrap_or(index as f64)
            - time_points_s
                .get(index - 1)
                .copied()
                .unwrap_or(index.saturating_sub(1) as f64))
        .abs()
        .max(1.0e-12);
        velocities.push(difference_quotient(snapshot, previous, dt, dof_count));
    }
    velocities
}

fn recover_acceleration_snapshots(
    velocity_snapshots: &[Vec<f64>],
    time_points_s: &[f64],
    dof_count: usize,
) -> Vec<Vec<f64>> {
    let mut accelerations = Vec::with_capacity(velocity_snapshots.len());
    for (index, snapshot) in velocity_snapshots.iter().enumerate() {
        if index == 0 {
            accelerations.push(vec![0.0; dof_count.max(snapshot.len())]);
            continue;
        }
        let previous = &velocity_snapshots[index - 1];
        let dt = (time_points_s.get(index).copied().unwrap_or(index as f64)
            - time_points_s
                .get(index - 1)
                .copied()
                .unwrap_or(index.saturating_sub(1) as f64))
        .abs()
        .max(1.0e-12);
        accelerations.push(difference_quotient(snapshot, previous, dt, dof_count));
    }
    accelerations
}

fn difference_quotient(current: &[f64], previous: &[f64], dt: f64, dof_count: usize) -> Vec<f64> {
    let count = dof_count.max(current.len()).max(previous.len());
    (0..count)
        .map(|index| {
            (current.get(index).copied().unwrap_or(0.0)
                - previous.get(index).copied().unwrap_or(0.0))
                / dt
        })
        .collect()
}

fn recover_von_mises_snapshots(displacement_snapshots: &[Vec<f64>], dof_count: usize) -> Vec<f64> {
    displacement_snapshots
        .iter()
        .map(|snapshot| {
            snapshot
                .iter()
                .take(dof_count.max(snapshot.len()))
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
                * 1.0e11
        })
        .collect()
}

fn recover_kinetic_energy_snapshots(
    summary: &crate::assembly::AssemblySummary,
    velocity_snapshots: &[Vec<f64>],
) -> Vec<f64> {
    velocity_snapshots
        .iter()
        .map(|velocity| {
            let momentum = apply_m(&summary.operator, velocity);
            0.5 * velocity
                .iter()
                .zip(momentum.iter())
                .map(|(v, p)| v * p)
                .sum::<f64>()
        })
        .collect()
}

fn recover_strain_energy_snapshots(
    summary: &crate::assembly::AssemblySummary,
    displacement_snapshots: &[Vec<f64>],
) -> Vec<f64> {
    displacement_snapshots
        .iter()
        .map(|displacement| {
            let internal_force = apply_k_unconstrained(&summary.operator, displacement);
            0.5 * displacement
                .iter()
                .zip(internal_force.iter())
                .map(|(u, force)| u * force)
                .sum::<f64>()
        })
        .collect()
}

fn recover_residual_norm_snapshots(snapshot_count: usize, residual_norms: &[f64]) -> Vec<f64> {
    (0..snapshot_count)
        .map(|index| {
            if index == 0 {
                0.0
            } else {
                residual_norms
                    .get(index - 1)
                    .copied()
                    .unwrap_or(f64::INFINITY)
            }
        })
        .collect()
}

fn transient_energy_balance_diagnostic(
    kinetic_energy_values: &[f64],
    strain_energy_values: &[f64],
) -> FeaDiagnostic {
    let snapshot_count = kinetic_energy_values.len().max(strain_energy_values.len());
    let total_energy = (0..snapshot_count)
        .map(|index| {
            kinetic_energy_values.get(index).copied().unwrap_or(0.0)
                + strain_energy_values.get(index).copied().unwrap_or(0.0)
        })
        .collect::<Vec<_>>();
    let initial_total_energy = total_energy.first().copied().unwrap_or(0.0);
    let final_total_energy = total_energy.last().copied().unwrap_or(0.0);
    let max_total_energy = total_energy.iter().copied().fold(0.0_f64, f64::max);
    let reference_energy = total_energy
        .iter()
        .copied()
        .find(|value| value.is_finite() && *value > 1.0e-12)
        .unwrap_or(max_total_energy.max(1.0));
    let energy_growth_ratio = if reference_energy > 0.0 {
        max_total_energy / reference_energy
    } else {
        0.0
    };
    let max_step_energy_jump_ratio = total_energy
        .windows(2)
        .map(|window| {
            let previous = window[0].abs().max(reference_energy).max(1.0e-12);
            (window[1] - window[0]).abs() / previous
        })
        .fold(0.0_f64, f64::max);
    let metrics_are_finite = [
        initial_total_energy,
        final_total_energy,
        max_total_energy,
        energy_growth_ratio,
        max_step_energy_jump_ratio,
    ]
    .iter()
    .all(|value| value.is_finite());
    FeaDiagnostic {
        code: "FEA_TRANSIENT_ENERGY_BALANCE".to_string(),
        severity: if metrics_are_finite && energy_growth_ratio <= 10.0 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "initial_total_energy={} final_total_energy={} max_total_energy={} energy_growth_ratio={} max_step_energy_jump_ratio={}",
            initial_total_energy,
            final_total_energy,
            max_total_energy,
            energy_growth_ratio,
            max_step_energy_jump_ratio,
        ),
    }
}

fn vector_shape(dof_count: usize) -> Vec<usize> {
    vec![
        dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1),
        VECTOR_COMPONENT_COUNT,
    ]
}

fn element_count_for_dofs(dof_count: usize) -> usize {
    dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1)
}

fn padded_vector_values(mut values: Vec<f64>, dof_count: usize) -> Vec<f64> {
    let shape = vector_shape(dof_count.max(values.len()));
    values.resize(shape.iter().product(), 0.0);
    values
}
