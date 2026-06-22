use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::assembly::{dofs::StructuralDofKind, AssemblySummary};
use crate::{
    assembly::assemble_linear_system,
    contracts::{
        fea_nonlinear_contact_gap_field_id, fea_nonlinear_contact_pressure_field_id,
        fea_nonlinear_displacement_field_id, fea_nonlinear_equivalent_plastic_strain_field_id,
        fea_nonlinear_load_factor_field_id, fea_nonlinear_plastic_strain_field_id,
        fea_nonlinear_residual_norm_field_id, fea_nonlinear_rotation_field_id,
        fea_nonlinear_von_mises_field_id, ComputeBackend, FeaContactInterfaceContext,
        FeaNonlinearRunResult, FeaRunError, FeaRunResult, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
    pipeline::electro_thermal::recover_electro_thermal_fields,
    pipeline::thermo_mechanical::recover_thermo_mechanical_snapshots,
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::nonlinear::{solve_nonlinear_system, NonlinearSolveOptions},
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const TENSOR_COMPONENT_COUNT: usize = 6;

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
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating nonlinear FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "nonlinear model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    let prep_context = options.prep_context.clone();
    let thermo_context = options.thermo_mechanical_context.clone();
    let electro_context = options.electro_thermal_context.clone();

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling nonlinear system",
        Some(1),
        Some(5),
    );
    let summary =
        assemble_linear_system(model, prep_context.clone(), thermo_context, electro_context);
    super::reject_moment_loads_without_rotational_dofs(model, &summary)?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "nonlinear system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving nonlinear increments",
        Some(2),
        Some(5),
    );
    let nonlinear = solve_nonlinear_system(&summary, options.clone(), backend);
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "nonlinear solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering nonlinear fields",
        Some(3),
        Some(5),
    );
    let mut diagnostics = nonlinear.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
            prep_context,
            iteration_metric: nonlinear
                .iteration_counts
                .iter()
                .copied()
                .max()
                .unwrap_or(0) as f64,
            residual_metric: nonlinear
                .residual_norms
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            requested_preconditioner: "auto",
            effective_preconditioner: &nonlinear.preconditioner,
        },
    );

    let displacement = nonlinear
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
        solver_backend: nonlinear.solver_backend,
        solver_device_apply_k_ratio: if nonlinear.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            nonlinear.device_apply_k_count as f64 / nonlinear.device_apply_k_attempt_count as f64
        },
        solver_method: nonlinear.solver_method,
        preconditioner: nonlinear.preconditioner,
        solver_host_sync_count: nonlinear.solver_host_sync_count,
        diagnostics,
        fields: vec![
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_DISPLACEMENT,
                vector_shape(displacement.len()),
                padded_vector_values(displacement, summary.dof_count),
            ),
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![1], vec![von_mises]),
        ],
    };

    let element_count = element_count_for_dofs(summary.dof_count);
    let rotation_values = recover_rotational_snapshots(&summary, &nonlinear.displacement_snapshots);

    let displacement_snapshots = nonlinear
        .displacement_snapshots
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_nonlinear_displacement_field_id(index),
                vector_shape(snapshot.len()),
                padded_vector_values(snapshot, summary.dof_count),
            )
        })
        .collect::<Vec<_>>();
    let rotation_snapshots = rotation_values
        .into_iter()
        .enumerate()
        .map(|(index, snapshot)| {
            AnalysisField::host_f64(
                fea_nonlinear_rotation_field_id(index),
                rotational_vector_shape(&summary),
                snapshot,
            )
        })
        .collect::<Vec<_>>();
    let von_mises_snapshots = nonlinear
        .von_mises_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_von_mises_field_id(index),
                vec![element_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let plastic_strain_snapshots = nonlinear
        .plastic_strain_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_plastic_strain_field_id(index),
                vec![element_count, TENSOR_COMPONENT_COUNT],
                values,
            )
        })
        .collect::<Vec<_>>();
    let equivalent_plastic_strain_snapshots = nonlinear
        .equivalent_plastic_strain_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_equivalent_plastic_strain_field_id(index),
                vec![element_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let contact_count = contact_entity_count(options.contact_context.as_ref());
    let contact_pressure_snapshots = nonlinear
        .contact_pressure_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_contact_pressure_field_id(index),
                vec![contact_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let contact_gap_snapshots = nonlinear
        .contact_gap_snapshots
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_contact_gap_field_id(index),
                vec![contact_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let load_factor_snapshots = nonlinear
        .load_factors
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_nonlinear_load_factor_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let residual_norm_snapshots = nonlinear
        .residual_norms
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            AnalysisField::host_f64(
                fea_nonlinear_residual_norm_field_id(index),
                vec![1],
                vec![value],
            )
        })
        .collect::<Vec<_>>();
    let thermo_mechanical_fields = recover_thermo_mechanical_snapshots(
        summary.thermo_mechanical.as_ref(),
        &nonlinear.load_factors,
        &displacement_snapshots,
        &von_mises_snapshots,
        &nonlinear.residual_norms,
        element_count,
    );
    let electro_thermal_fields = recover_electro_thermal_fields(
        summary.electro_thermal.as_ref(),
        &nonlinear.load_factors,
        &nonlinear.residual_norms,
        summary.dof_count,
    );
    run.diagnostics.extend(thermo_mechanical_fields.diagnostics);
    run.diagnostics.extend(electro_thermal_fields.diagnostics);
    run.fields.extend(electro_thermal_fields.static_fields);

    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "nonlinear result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_nonlinear")?;
    emit_phase(
        "fea.run_nonlinear",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA nonlinear run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaNonlinearRunResult {
        run,
        load_factors: nonlinear.load_factors,
        displacement_snapshots,
        rotation_snapshots,
        von_mises_snapshots,
        plastic_strain_snapshots,
        equivalent_plastic_strain_snapshots,
        contact_pressure_snapshots,
        contact_gap_snapshots,
        load_factor_snapshots,
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
        residual_norms: nonlinear.residual_norms,
        increment_norms: nonlinear.increment_norms,
        iteration_counts: nonlinear.iteration_counts,
        failed_increments: nonlinear.failed_increments,
        line_search_backtracks: nonlinear.line_search_backtracks,
        max_line_search_backtracks_per_increment: nonlinear
            .max_line_search_backtracks_per_increment,
        tangent_rebuild_count: nonlinear.tangent_rebuild_count,
        iteration_spike_count: nonlinear.iteration_spike_count,
        convergence_stall_count: nonlinear.convergence_stall_count,
        backtrack_burst_count: nonlinear.backtrack_burst_count,
    })
}

fn recover_rotational_snapshots(
    summary: &AssemblySummary,
    snapshots: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    if summary.structural_rotational_dof_count == 0 {
        return Vec::new();
    }
    snapshots
        .iter()
        .map(|snapshot| recover_rotation_snapshot(summary, snapshot))
        .collect()
}

fn recover_rotation_snapshot(summary: &AssemblySummary, snapshot: &[f64]) -> Vec<f64> {
    let mut rotation = vec![0.0; rotational_vector_shape(summary).iter().product()];
    for row in 0..summary.structural_dof_layout.total_dof_count() {
        let Some(address) = summary.structural_dof_layout.address(row) else {
            continue;
        };
        let Some(component) = rotational_component(address.kind) else {
            continue;
        };
        let target = address.node_index * VECTOR_COMPONENT_COUNT + component;
        if target < rotation.len() {
            rotation[target] = snapshot.get(row).copied().unwrap_or(0.0);
        }
    }
    rotation
}

fn rotational_vector_shape(summary: &AssemblySummary) -> Vec<usize> {
    vec![
        summary.structural_dof_layout.node_count(),
        VECTOR_COMPONENT_COUNT,
    ]
}

fn rotational_component(kind: StructuralDofKind) -> Option<usize> {
    match kind {
        StructuralDofKind::Rx => Some(0),
        StructuralDofKind::Ry => Some(1),
        StructuralDofKind::Rz => Some(2),
        _ => None,
    }
}

fn element_count_for_dofs(dof_count: usize) -> usize {
    dof_count
        .div_ceil(VECTOR_COMPONENT_COUNT)
        .saturating_sub(1)
        .max(1)
}

fn vector_shape(dof_count: usize) -> Vec<usize> {
    vec![
        dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1),
        VECTOR_COMPONENT_COUNT,
    ]
}

fn padded_vector_values(mut values: Vec<f64>, dof_count: usize) -> Vec<f64> {
    let shape = vector_shape(dof_count.max(values.len()));
    values.resize(shape.iter().product(), 0.0);
    values
}

fn contact_entity_count(contact: Option<&FeaContactInterfaceContext>) -> usize {
    if contact.is_some_and(|context| context.enabled) {
        1
    } else {
        0
    }
}
