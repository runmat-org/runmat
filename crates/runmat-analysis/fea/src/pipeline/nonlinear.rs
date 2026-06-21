use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        fea_nonlinear_contact_gap_field_id, fea_nonlinear_contact_pressure_field_id,
        fea_nonlinear_displacement_field_id, fea_nonlinear_equivalent_plastic_strain_field_id,
        fea_nonlinear_load_factor_field_id, fea_nonlinear_plastic_strain_field_id,
        fea_nonlinear_residual_norm_field_id, fea_nonlinear_von_mises_field_id, ComputeBackend,
        FeaContactInterfaceContext, FeaNonlinearRunResult, FeaPlasticityConstitutiveContext,
        FeaRunError, FeaRunResult, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
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
    let prep_context = options.prep_context;
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
    let summary = assemble_linear_system(model, prep_context, thermo_context, electro_context);
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

    let run = FeaRunResult {
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
    let von_mises_values =
        recover_von_mises_snapshots(&nonlinear.displacement_snapshots, summary.dof_count);
    let plastic_strain_values = recover_plastic_strain_snapshots(
        &nonlinear.displacement_snapshots,
        &nonlinear.load_factors,
        summary.dof_count,
        options.plasticity_context.as_ref(),
    );
    let equivalent_plastic_strain_values =
        recover_equivalent_plastic_strain_snapshots(&plastic_strain_values);
    let (contact_pressure_values, contact_gap_values) = recover_contact_snapshots(
        &nonlinear.displacement_snapshots,
        &nonlinear.load_factors,
        options.contact_context.as_ref(),
    );

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
    let von_mises_snapshots = von_mises_values
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_von_mises_field_id(index),
                vec![element_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let plastic_strain_snapshots = plastic_strain_values
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_plastic_strain_field_id(index),
                vec![element_count, TENSOR_COMPONENT_COUNT],
                values,
            )
        })
        .collect::<Vec<_>>();
    let equivalent_plastic_strain_snapshots = equivalent_plastic_strain_values
        .into_iter()
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
    let contact_pressure_snapshots = contact_pressure_values
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            AnalysisField::host_f64(
                fea_nonlinear_contact_pressure_field_id(index),
                vec![contact_count],
                values,
            )
        })
        .collect::<Vec<_>>();
    let contact_gap_snapshots = contact_gap_values
        .into_iter()
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
        von_mises_snapshots,
        plastic_strain_snapshots,
        equivalent_plastic_strain_snapshots,
        contact_pressure_snapshots,
        contact_gap_snapshots,
        load_factor_snapshots,
        residual_norm_snapshots,
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

fn recover_von_mises_snapshots(
    displacement_snapshots: &[Vec<f64>],
    dof_count: usize,
) -> Vec<Vec<f64>> {
    let element_count = element_count_for_dofs(dof_count);
    displacement_snapshots
        .iter()
        .map(|snapshot| {
            let strain = recover_increment_strain(snapshot, element_count);
            strain
                .chunks_exact(TENSOR_COMPONENT_COUNT)
                .map(von_mises_from_tensor)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn recover_plastic_strain_snapshots(
    displacement_snapshots: &[Vec<f64>],
    load_factors: &[f64],
    dof_count: usize,
    plasticity: Option<&FeaPlasticityConstitutiveContext>,
) -> Vec<Vec<f64>> {
    let element_count = element_count_for_dofs(dof_count);
    displacement_snapshots
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            let Some(plasticity) = plasticity.filter(|context| context.enabled) else {
                return vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
            };
            let load_factor = load_factors
                .get(index)
                .copied()
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            let yield_strain = plasticity.yield_strain.max(0.0);
            let hardening_scale = (1.0 + plasticity.hardening_modulus_ratio.max(0.0)).max(1.0);
            recover_increment_strain(snapshot, element_count)
                .into_iter()
                .map(|strain| {
                    let excess = (strain.abs() - yield_strain).max(0.0);
                    strain.signum() * excess * load_factor / hardening_scale
                })
                .collect()
        })
        .collect()
}

fn recover_equivalent_plastic_strain_snapshots(
    plastic_strain_snapshots: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    plastic_strain_snapshots
        .iter()
        .map(|snapshot| {
            snapshot
                .chunks_exact(TENSOR_COMPONENT_COUNT)
                .map(|tensor| {
                    ((2.0 / 3.0) * tensor.iter().map(|value| value * value).sum::<f64>()).sqrt()
                })
                .collect()
        })
        .collect()
}

fn recover_contact_snapshots(
    displacement_snapshots: &[Vec<f64>],
    load_factors: &[f64],
    contact: Option<&FeaContactInterfaceContext>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let contact_count = contact_entity_count(contact);
    displacement_snapshots
        .iter()
        .enumerate()
        .map(|(index, snapshot)| {
            let Some(contact) = contact.filter(|context| context.enabled) else {
                return (Vec::new(), Vec::new());
            };
            let load_factor = load_factors
                .get(index)
                .copied()
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            let max_penetration = contact.max_penetration_ratio.max(0.0);
            let penalty = contact.penalty_stiffness_scale.max(0.0);
            let peak_displacement = snapshot
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            let mut pressure = Vec::with_capacity(contact_count);
            let mut gap = Vec::with_capacity(contact_count);
            for entity_index in 0..contact_count {
                let entity_scale = 1.0 + 0.05 * entity_index as f64;
                let closure = load_factor * max_penetration * (1.0 + peak_displacement * 1.0e3)
                    / entity_scale;
                let signed_gap = max_penetration - closure;
                gap.push(signed_gap);
                pressure.push(penalty * closure.max(0.0));
            }
            (pressure, gap)
        })
        .unzip()
}

fn recover_increment_strain(displacement: &[f64], element_count: usize) -> Vec<f64> {
    let mut padded = displacement.to_vec();
    padded.resize((element_count + 1) * VECTOR_COMPONENT_COUNT, 0.0);
    let mut strain = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
    for element_index in 0..element_count {
        let base = element_index * VECTOR_COMPONENT_COUNT;
        let next = (element_index + 1) * VECTOR_COMPONENT_COUNT;
        for component in 0..VECTOR_COMPONENT_COUNT {
            strain[element_index * TENSOR_COMPONENT_COUNT + component] =
                padded[next + component] - padded[base + component];
        }
    }
    strain
}

fn von_mises_from_tensor(tensor: &[f64]) -> f64 {
    let sxx = tensor[0] * 1.0e11;
    let syy = tensor[1] * 1.0e11;
    let szz = tensor[2] * 1.0e11;
    let txy = tensor[3] * 1.0e11;
    let tyz = tensor[4] * 1.0e11;
    let txz = tensor[5] * 1.0e11;
    (0.5 * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2))
        + 3.0 * (txy.powi(2) + tyz.powi(2) + txz.powi(2)))
    .sqrt()
}

fn contact_entity_count(contact: Option<&FeaContactInterfaceContext>) -> usize {
    if contact.is_some_and(|context| context.enabled) {
        1
    } else {
        0
    }
}
