use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        fea_modal_mode_shape_field_id, ComputeBackend, FeaModalRunResult, FeaRunError,
        FeaRunResult, ModalSolveOptions, FEA_FIELD_MODAL_EIGENVALUE, FEA_FIELD_MODAL_FREQUENCY_HZ,
        FEA_FIELD_MODAL_MODAL_MASS, FEA_FIELD_MODAL_MODAL_STIFFNESS,
        FEA_FIELD_MODAL_M_ORTHOGONALITY, FEA_FIELD_MODAL_PARTICIPATION_FACTOR,
        FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION, FEA_FIELD_MODAL_RESIDUAL_NORM,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    diagnostics::builders::{extend_common_run_diagnostics, CommonRunDiagnosticInputs},
    operator::{apply_k, apply_m},
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::modal::solve_modal_system,
};

pub fn run_modal(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaModalRunResult, FeaRunError> {
    run_modal_with_options(model, backend, ModalSolveOptions::default())
}

pub fn run_modal_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ModalSolveOptions,
) -> Result<FeaModalRunResult, FeaRunError> {
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating modal FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "modal model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;

    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling modal system",
        Some(1),
        Some(5),
    );
    let summary = assemble_linear_system(
        model,
        options.prep_context,
        options.thermo_mechanical_context,
        options.electro_thermal_context,
    );
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "modal system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;

    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving modal basis",
        Some(2),
        Some(5),
    );
    let modal = solve_modal_system(&summary, options.mode_count, backend);
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "modal solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;

    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering modal fields",
        Some(3),
        Some(5),
    );
    let mut diagnostics = modal.diagnostics.clone();
    extend_common_run_diagnostics(
        &mut diagnostics,
        CommonRunDiagnosticInputs {
            model,
            summary: &summary,
            prep_context: options.prep_context,
            iteration_metric: mode_shapes_iteration_proxy(&modal.residual_norms),
            residual_metric: modal.residual_norms.iter().copied().fold(0.0_f64, f64::max),
            requested_preconditioner: "auto",
            effective_preconditioner: if backend == ComputeBackend::Gpu {
                "jacobi"
            } else {
                "none"
            },
        },
    );

    let displacement = modal
        .mode_shapes
        .first()
        .cloned()
        .unwrap_or_else(|| vec![0.0; summary.dof_count.max(3)]);
    let von_mises = displacement
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        * 1.0e11;
    let modal_fields = recover_modal_result_fields(
        &summary,
        &modal.eigenvalues_hz,
        &modal.mode_shapes,
        &modal.residual_norms,
    );
    let mut run_fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            vec![displacement.len()],
            displacement,
        ),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![1], vec![von_mises]),
    ];
    run_fields.extend(modal_fields);

    let run = FeaRunResult {
        backend,
        solver_backend: modal.solver_backend,
        solver_device_apply_k_ratio: if modal.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            modal.device_apply_k_count as f64 / modal.device_apply_k_attempt_count as f64
        },
        solver_method: modal.solver_method,
        preconditioner: if backend == ComputeBackend::Gpu {
            "jacobi".to_string()
        } else {
            "none".to_string()
        },
        solver_host_sync_count: modal.solver_host_sync_count,
        diagnostics,
        fields: run_fields,
    };

    let mode_shapes = modal
        .mode_shapes
        .into_iter()
        .enumerate()
        .map(|(index, shape)| {
            AnalysisField::host_f64(
                fea_modal_mode_shape_field_id(index + 1),
                vec![shape.len()],
                shape,
            )
        })
        .collect();

    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "modal result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_modal")?;
    emit_phase(
        "fea.run_modal",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA modal run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaModalRunResult {
        run,
        eigenvalues_hz: modal.eigenvalues_hz,
        mode_shapes,
        residual_norms: modal.residual_norms,
    })
}

fn mode_shapes_iteration_proxy(residual_norms: &[f64]) -> f64 {
    residual_norms.len() as f64
}

fn recover_modal_result_fields(
    summary: &crate::assembly::AssemblySummary,
    frequencies_hz: &[f64],
    mode_shapes: &[Vec<f64>],
    residual_norms: &[f64],
) -> Vec<AnalysisField> {
    let mode_count = frequencies_hz.len().min(mode_shapes.len());
    let frequencies = frequencies_hz
        .iter()
        .copied()
        .take(mode_count)
        .collect::<Vec<_>>();
    let eigenvalues = frequencies
        .iter()
        .map(|frequency| (2.0 * std::f64::consts::PI * frequency).powi(2))
        .collect::<Vec<_>>();
    let mut modal_mass = Vec::with_capacity(mode_count);
    let mut modal_stiffness = Vec::with_capacity(mode_count);
    let mut participation_factor = Vec::with_capacity(mode_count);
    let residual_norm = (0..mode_count)
        .map(|index| residual_norms.get(index).copied().unwrap_or(f64::INFINITY))
        .collect::<Vec<_>>();
    let relative_frequency_separation = relative_frequency_separation(&frequencies);
    let mut m_orthogonality = Vec::with_capacity(mode_count.saturating_mul(mode_count));

    for mode_shape in mode_shapes.iter().take(mode_count) {
        let mass_applied = apply_m(&summary.operator, mode_shape);
        let stiffness_applied = apply_k(&summary.operator, mode_shape);
        let mass = dot(mode_shape, &mass_applied).abs();
        let stiffness = dot(mode_shape, &stiffness_applied).abs();
        modal_mass.push(mass);
        modal_stiffness.push(stiffness);
        participation_factor.push(mass_applied.iter().sum::<f64>() / mass.sqrt().max(1.0e-12));
    }

    for left in mode_shapes.iter().take(mode_count) {
        for right in mode_shapes.iter().take(mode_count) {
            let mass_applied = apply_m(&summary.operator, right);
            m_orthogonality.push(dot(left, &mass_applied));
        }
    }

    vec![
        AnalysisField::host_f64(FEA_FIELD_MODAL_FREQUENCY_HZ, vec![mode_count], frequencies),
        AnalysisField::host_f64(FEA_FIELD_MODAL_EIGENVALUE, vec![mode_count], eigenvalues),
        AnalysisField::host_f64(FEA_FIELD_MODAL_MODAL_MASS, vec![mode_count], modal_mass),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_MODAL_STIFFNESS,
            vec![mode_count],
            modal_stiffness,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_PARTICIPATION_FACTOR,
            vec![mode_count],
            participation_factor,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_RESIDUAL_NORM,
            vec![mode_count],
            residual_norm,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION,
            vec![mode_count],
            relative_frequency_separation,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_MODAL_M_ORTHOGONALITY,
            vec![mode_count, mode_count],
            m_orthogonality,
        ),
    ]
}

fn relative_frequency_separation(frequencies: &[f64]) -> Vec<f64> {
    if frequencies.len() < 2 {
        return vec![1.0; frequencies.len()];
    }

    let mut separation = vec![f64::INFINITY; frequencies.len()];
    for (index, window) in frequencies.windows(2).enumerate() {
        let a = window[0].abs().max(1.0e-12);
        let b = window[1].abs().max(1.0e-12);
        let value = (b - a).abs() / a.max(b);
        separation[index] = separation[index].min(value);
        separation[index + 1] = separation[index + 1].min(value);
    }
    separation
        .into_iter()
        .map(|value| if value.is_finite() { value } else { 1.0 })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}
