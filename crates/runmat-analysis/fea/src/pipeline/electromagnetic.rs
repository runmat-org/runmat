use runmat_analysis_core::{validate_model, AnalysisField, AnalysisModel, EvidenceConfidence};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        ComputeBackend, ElectromagneticSolveOptions, FeaElectromagneticRunResult, FeaRunError,
        FeaRunResult,
    },
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::OperatorSystem,
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::solve_linear_system,
        preconditioner::SpdPreconditionerKind,
    },
};

pub fn run_electromagnetic(
    model: &AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaElectromagneticRunResult, FeaRunError> {
    run_electromagnetic_with_options(model, backend, ElectromagneticSolveOptions::default())
}

pub fn run_electromagnetic_with_options(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: ElectromagneticSolveOptions,
) -> Result<FeaElectromagneticRunResult, FeaRunError> {
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;

    let Some(domain) = model.electromagnetic.as_ref() else {
        return Err(FeaRunError::InvalidModel(
            "electromagnetic solve requires model.electromagnetic".to_string(),
        ));
    };
    if !domain.enabled {
        return Err(FeaRunError::InvalidModel(
            "electromagnetic solve requires enabled model.electromagnetic".to_string(),
        ));
    }
    if !domain.reference_frequency_hz.is_finite() || domain.reference_frequency_hz <= 0.0 {
        return Err(FeaRunError::InvalidModel(
            "electromagnetic solve requires finite positive reference_frequency_hz".to_string(),
        ));
    }
    if !domain.applied_current_a.is_finite() || domain.applied_current_a <= 0.0 {
        return Err(FeaRunError::InvalidModel(
            "electromagnetic solve requires finite positive applied_current_a".to_string(),
        ));
    }

    let mut summary = assemble_linear_system(model, options.prep_context, None, None);
    let node_count = summary.dof_count.max(8);
    let material_stats = electromagnetic_material_stats(model);

    let mu0 = 4.0e-7 * std::f64::consts::PI;
    let epsilon0 = 8.854_187_812_8e-12_f64;
    let reluctivity = 1.0 / (mu0 * material_stats.relative_permeability_mean.max(1.0e-9));
    let omega = 2.0 * std::f64::consts::PI * domain.reference_frequency_hz;
    let effective_permittivity = epsilon0 * material_stats.relative_permittivity_mean.max(1.0e-9);
    let h = 1.0 / (node_count - 1) as f64;
    let coupling = reluctivity / (h * h);
    let mass_term = (omega * material_stats.conductivity_mean + omega * omega * effective_permittivity)
        .max(1.0e-9);

    let mut constrained = vec![false; node_count];
    constrained[0] = true;
    constrained[node_count - 1] = true;
    let mut rhs = vec![0.0_f64; node_count];
    for (i, rhs_i) in rhs.iter_mut().enumerate() {
        if constrained[i] {
            continue;
        }
        let x = i as f64 * h;
        let harmonic = (std::f64::consts::PI * x).sin().abs();
        *rhs_i = domain.applied_current_a * harmonic / (1.0 + omega * effective_permittivity);
    }

    summary.dof_count = node_count;
    summary.constrained_dof_count = constrained.iter().filter(|v| **v).count();
    summary.operator = OperatorSystem {
        dof_count: node_count,
        constrained: constrained.clone(),
        stiffness_diag: vec![2.0 * coupling + mass_term; node_count],
        stiffness_upper: vec![coupling; node_count - 1],
        mass_diag: vec![1.0; node_count],
        damping_diag: vec![0.0; node_count],
        rhs,
    };

    let backend_kind = if backend == ComputeBackend::Gpu {
        LinearAlgebraBackendKind::RuntimeTensor
    } else {
        LinearAlgebraBackendKind::CpuReference
    };
    let cpu_backend = CpuReferenceBackend;
    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        backend_kind,
        &cpu_backend,
    );

    let vector_potential = solve.solution;
    let mut flux_density = vec![0.0_f64; node_count];
    for i in 1..(node_count - 1) {
        flux_density[i] = ((vector_potential[i + 1] - vector_potential[i - 1]) / (2.0 * h)).abs();
    }
    if node_count > 1 {
        flux_density[0] = flux_density[1];
        flux_density[node_count - 1] = flux_density[node_count - 2];
    }

    let max_residual_norm = solve.residual_norm;
    let solve_quality = if options.residual_target <= 0.0 || !max_residual_norm.is_finite() {
        0.0
    } else {
        (options.residual_target / (max_residual_norm + options.residual_target)).clamp(0.0, 1.0)
    };
    let energy_proxy = flux_density.iter().map(|v| v * v).sum::<f64>();

    let mut diagnostics = solve.diagnostics;
    diagnostics.push(FeaDiagnostic {
        code: "FEA_EM_STATIC".to_string(),
        severity: if max_residual_norm <= options.residual_target {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "enabled={} reference_frequency_hz={} applied_current_a={} conductivity_mean_s_per_m={} relative_permittivity_mean={} relative_permeability_mean={} conductivity_spread_ratio={} relative_permittivity_spread_ratio={} relative_permeability_spread_ratio={} electromagnetic_material_heterogeneity_index={} reluctivity={} effective_permittivity={} max_residual_norm={} solve_quality={} placeholder_quality={} energy_proxy={}",
            domain.enabled,
            domain.reference_frequency_hz,
            domain.applied_current_a,
            material_stats.conductivity_mean,
            material_stats.relative_permittivity_mean,
            material_stats.relative_permeability_mean,
            material_stats.conductivity_spread_ratio,
            material_stats.relative_permittivity_spread_ratio,
            material_stats.relative_permeability_spread_ratio,
            material_stats.assignment_heterogeneity_index,
            reluctivity,
            effective_permittivity,
            max_residual_norm,
            solve_quality,
            solve_quality,
            energy_proxy,
        ),
    });

    let run = FeaRunResult {
        backend,
        solver_backend: solve.solver_backend,
        solver_device_apply_k_ratio: if solve.device_apply_k_attempt_count == 0 {
            0.0
        } else {
            solve.device_apply_k_count as f64 / solve.device_apply_k_attempt_count as f64
        },
        solver_method: solve.solver_method,
        preconditioner: solve.preconditioner,
        solver_host_sync_count: solve.host_sync_count,
        diagnostics,
        displacement_field: AnalysisField::host_f64(
            "field_em_vector_potential",
            vec![node_count],
            vector_potential.clone(),
        ),
        von_mises_field: AnalysisField::host_f64(
            "field_em_flux_density",
            vec![node_count],
            flux_density.clone(),
        ),
    };

    Ok(FeaElectromagneticRunResult {
        run,
        reference_frequency_hz: domain.reference_frequency_hz,
        applied_current_a: domain.applied_current_a,
        vector_potential_field: AnalysisField::host_f64(
            "field_em_vector_potential",
            vec![node_count],
            vector_potential,
        ),
        flux_density_field: AnalysisField::host_f64(
            "field_em_flux_density",
            vec![node_count],
            flux_density,
        ),
        max_residual_norm,
        solve_quality,
    })
}

#[derive(Debug, Clone, Copy)]
struct ElectromagneticMaterialStats {
    conductivity_mean: f64,
    relative_permittivity_mean: f64,
    relative_permeability_mean: f64,
    conductivity_spread_ratio: f64,
    relative_permittivity_spread_ratio: f64,
    relative_permeability_spread_ratio: f64,
    assignment_heterogeneity_index: f64,
}

fn electromagnetic_material_stats(model: &AnalysisModel) -> ElectromagneticMaterialStats {
    let material_by_id = model
        .materials
        .iter()
        .map(|material| (material.material_id.as_str(), material))
        .collect::<std::collections::HashMap<_, _>>();
    let mut samples = Vec::new();
    for assignment in &model.material_assignments {
        let Some(material) = material_by_id
            .get(assignment.assigned_material_id.as_str())
            .or_else(|| material_by_id.get(assignment.expected_material_id.as_str()))
        else {
            continue;
        };
        let Some(electrical) = material.electrical.as_ref() else {
            continue;
        };
        samples.push((
            electrical.conductivity_s_per_m.max(1.0e-9),
            electrical.relative_permittivity.max(1.0e-9),
            electrical.relative_permeability.max(1.0e-9),
            confidence_weight(assignment.confidence),
        ));
    }

    if samples.is_empty() {
        for material in &model.materials {
            let Some(electrical) = material.electrical.as_ref() else {
                continue;
            };
            samples.push((
                electrical.conductivity_s_per_m.max(1.0e-9),
                electrical.relative_permittivity.max(1.0e-9),
                electrical.relative_permeability.max(1.0e-9),
                1.0,
            ));
        }
    }

    if samples.is_empty() {
        return ElectromagneticMaterialStats {
            conductivity_mean: 1.0,
            relative_permittivity_mean: 1.0,
            relative_permeability_mean: 1.0,
            conductivity_spread_ratio: 1.0,
            relative_permittivity_spread_ratio: 1.0,
            relative_permeability_spread_ratio: 1.0,
            assignment_heterogeneity_index: 0.0,
        };
    }

    let conductivity_values = samples.iter().map(|(v, _, _, _)| *v).collect::<Vec<_>>();
    let relative_permittivity_values = samples.iter().map(|(_, v, _, _)| *v).collect::<Vec<_>>();
    let relative_permeability_values = samples.iter().map(|(_, _, v, _)| *v).collect::<Vec<_>>();

    let weighted_mean = |values: &[f64]| -> f64 {
        let weighted_sum = samples
            .iter()
            .zip(values.iter())
            .map(|((_, _, _, w), value)| value * w)
            .sum::<f64>();
        let weight_sum = samples.iter().map(|(_, _, _, w)| *w).sum::<f64>().max(1.0e-9);
        weighted_sum / weight_sum
    };
    let spread_ratio = |values: &[f64]| -> f64 {
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if min <= 1.0e-12 {
            1.0
        } else {
            (max / min).max(1.0)
        }
    };

    let conductivity_mean = weighted_mean(&conductivity_values);
    let relative_permittivity_mean = weighted_mean(&relative_permittivity_values);
    let relative_permeability_mean = weighted_mean(&relative_permeability_values);
    let conductivity_spread_ratio = spread_ratio(&conductivity_values);
    let relative_permittivity_spread_ratio = spread_ratio(&relative_permittivity_values);
    let relative_permeability_spread_ratio = spread_ratio(&relative_permeability_values);

    let weighted_cv = |values: &[f64], mean: f64| -> f64 {
        if mean <= 1.0e-12 {
            return 0.0;
        }
        let variance_weighted_sum = samples
            .iter()
            .zip(values.iter())
            .map(|((_, _, _, w), value)| w * (value - mean).powi(2))
            .sum::<f64>();
        let weight_sum = samples.iter().map(|(_, _, _, w)| *w).sum::<f64>().max(1.0e-9);
        let std_dev = (variance_weighted_sum / weight_sum).sqrt();
        (std_dev / mean).max(0.0)
    };
    let assignment_heterogeneity_index = (weighted_cv(&conductivity_values, conductivity_mean)
        + weighted_cv(&relative_permittivity_values, relative_permittivity_mean)
        + weighted_cv(&relative_permeability_values, relative_permeability_mean))
        / 3.0;

    ElectromagneticMaterialStats {
        conductivity_mean,
        relative_permittivity_mean,
        relative_permeability_mean,
        conductivity_spread_ratio,
        relative_permittivity_spread_ratio,
        relative_permeability_spread_ratio,
        assignment_heterogeneity_index,
    }
}

fn confidence_weight(confidence: EvidenceConfidence) -> f64 {
    match confidence {
        EvidenceConfidence::Verified => 1.0,
        EvidenceConfidence::Probable => 0.75,
        EvidenceConfidence::Inferred => 0.5,
    }
}
