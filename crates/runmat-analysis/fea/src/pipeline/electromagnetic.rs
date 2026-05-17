use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, BoundaryConditionKind, EvidenceConfidence,
    LoadKind,
};

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
    let coefficient_profile = electromagnetic_coefficient_profile(model);
    let material_stats = coefficient_profile.stats;
    let total_bc_count = model.boundary_conditions.len().max(1);
    let mut magnetic_insulation_count = 0usize;
    let mut vector_potential_ground_count = 0usize;
    for bc in &model.boundary_conditions {
        match bc.kind {
            BoundaryConditionKind::MagneticInsulation => {
                magnetic_insulation_count += 1;
            }
            BoundaryConditionKind::VectorPotentialGround => {
                vector_potential_ground_count += 1;
            }
            _ => {}
        }
    }
    let electromagnetic_boundary_count = magnetic_insulation_count + vector_potential_ground_count;
    let boundary_anchor_ratio = electromagnetic_boundary_count as f64 / total_bc_count as f64;

    let total_load_count = model.loads.len().max(1);
    let mut electromagnetic_source_count = 0usize;
    let mut electromagnetic_source_strength = 0.0_f64;
    for load in &model.loads {
        match load.kind {
            LoadKind::CurrentDensity { jx, jy, jz } => {
                electromagnetic_source_count += 1;
                electromagnetic_source_strength +=
                    (jx * jx + jy * jy + jz * jz).sqrt().clamp(0.0, 2.5);
            }
            LoadKind::CoilCurrent { current_a } => {
                electromagnetic_source_count += 1;
                electromagnetic_source_strength +=
                    (current_a.abs() / domain.applied_current_a.max(1.0e-9)).clamp(0.0, 2.5);
            }
            _ => {}
        }
    }
    let source_realization_ratio = electromagnetic_source_count as f64 / total_load_count as f64;
    let source_drive_scale = if electromagnetic_source_count == 0 {
        0.25
    } else {
        (electromagnetic_source_strength / electromagnetic_source_count as f64).clamp(0.25, 2.5)
    };

    let mu0 = 4.0e-7 * std::f64::consts::PI;
    let epsilon0 = 8.854_187_812_8e-12_f64;
    let reluctivity = 1.0 / (mu0 * material_stats.relative_permeability_mean.max(1.0e-9));
    let omega = 2.0 * std::f64::consts::PI * domain.reference_frequency_hz;
    let effective_permittivity = epsilon0 * material_stats.relative_permittivity_mean.max(1.0e-9);
    let h = 1.0 / (node_count - 1) as f64;

    let region_count = coefficient_profile.region_coefficients.len().max(1);
    let mut node_sigma = vec![material_stats.conductivity_mean.max(1.0e-9); node_count];
    let mut node_eps_r = vec![material_stats.relative_permittivity_mean.max(1.0e-9); node_count];
    let mut node_mu_r = vec![material_stats.relative_permeability_mean.max(1.0e-9); node_count];
    for i in 0..node_count {
        let region_index = ((i * region_count) / node_count).min(region_count - 1);
        let region = coefficient_profile.region_coefficients[region_index];
        node_sigma[i] = region.conductivity_s_per_m;
        node_eps_r[i] = region.relative_permittivity;
        node_mu_r[i] = region.relative_permeability;
    }

    let mut stiffness_upper = vec![0.0_f64; node_count.saturating_sub(1)];
    for i in 0..stiffness_upper.len() {
        let mu_left = node_mu_r[i].max(1.0e-9);
        let mu_right = node_mu_r[i + 1].max(1.0e-9);
        let mu_edge = harmonic_mean(mu_left, mu_right).max(1.0e-9);
        let reluctivity_edge = 1.0 / (mu0 * mu_edge);
        stiffness_upper[i] = reluctivity_edge / (h * h);
    }
    let insulation_ratio = if electromagnetic_boundary_count == 0 {
        0.0
    } else {
        magnetic_insulation_count as f64 / electromagnetic_boundary_count as f64
    };
    if !stiffness_upper.is_empty() {
        let boundary_coupling_scale = (1.0 - 0.35 * insulation_ratio).clamp(0.35, 1.0);
        stiffness_upper[0] *= boundary_coupling_scale;
        let last = stiffness_upper.len() - 1;
        stiffness_upper[last] *= boundary_coupling_scale;
    }

    let mut stiffness_diag = vec![0.0_f64; node_count];
    let mut mass_terms = vec![0.0_f64; node_count];
    for i in 0..node_count {
        let left = if i > 0 { stiffness_upper[i - 1] } else { 0.0 };
        let right = if i + 1 < node_count {
            stiffness_upper[i]
        } else {
            0.0
        };
        let effective_permittivity_i = epsilon0 * node_eps_r[i].max(1.0e-9);
        let mass_i = (omega * node_sigma[i].max(1.0e-9) + omega * omega * effective_permittivity_i)
            .max(1.0e-9);
        mass_terms[i] = mass_i;
        stiffness_diag[i] = left + right + mass_i;
    }

    let mut constrained = vec![false; node_count];
    let mut anchor_count = if electromagnetic_boundary_count == 0 {
        2
    } else {
        ((node_count as f64) * (0.05 + 0.25 * boundary_anchor_ratio))
            .round()
            .clamp(2.0, (node_count as f64) * 0.6) as usize
    };
    if vector_potential_ground_count > 0 {
        anchor_count = anchor_count.max(2 + vector_potential_ground_count);
    }
    anchor_count = anchor_count.min(node_count);
    if anchor_count >= node_count {
        for value in &mut constrained {
            *value = true;
        }
    } else {
        let step =
            ((node_count - 1) as f64 / (anchor_count.saturating_sub(1).max(1)) as f64).max(1.0);
        for idx in 0..anchor_count {
            let dof = ((idx as f64) * step).round() as usize;
            constrained[dof.min(node_count - 1)] = true;
        }
        constrained[0] = true;
        constrained[node_count - 1] = true;
    }
    let mut rhs = vec![0.0_f64; node_count];
    for (i, rhs_i) in rhs.iter_mut().enumerate() {
        if constrained[i] {
            continue;
        }
        let x = i as f64 * h;
        let harmonic = (std::f64::consts::PI * x).sin().abs();
        let conductivity_scale = (node_sigma[i].max(1.0e-9)
            / material_stats.conductivity_mean.max(1.0e-9))
        .clamp(0.2, 5.0);
        let effective_permittivity_i = epsilon0 * node_eps_r[i].max(1.0e-9);
        *rhs_i = domain.applied_current_a * source_drive_scale * harmonic * conductivity_scale
            / (1.0 + omega * effective_permittivity_i);
    }

    summary.dof_count = node_count;
    summary.constrained_dof_count = constrained.iter().filter(|v| **v).count();
    summary.operator = OperatorSystem {
        dof_count: node_count,
        constrained: constrained.clone(),
        stiffness_diag,
        stiffness_upper,
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
    let unconstrained_diagonals = summary
        .operator
        .stiffness_diag
        .iter()
        .enumerate()
        .filter_map(|(i, value)| (!constrained[i]).then_some(*value))
        .collect::<Vec<_>>();
    let (min_diag, max_diag) = if unconstrained_diagonals.is_empty() {
        (1.0, 1.0)
    } else {
        (
            unconstrained_diagonals
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
                .max(1.0e-9),
            unconstrained_diagonals
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(1.0e-9),
        )
    };
    let solver_conditioning_proxy = (max_diag / min_diag).max(1.0);

    let mut diagnostics = solve.diagnostics;
    diagnostics.push(FeaDiagnostic {
        code: "FEA_EM_STATIC".to_string(),
        severity: if max_residual_norm <= options.residual_target {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "enabled={} reference_frequency_hz={} applied_current_a={} conductivity_mean_s_per_m={} relative_permittivity_mean={} relative_permeability_mean={} conductivity_spread_ratio={} relative_permittivity_spread_ratio={} relative_permeability_spread_ratio={} electromagnetic_material_heterogeneity_index={} assignment_coverage_ratio={} fallback_coefficient_ratio={} region_coefficient_contrast_index={} solver_conditioning_proxy={} source_realization_ratio={} boundary_anchor_ratio={} reluctivity={} effective_permittivity={} max_residual_norm={} solve_quality={} placeholder_quality={} energy_proxy={}",
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
            material_stats.assignment_coverage_ratio,
            material_stats.fallback_coefficient_ratio,
            material_stats.region_coefficient_contrast_index,
            solver_conditioning_proxy,
            source_realization_ratio,
            boundary_anchor_ratio,
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
    assignment_coverage_ratio: f64,
    fallback_coefficient_ratio: f64,
    region_coefficient_contrast_index: f64,
}

#[derive(Debug, Clone, Copy)]
struct RegionElectromagneticCoefficients {
    conductivity_s_per_m: f64,
    relative_permittivity: f64,
    relative_permeability: f64,
    weight: f64,
}

#[derive(Debug, Clone)]
struct ElectromagneticCoefficientProfile {
    region_coefficients: Vec<RegionElectromagneticCoefficients>,
    stats: ElectromagneticMaterialStats,
}

fn electromagnetic_coefficient_profile(model: &AnalysisModel) -> ElectromagneticCoefficientProfile {
    let material_by_id = model
        .materials
        .iter()
        .map(|material| (material.material_id.as_str(), material))
        .collect::<std::collections::HashMap<_, _>>();
    let mut samples = Vec::new();
    let mut covered_assignments = 0usize;
    let mut fallback_coefficients = 0usize;

    for assignment in &model.material_assignments {
        let assigned = material_by_id
            .get(assignment.assigned_material_id.as_str())
            .and_then(|material| material.electrical.as_ref());
        let expected = material_by_id
            .get(assignment.expected_material_id.as_str())
            .and_then(|material| material.electrical.as_ref());
        let (conductivity, permittivity, permeability, covered, used_fallback) =
            if let Some(electrical) = assigned {
                (
                    electrical.conductivity_s_per_m.max(1.0e-9),
                    electrical.relative_permittivity.max(1.0e-9),
                    electrical.relative_permeability.max(1.0e-9),
                    true,
                    false,
                )
            } else if let Some(electrical) = expected {
                (
                    electrical.conductivity_s_per_m.max(1.0e-9),
                    electrical.relative_permittivity.max(1.0e-9),
                    electrical.relative_permeability.max(1.0e-9),
                    true,
                    true,
                )
            } else {
                (1.0, 1.0, 1.0, false, true)
            };
        covered_assignments += usize::from(covered);
        fallback_coefficients += usize::from(used_fallback);
        samples.push(RegionElectromagneticCoefficients {
            conductivity_s_per_m: conductivity,
            relative_permittivity: permittivity,
            relative_permeability: permeability,
            weight: confidence_weight(assignment.confidence),
        });
    }

    if samples.is_empty() {
        for material in &model.materials {
            let Some(electrical) = material.electrical.as_ref() else {
                continue;
            };
            samples.push(RegionElectromagneticCoefficients {
                conductivity_s_per_m: electrical.conductivity_s_per_m.max(1.0e-9),
                relative_permittivity: electrical.relative_permittivity.max(1.0e-9),
                relative_permeability: electrical.relative_permeability.max(1.0e-9),
                weight: 1.0,
            });
        }
    }

    if samples.is_empty() {
        samples.push(RegionElectromagneticCoefficients {
            conductivity_s_per_m: 1.0,
            relative_permittivity: 1.0,
            relative_permeability: 1.0,
            weight: 1.0,
        });
    }

    let conductivity_values = samples
        .iter()
        .map(|sample| sample.conductivity_s_per_m)
        .collect::<Vec<_>>();
    let relative_permittivity_values = samples
        .iter()
        .map(|sample| sample.relative_permittivity)
        .collect::<Vec<_>>();
    let relative_permeability_values = samples
        .iter()
        .map(|sample| sample.relative_permeability)
        .collect::<Vec<_>>();

    let weighted_mean = |values: &[f64]| -> f64 {
        let weighted_sum = samples
            .iter()
            .zip(values.iter())
            .map(|(sample, value)| value * sample.weight)
            .sum::<f64>();
        let weight_sum = samples
            .iter()
            .map(|sample| sample.weight)
            .sum::<f64>()
            .max(1.0e-9);
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
            .map(|(sample, value)| sample.weight * (value - mean).powi(2))
            .sum::<f64>();
        let weight_sum = samples
            .iter()
            .map(|sample| sample.weight)
            .sum::<f64>()
            .max(1.0e-9);
        let std_dev = (variance_weighted_sum / weight_sum).sqrt();
        (std_dev / mean).max(0.0)
    };
    let assignment_heterogeneity_index = (weighted_cv(&conductivity_values, conductivity_mean)
        + weighted_cv(&relative_permittivity_values, relative_permittivity_mean)
        + weighted_cv(&relative_permeability_values, relative_permeability_mean))
        / 3.0;
    let assignment_coverage_ratio = if model.material_assignments.is_empty() {
        1.0
    } else {
        covered_assignments as f64 / model.material_assignments.len() as f64
    };
    let fallback_coefficient_ratio = if model.material_assignments.is_empty() {
        0.0
    } else {
        fallback_coefficients as f64 / model.material_assignments.len() as f64
    };
    let region_coefficient_contrast_index = ((conductivity_spread_ratio.max(1.0)).ln()
        + (relative_permittivity_spread_ratio.max(1.0)).ln()
        + (relative_permeability_spread_ratio.max(1.0)).ln())
        / 3.0;
    let region_coefficient_contrast_index = region_coefficient_contrast_index.max(0.0);

    ElectromagneticCoefficientProfile {
        region_coefficients: samples,
        stats: ElectromagneticMaterialStats {
            conductivity_mean,
            relative_permittivity_mean,
            relative_permeability_mean,
            conductivity_spread_ratio,
            relative_permittivity_spread_ratio,
            relative_permeability_spread_ratio,
            assignment_heterogeneity_index,
            assignment_coverage_ratio,
            fallback_coefficient_ratio,
            region_coefficient_contrast_index,
        },
    }
}

fn confidence_weight(confidence: EvidenceConfidence) -> f64 {
    match confidence {
        EvidenceConfidence::Verified => 1.0,
        EvidenceConfidence::Probable => 0.75,
        EvidenceConfidence::Inferred => 0.5,
    }
}

fn harmonic_mean(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if sum <= 1.0e-12 {
        0.0
    } else {
        2.0 * a * b / sum
    }
}
