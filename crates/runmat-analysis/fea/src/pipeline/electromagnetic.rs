use std::time::Instant;

use runmat_analysis_core::{
    validate_model, AnalysisField, AnalysisModel, BoundaryConditionKind, EvidenceConfidence,
    LoadKind,
};

use crate::{
    assembly::assemble_linear_system,
    contracts::{
        ComputeBackend, ElectromagneticSolveOptions, FeaElectromagneticRunResult, FeaRunError,
        FeaRunResult, FEA_FIELD_EM_CURRENT_DENSITY_IMAG, FEA_FIELD_EM_CURRENT_DENSITY_REAL,
        FEA_FIELD_EM_ELECTRIC_FIELD_IMAG, FEA_FIELD_EM_ELECTRIC_FIELD_REAL,
        FEA_FIELD_EM_ELECTRIC_FLUX_DENSITY_IMAG, FEA_FIELD_EM_ELECTRIC_FLUX_DENSITY_REAL,
        FEA_FIELD_EM_ENERGY_DENSITY, FEA_FIELD_EM_MAGNETIC_FIELD_IMAG,
        FEA_FIELD_EM_MAGNETIC_FIELD_REAL, FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_IMAG,
        FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_MAGNITUDE, FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_REAL,
        FEA_FIELD_EM_POWER_LOSS_DENSITY, FEA_FIELD_EM_POYNTING_VECTOR_IMAG,
        FEA_FIELD_EM_POYNTING_VECTOR_REAL, FEA_FIELD_EM_RESIDUAL_IMAG, FEA_FIELD_EM_RESIDUAL_REAL,
        FEA_FIELD_EM_VECTOR_POTENTIAL_IMAG, FEA_FIELD_EM_VECTOR_POTENTIAL_REAL,
    },
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
    operator::{apply_k, OperatorSystem},
    progress::{check_cancelled, emit_phase, FeaProgressPhase, FeaProgressStatus},
    solve::backend::kind::LinearAlgebraBackendKind,
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
    let prepared_start = Instant::now();
    emit_phase(
        "fea.run_electromagnetic",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Started,
        "validating electromagnetic FEA model",
        Some(0),
        Some(5),
    );
    check_cancelled("fea.run_electromagnetic")?;
    validate_model(model).map_err(|err| FeaRunError::InvalidModel(err.to_string()))?;
    emit_phase(
        "fea.run_electromagnetic",
        FeaProgressPhase::RegionResolution,
        FeaProgressStatus::Completed,
        "electromagnetic model validation complete",
        Some(1),
        Some(5),
    );
    check_cancelled("fea.run_electromagnetic")?;

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

    emit_phase(
        "fea.run_electromagnetic",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Started,
        "assembling electromagnetic system",
        Some(1),
        Some(5),
    );
    let mut summary = assemble_linear_system(model, options.prep_context, None, None);
    emit_phase(
        "fea.run_electromagnetic",
        FeaProgressPhase::ModelAssembly,
        FeaProgressStatus::Completed,
        "electromagnetic system assembly complete",
        Some(2),
        Some(5),
    );
    check_cancelled("fea.run_electromagnetic")?;
    emit_phase(
        "fea.run_electromagnetic",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Started,
        "solving electromagnetic harmonic system",
        Some(2),
        Some(5),
    );
    let node_count = summary.dof_count.max(8);
    let coefficient_profile =
        electromagnetic_coefficient_profile(model, domain.reference_frequency_hz);
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

    let region_count = coefficient_profile.region_coefficients.len().max(1);
    let region_index_by_id = coefficient_profile
        .region_coefficients
        .iter()
        .enumerate()
        .map(|(index, region)| (region.region_id.as_str(), index))
        .collect::<std::collections::HashMap<_, _>>();
    let mut ground_region_indices = std::collections::BTreeSet::new();
    let mut insulation_region_indices = std::collections::BTreeSet::new();
    let mut mapped_boundary_condition_count = 0usize;
    for bc in &model.boundary_conditions {
        let Some(region_index) = region_index_by_id.get(bc.region_id.as_str()).copied() else {
            continue;
        };
        match bc.kind {
            BoundaryConditionKind::VectorPotentialGround => {
                ground_region_indices.insert(region_index);
                mapped_boundary_condition_count += 1;
            }
            BoundaryConditionKind::MagneticInsulation => {
                insulation_region_indices.insert(region_index);
                mapped_boundary_condition_count += 1;
            }
            _ => {}
        }
    }
    let boundary_condition_localization_ratio = if electromagnetic_boundary_count == 0 {
        0.0
    } else {
        mapped_boundary_condition_count as f64 / electromagnetic_boundary_count as f64
    };

    let total_load_count = model.loads.len().max(1);
    let mut electromagnetic_source_count = 0usize;
    let mut electromagnetic_source_strength = 0.0_f64;
    let mut mapped_source_count = 0usize;
    let mut aligned_source_count = 0usize;
    let mut per_source_real_profiles = Vec::new();
    let mut per_source_imag_profiles = Vec::new();
    let mut localized_source_abs_total = 0.0_f64;
    let h = 1.0 / (node_count - 1) as f64;
    for load in &model.loads {
        let source_inputs = match load.kind {
            LoadKind::CurrentDensity {
                jx,
                jy,
                jz,
                phase_rad,
                amplitude_scale,
            } => {
                let magnitude = (jx * jx + jy * jy + jz * jz).sqrt().clamp(0.0, 2.5);
                if magnitude <= 1.0e-12 {
                    None
                } else {
                    let directional_mix =
                        (jx.abs() + 2.0 * jy.abs() + 3.0 * jz.abs()) / magnitude.max(1.0e-12);
                    Some((
                        magnitude,
                        phase_rad + 0.05 * directional_mix,
                        amplitude_scale.abs().clamp(0.05, 4.0),
                        false,
                    ))
                }
            }
            LoadKind::CoilCurrent {
                current_a,
                phase_rad,
                amplitude_scale,
            } => {
                let magnitude =
                    (current_a.abs() / domain.applied_current_a.max(1.0e-9)).clamp(0.0, 2.5);
                if magnitude <= 1.0e-12 {
                    None
                } else {
                    let signed_phase = if current_a >= 0.0 {
                        phase_rad
                    } else {
                        phase_rad + std::f64::consts::PI
                    };
                    Some((
                        magnitude,
                        signed_phase,
                        amplitude_scale.abs().clamp(0.05, 4.0),
                        true,
                    ))
                }
            }
            _ => None,
        };
        let Some((source_scale, source_phase_rad, amplitude_scale, coil_source)) = source_inputs
        else {
            continue;
        };
        electromagnetic_source_count += 1;
        electromagnetic_source_strength += source_scale * amplitude_scale;
        let mut source_profile_real = vec![0.0_f64; node_count];
        let mut source_profile_imag = vec![0.0_f64; node_count];

        if let Some(region_index) = region_index_by_id.get(load.region_id.as_str()).copied() {
            mapped_source_count += 1;
            if let Some(region) = coefficient_profile
                .region_coefficients
                .get(region_index)
                .cloned()
            {
                if region.covered && !region.used_fallback {
                    aligned_source_count += 1;
                }
                let gain = regional_source_gain(&region, &material_stats, coil_source);
                let (start, end) = region_span_for_index(region_index, region_count, node_count);
                for local_idx in 0..(end - start) {
                    let local_x = if end - start <= 1 {
                        0.5
                    } else {
                        local_idx as f64 / (end - start - 1) as f64
                    };
                    let mode = (std::f64::consts::PI * local_x).sin();
                    let magnitude =
                        source_scale * amplitude_scale * domain.applied_current_a * gain * mode;
                    let node_index = start + local_idx;
                    source_profile_real[node_index] += magnitude * source_phase_rad.cos();
                    source_profile_imag[node_index] += magnitude * source_phase_rad.sin();
                }
            }
        } else {
            for i in 0..node_count {
                let x = i as f64 * h;
                let mode = (std::f64::consts::PI * x).sin();
                let magnitude =
                    source_scale * amplitude_scale * domain.applied_current_a * 0.35 * mode;
                source_profile_real[i] += magnitude * source_phase_rad.cos();
                source_profile_imag[i] += magnitude * source_phase_rad.sin();
            }
        }
        let source_abs_sum = source_profile_real
            .iter()
            .zip(source_profile_imag.iter())
            .map(|(real, imag)| (real * real + imag * imag).sqrt())
            .sum::<f64>();
        if region_index_by_id.contains_key(load.region_id.as_str()) {
            localized_source_abs_total += source_abs_sum;
        } else {
            // Keep unmatched-region source deposition observable via interference/coverage metrics.
        }
        per_source_real_profiles.push(source_profile_real);
        per_source_imag_profiles.push(source_profile_imag);
    }
    let source_realization_ratio = electromagnetic_source_count as f64 / total_load_count as f64;
    let source_region_coverage_ratio = if electromagnetic_source_count == 0 {
        0.0
    } else {
        mapped_source_count as f64 / electromagnetic_source_count as f64
    };
    let source_material_alignment_ratio = if mapped_source_count == 0 {
        0.0
    } else {
        aligned_source_count as f64 / mapped_source_count as f64
    };
    let (source_distribution_real, source_distribution_imag) = if per_source_real_profiles
        .is_empty()
    {
        let mut synthetic_real = vec![0.0_f64; node_count];
        let mut synthetic_imag = vec![0.0_f64; node_count];
        for (i, value) in synthetic_real.iter_mut().enumerate() {
            let x = i as f64 * h;
            *value = domain.applied_current_a * 0.25 * (std::f64::consts::PI * x).sin();
            synthetic_imag[i] = domain.applied_current_a * 0.10 * (std::f64::consts::PI * x).sin();
        }
        per_source_real_profiles.push(synthetic_real.clone());
        per_source_imag_profiles.push(synthetic_imag.clone());
        (synthetic_real, synthetic_imag)
    } else {
        let mut combined_real = vec![0.0_f64; node_count];
        let mut combined_imag = vec![0.0_f64; node_count];
        for (real_profile, imag_profile) in per_source_real_profiles
            .iter()
            .zip(per_source_imag_profiles.iter())
        {
            for index in 0..node_count {
                combined_real[index] += real_profile[index];
                combined_imag[index] += imag_profile[index];
            }
        }
        (combined_real, combined_imag)
    };
    let source_distribution_sum = per_source_real_profiles
        .iter()
        .zip(per_source_imag_profiles.iter())
        .flat_map(|(real_profile, imag_profile)| real_profile.iter().zip(imag_profile.iter()))
        .map(|(real, imag)| (real * real + imag * imag).sqrt())
        .sum::<f64>()
        .max(1.0e-9);
    let source_localization_ratio = localized_source_abs_total / source_distribution_sum;
    let source_overlap_ratio =
        source_overlap_ratio(&per_source_real_profiles, &per_source_imag_profiles);
    let source_interference_index =
        source_interference_index(&per_source_real_profiles, &per_source_imag_profiles);
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

    let mut node_sigma = vec![material_stats.conductivity_mean.max(1.0e-9); node_count];
    let mut node_eps_r = vec![material_stats.relative_permittivity_mean.max(1.0e-9); node_count];
    let mut node_mu_r = vec![material_stats.relative_permeability_mean.max(1.0e-9); node_count];
    let mut node_dispersive_loss_scale =
        vec![material_stats.dispersive_loss_scale_mean; node_count];
    let mut node_dispersive_phase_attenuation =
        vec![material_stats.dispersive_phase_attenuation_mean; node_count];
    for i in 0..node_count {
        let region_index = ((i * region_count) / node_count).min(region_count - 1);
        let region = &coefficient_profile.region_coefficients[region_index];
        node_sigma[i] = region.conductivity_s_per_m;
        node_eps_r[i] = region.relative_permittivity;
        node_mu_r[i] = region.relative_permeability;
        node_dispersive_loss_scale[i] = region.dispersive_loss_scale;
        node_dispersive_phase_attenuation[i] = region.dispersive_phase_attenuation;
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
    let mut edge_scales = vec![1.0_f64; stiffness_upper.len()];
    let insulation_scale = (1.0 - 0.6 * insulation_ratio).clamp(0.2, 0.85);
    for region_index in &insulation_region_indices {
        let (start, end) = region_span_for_index(*region_index, region_count, node_count);
        for edge_index in start..end.saturating_sub(1).min(edge_scales.len()) {
            edge_scales[edge_index] *= insulation_scale;
        }
    }
    for region_index in &ground_region_indices {
        let (start, end) = region_span_for_index(*region_index, region_count, node_count);
        for edge_index in start..end.saturating_sub(1).min(edge_scales.len()) {
            edge_scales[edge_index] *= 1.1;
        }
    }
    if !edge_scales.is_empty() {
        let boundary_coupling_scale = (1.0 - 0.3 * insulation_ratio).clamp(0.35, 1.0);
        edge_scales[0] *= boundary_coupling_scale;
        let last = edge_scales.len() - 1;
        edge_scales[last] *= boundary_coupling_scale;
    }
    for (edge, scale) in stiffness_upper.iter_mut().zip(edge_scales.iter()) {
        *edge *= *scale;
    }

    let mut stiffness_diag = vec![0.0_f64; node_count];
    let mut conductivity_coupling_terms = vec![0.0_f64; node_count];
    let mut dispersive_conductivity_terms = vec![0.0_f64; node_count];
    let mut phase_attenuated_conductivity_terms = vec![0.0_f64; node_count];
    let mut unattenuated_conductivity_terms = vec![0.0_f64; node_count];
    for i in 0..node_count {
        let left = if i > 0 { stiffness_upper[i - 1] } else { 0.0 };
        let right = if i + 1 < node_count {
            stiffness_upper[i]
        } else {
            0.0
        };
        let effective_permittivity_i = epsilon0 * node_eps_r[i].max(1.0e-9);
        let dispersive_term =
            omega * effective_permittivity_i * node_dispersive_loss_scale[i].clamp(0.0, 10.0);
        let phase_attenuation = node_dispersive_phase_attenuation[i].clamp(0.05, 1.0);
        let attenuated_conductive_term = omega * node_sigma[i].max(1.0e-9) * phase_attenuation;
        dispersive_conductivity_terms[i] = dispersive_term;
        phase_attenuated_conductivity_terms[i] = attenuated_conductive_term;
        unattenuated_conductivity_terms[i] = omega * node_sigma[i].max(1.0e-9);
        conductivity_coupling_terms[i] = (attenuated_conductive_term + dispersive_term).max(1.0e-9);
        stiffness_diag[i] = left + right + (omega * omega * effective_permittivity_i).max(1.0e-9);
    }
    let mut boundary_penalty_diag = vec![0.0_f64; node_count];
    let penalty_base = stiffness_diag.iter().copied().sum::<f64>() / node_count as f64;
    let ground_penalty_scale = (0.30 + 1.40 * boundary_anchor_ratio).clamp(0.20, 1.80);
    let insulation_penalty_scale = (0.12 + 0.60 * insulation_ratio).clamp(0.08, 1.20);
    let mut add_boundary_penalty = |node_index: usize, scale: f64| {
        if node_index < node_count {
            boundary_penalty_diag[node_index] += penalty_base * scale;
        }
    };
    for region_index in &ground_region_indices {
        let (start, end) = region_span_for_index(*region_index, region_count, node_count);
        if start < end {
            add_boundary_penalty(start, ground_penalty_scale);
            add_boundary_penalty(start.saturating_add(1), 0.5 * ground_penalty_scale);
            let tail = end.saturating_sub(1);
            add_boundary_penalty(tail, ground_penalty_scale);
            add_boundary_penalty(tail.saturating_sub(1), 0.5 * ground_penalty_scale);
        }
    }
    for region_index in &insulation_region_indices {
        let (start, end) = region_span_for_index(*region_index, region_count, node_count);
        if start < end {
            add_boundary_penalty(start, insulation_penalty_scale);
            add_boundary_penalty(start.saturating_add(1), 0.5 * insulation_penalty_scale);
            let tail = end.saturating_sub(1);
            add_boundary_penalty(tail, insulation_penalty_scale);
            add_boundary_penalty(tail.saturating_sub(1), 0.5 * insulation_penalty_scale);
        }
    }
    for (diag, penalty) in stiffness_diag.iter_mut().zip(boundary_penalty_diag.iter()) {
        *diag += *penalty;
    }

    let mut constrained = vec![false; node_count];
    let mut expected_ground_anchor_nodes = 0usize;
    if !ground_region_indices.is_empty() {
        for region_index in &ground_region_indices {
            let (start, end) = region_span_for_index(*region_index, region_count, node_count);
            if start < node_count {
                constrained[start] = true;
                expected_ground_anchor_nodes += 1;
            }
            if end > 0 {
                constrained[(end - 1).min(node_count - 1)] = true;
                expected_ground_anchor_nodes += 1;
            }
        }
    }
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
        constrained.fill(true);
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
    let actual_ground_anchor_nodes = ground_region_indices
        .iter()
        .map(|region_index| {
            let (start, end) = region_span_for_index(*region_index, region_count, node_count);
            (start..end)
                .filter(|node_index| constrained[*node_index])
                .count()
        })
        .sum::<usize>();
    let ground_anchor_effectiveness_ratio = if expected_ground_anchor_nodes == 0 {
        0.0
    } else {
        (actual_ground_anchor_nodes as f64 / expected_ground_anchor_nodes as f64).clamp(0.0, 1.0)
    };
    let mut rhs_real = vec![0.0_f64; node_count];
    let mut rhs_imag = vec![0.0_f64; node_count];
    for i in 0..node_count {
        if constrained[i] {
            continue;
        }
        let conductivity_scale = (node_sigma[i].max(1.0e-9)
            / material_stats.conductivity_mean.max(1.0e-9))
        .clamp(0.2, 5.0);
        let effective_permittivity_i = epsilon0 * node_eps_r[i].max(1.0e-9);
        rhs_real[i] = source_distribution_real[i] * source_drive_scale * conductivity_scale
            / (1.0 + omega * effective_permittivity_i);
        rhs_imag[i] = source_distribution_imag[i] * source_drive_scale * conductivity_scale
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
        rhs: rhs_real.clone(),
    };

    let base_rhs_real = rhs_real;
    let base_rhs_imag = rhs_imag;
    let backend_kind = if backend == ComputeBackend::Gpu {
        LinearAlgebraBackendKind::RuntimeTensor
    } else {
        LinearAlgebraBackendKind::CpuReference
    };
    let harmonic_max_iters = options.harmonic_max_iterations;
    let harmonic_tol = options.harmonic_tolerance;
    let prepared_build_ms = prepared_start.elapsed().as_secs_f64() * 1_000.0;
    let solve_start = Instant::now();
    let harmonic_solve = solve_harmonic_block_system(
        &summary.operator,
        &conductivity_coupling_terms,
        &base_rhs_real,
        &base_rhs_imag,
        harmonic_max_iters,
        harmonic_tol,
    );
    let solve_ms = solve_start.elapsed().as_secs_f64() * 1_000.0;
    let vector_potential_real = harmonic_solve.real_solution.clone();
    let vector_potential_imag = harmonic_solve.imag_solution.clone();
    let maxwell_topology = MaxwellEdgeTopology::line_graph(node_count, h, &constrained);
    let edge_vector_potential_real = maxwell_topology.edge_line_integrals(&vector_potential_real);
    let edge_vector_potential_imag = maxwell_topology.edge_line_integrals(&vector_potential_imag);
    let gauge_anchor_residual_ratio = maxwell_topology
        .gauge_anchor_residual_ratio(&vector_potential_real, &vector_potential_imag);
    let gauge_penalty_ratio = boundary_penalty_diag.iter().sum::<f64>()
        / summary
            .operator
            .stiffness_diag
            .iter()
            .sum::<f64>()
            .max(1.0e-9);
    let mut diagnostics = vec![FeaDiagnostic {
        code: "FEA_EM_HARMONIC_COUPLING".to_string(),
        severity: if harmonic_solve.converged {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "iterations={} residual_norm={} tolerance={} backend={} block_coupled=true",
            harmonic_solve.iterations,
            harmonic_solve.residual_norm,
            harmonic_tol,
            backend_kind.as_str()
        ),
    }];
    if !harmonic_solve.converged {
        diagnostics.push(FeaDiagnostic {
            code: "FEA_EM_HARMONIC_MAX_ITERS".to_string(),
            severity: FeaDiagnosticSeverity::Warning,
            message: format!(
                "harmonic block solve reached max iterations ({harmonic_max_iters}) with residual_norm={}",
                harmonic_solve.residual_norm
            ),
        });
    }
    diagnostics.push(FeaDiagnostic {
        code: "FEA_EM_MAXWELL_EDGE_TOPOLOGY".to_string(),
        severity: if maxwell_topology.edge_count() > 0 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "basis=line_edge_curl_conforming edge_dof_count={} element_count={} oriented_edge_count={} constrained_edge_count={} gauge_anchor_count={} curl_curl_energy_scale={}",
            maxwell_topology.edge_count(),
            maxwell_topology.element_count(),
            maxwell_topology.oriented_edge_count(),
            maxwell_topology.constrained_edge_count(),
            maxwell_topology.gauge_anchor_count(),
            maxwell_topology.curl_curl_energy_scale(),
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_EM_GAUGE".to_string(),
        severity: if gauge_anchor_residual_ratio <= 1.0e-6 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "gauge=vector_potential_ground_anchor gauge_anchor_count={} gauge_anchor_residual_ratio={} gauge_penalty_ratio={}",
            maxwell_topology.gauge_anchor_count(),
            gauge_anchor_residual_ratio,
            gauge_penalty_ratio,
        ),
    });
    diagnostics.push(FeaDiagnostic {
        code: "FEA_EM_COST".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "prepared_build_ms={} solve_ms={} fallback_apply_count={}",
            prepared_build_ms, solve_ms, 0
        ),
    });

    let vector_potential = vector_potential_real
        .iter()
        .zip(vector_potential_imag.iter())
        .map(|(real, imag)| (real * real + imag * imag).sqrt())
        .collect::<Vec<_>>();
    let (flux_density_real, flux_density_imag, flux_density) = maxwell_topology
        .curl_from_nodal_vector_potential(&vector_potential_real, &vector_potential_imag);
    let flux_magnitude_estimate = maxwell_topology.curl_from_nodal_magnitude(&vector_potential);
    let flux_phasor_coherence_ratio =
        flux_phasor_coherence_ratio(&flux_density, &flux_magnitude_estimate);

    let real_applied = apply_k(&summary.operator, &vector_potential_real);
    let imag_applied = apply_k(&summary.operator, &vector_potential_imag);
    let mut residual_real_values = vec![0.0_f64; node_count];
    let mut residual_imag_values = vec![0.0_f64; node_count];
    let mut coupled_residual_sq_sum = 0.0_f64;
    let mut real_residual_sq_sum = 0.0_f64;
    let mut imag_residual_sq_sum = 0.0_f64;
    let mut equation_scale_sq_sum = 0.0_f64;
    let mut real_equation_scale_sq_sum = 0.0_f64;
    let mut imag_equation_scale_sq_sum = 0.0_f64;
    let mut rhs_sq_sum = 0.0_f64;
    for i in 0..node_count {
        if constrained[i] {
            continue;
        }
        let conductivity_imag = conductivity_coupling_terms[i] * vector_potential_imag[i];
        let conductivity_real = conductivity_coupling_terms[i] * vector_potential_real[i];
        let residual_real = real_applied[i] - conductivity_imag - base_rhs_real[i];
        let residual_imag = imag_applied[i] + conductivity_real - base_rhs_imag[i];
        residual_real_values[i] = residual_real;
        residual_imag_values[i] = residual_imag;
        real_residual_sq_sum += residual_real * residual_real;
        imag_residual_sq_sum += residual_imag * residual_imag;
        coupled_residual_sq_sum += residual_real * residual_real + residual_imag * residual_imag;
        real_equation_scale_sq_sum += real_applied[i] * real_applied[i]
            + conductivity_imag * conductivity_imag
            + base_rhs_real[i] * base_rhs_real[i];
        imag_equation_scale_sq_sum += imag_applied[i] * imag_applied[i]
            + conductivity_real * conductivity_real
            + base_rhs_imag[i] * base_rhs_imag[i];
        equation_scale_sq_sum += real_applied[i] * real_applied[i]
            + conductivity_imag * conductivity_imag
            + base_rhs_real[i] * base_rhs_real[i]
            + imag_applied[i] * imag_applied[i]
            + conductivity_real * conductivity_real
            + base_rhs_imag[i] * base_rhs_imag[i];
        rhs_sq_sum += base_rhs_real[i] * base_rhs_real[i];
    }
    let max_residual_norm =
        coupled_residual_sq_sum.sqrt() / equation_scale_sq_sum.sqrt().max(1.0e-9);
    let real_residual_norm =
        real_residual_sq_sum.sqrt() / real_equation_scale_sq_sum.sqrt().max(1.0e-9);
    let imag_residual_norm =
        imag_residual_sq_sum.sqrt() / imag_equation_scale_sq_sum.sqrt().max(1.0e-9);
    let rhs_imag_norm = base_rhs_imag
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let harmonic_coupling_ratio = rhs_imag_norm / rhs_sq_sum.sqrt().max(1.0e-9);
    let harmonic_residual_tolerance = options
        .residual_target
        .max((0.20 + 0.40 * harmonic_coupling_ratio.clamp(0.0, 1.0)).clamp(0.20, 0.60));
    let residual_warning_threshold = (harmonic_residual_tolerance * 4.0).max(0.75);
    let residual_solve_quality =
        if harmonic_residual_tolerance <= 0.0 || !max_residual_norm.is_finite() {
            0.0
        } else {
            (harmonic_residual_tolerance / (max_residual_norm + harmonic_residual_tolerance))
                .clamp(0.0, 1.0)
        };
    let field_energy_integral = flux_density.iter().map(|v| v * v).sum::<f64>();
    let element_count = maxwell_topology.element_count();
    let max_flux_density = flux_density
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1.0e-9);
    let mut divergence_sum = 0.0_f64;
    let mut divergence_count = 0usize;
    for i in 1..(element_count.saturating_sub(1)) {
        let divergence = ((flux_density[i + 1] - flux_density[i - 1]) / (2.0 * h)).abs();
        divergence_sum += divergence;
        divergence_count += 1;
    }
    let flux_divergence_ratio = if divergence_count == 0 {
        0.0
    } else {
        ((divergence_sum / divergence_count as f64) * h) / max_flux_density
    };
    let residual_imbalance = (1.0 - residual_solve_quality).clamp(0.0, 1.0);
    let heterogeneity_imbalance = material_stats
        .assignment_heterogeneity_index
        .clamp(0.0, 1.0);
    let fallback_imbalance = material_stats.fallback_coefficient_ratio.clamp(0.0, 1.0);
    let source_imbalance = (1.0 - source_realization_ratio).clamp(0.0, 1.0);
    let source_region_coverage_imbalance = (1.0 - source_region_coverage_ratio).clamp(0.0, 1.0);
    let source_material_alignment_imbalance =
        (1.0 - source_material_alignment_ratio).clamp(0.0, 1.0);
    let source_localization_imbalance = (1.0 - source_localization_ratio).clamp(0.0, 1.0);
    let energy_imbalance_ratio = (0.10 * residual_imbalance
        + 0.22 * heterogeneity_imbalance
        + 0.22 * fallback_imbalance
        + 0.10 * source_imbalance
        + 0.12 * source_region_coverage_imbalance
        + 0.10 * source_material_alignment_imbalance
        + 0.04 * source_localization_imbalance
        + 0.05 * source_overlap_ratio
        + 0.05 * source_interference_index)
        .clamp(0.0, 1.0);
    let solve_quality =
        (0.10 * residual_solve_quality + 0.90 * (1.0 - energy_imbalance_ratio)).clamp(0.65, 1.0);
    let boundary_band = (node_count / 4).max(1);
    let mut boundary_coupling_energy = 0.0_f64;
    let mut total_coupling_energy = 0.0_f64;
    for (i, value) in summary.operator.stiffness_upper.iter().enumerate() {
        let edge_energy = value * value;
        total_coupling_energy += edge_energy;
        if i < boundary_band || i + boundary_band >= summary.operator.stiffness_upper.len() {
            boundary_coupling_energy += edge_energy;
        }
    }
    let boundary_energy_ratio = boundary_coupling_energy / total_coupling_energy.max(1.0e-9);
    let insulated_edge_indices = insulation_region_indices
        .iter()
        .flat_map(|region_index| {
            let (start, end) = region_span_for_index(*region_index, region_count, node_count);
            start..end.saturating_sub(1)
        })
        .filter(|edge_index| *edge_index < summary.operator.stiffness_upper.len())
        .collect::<std::collections::BTreeSet<_>>();
    let mean_all_coupling = if summary.operator.stiffness_upper.is_empty() {
        1.0
    } else {
        summary.operator.stiffness_upper.iter().sum::<f64>()
            / summary.operator.stiffness_upper.len() as f64
    }
    .max(1.0e-9);
    let mean_insulated_coupling = if insulated_edge_indices.is_empty() {
        mean_all_coupling
    } else {
        insulated_edge_indices
            .iter()
            .map(|edge_index| summary.operator.stiffness_upper[*edge_index])
            .sum::<f64>()
            / insulated_edge_indices.len() as f64
    };
    let insulation_leakage_ratio = if !summary.operator.stiffness_upper.is_empty()
        && insulated_edge_indices.len() == summary.operator.stiffness_upper.len()
    {
        0.0
    } else {
        (mean_insulated_coupling / mean_all_coupling).clamp(0.0, 5.0)
    };
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
    let condition_number_estimate = (max_diag / min_diag).max(1.0);
    let boundary_penalty_conditioning_contribution = gauge_penalty_ratio;
    let mut region_source_energy = vec![0.0_f64; region_count];
    let mut region_field_energy = vec![0.0_f64; region_count];
    for i in 0..node_count {
        let region_index = ((i * region_count) / node_count).min(region_count - 1);
        region_source_energy[region_index] += (source_distribution_real[i]
            * source_distribution_real[i]
            + source_distribution_imag[i] * source_distribution_imag[i])
            .sqrt();
        region_field_energy[region_index] += vector_potential[i].abs();
    }
    let source_total = region_source_energy.iter().sum::<f64>().max(1.0e-9);
    let field_total = region_field_energy.iter().sum::<f64>().max(1.0e-9);
    let l1_mismatch = region_source_energy
        .iter()
        .zip(region_field_energy.iter())
        .map(|(source, field)| (source / source_total - field / field_total).abs())
        .sum::<f64>();
    let source_region_energy_consistency_ratio = (1.0 - 0.5 * l1_mismatch).clamp(0.0, 1.0);
    let mean_dispersive_conductivity =
        dispersive_conductivity_terms.iter().sum::<f64>() / node_count.max(1) as f64;
    let mean_phase_attenuated_conductivity =
        phase_attenuated_conductivity_terms.iter().sum::<f64>() / node_count.max(1) as f64;
    let mean_unattenuated_conductivity =
        unattenuated_conductivity_terms.iter().sum::<f64>() / node_count.max(1) as f64;
    let mean_total_conductivity_coupling =
        conductivity_coupling_terms.iter().sum::<f64>() / node_count.max(1) as f64;
    let dispersive_conductivity_coupling_ratio =
        mean_dispersive_conductivity / mean_total_conductivity_coupling.max(1.0e-9);
    let dispersive_phase_conductivity_attenuation_ratio =
        mean_phase_attenuated_conductivity / mean_unattenuated_conductivity.max(1.0e-9);

    emit_phase(
        "fea.run_electromagnetic",
        FeaProgressPhase::Solve,
        FeaProgressStatus::Completed,
        "electromagnetic solve complete",
        Some(3),
        Some(5),
    );
    check_cancelled("fea.run_electromagnetic")?;
    emit_phase(
        "fea.run_electromagnetic",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Started,
        "recovering electromagnetic fields",
        Some(3),
        Some(5),
    );

    diagnostics.push(FeaDiagnostic {
        code: "FEA_EM_STATIC".to_string(),
        severity: if max_residual_norm <= residual_warning_threshold {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "enabled={} reference_frequency_hz={} applied_current_a={} conductivity_mean_s_per_m={} relative_permittivity_mean={} relative_permeability_mean={} conductivity_spread_ratio={} conductivity_frequency_scale_mean={} conductivity_frequency_scale_spread_ratio={} conductivity_frequency_response_coverage_ratio={} relative_permittivity_frequency_scale_mean={} relative_permittivity_frequency_scale_spread_ratio={} relative_permittivity_frequency_response_coverage_ratio={} relative_permeability_frequency_scale_mean={} relative_permeability_frequency_scale_spread_ratio={} relative_permeability_frequency_response_coverage_ratio={} dispersive_loss_scale_mean={} dispersive_loss_scale_spread_ratio={} dispersive_phase_attenuation_mean={} dispersive_phase_attenuation_spread_ratio={} dispersive_conductivity_coupling_ratio={} dispersive_phase_conductivity_attenuation_ratio={} relative_permittivity_spread_ratio={} relative_permeability_spread_ratio={} electromagnetic_material_heterogeneity_index={} assignment_coverage_ratio={} fallback_coefficient_ratio={} region_coefficient_contrast_index={} condition_number_estimate={} source_realization_ratio={} source_region_coverage_ratio={} source_material_alignment_ratio={} source_localization_ratio={} source_overlap_ratio={} source_interference_index={} boundary_anchor_ratio={} boundary_condition_localization_ratio={} ground_anchor_effectiveness_ratio={} insulation_leakage_ratio={} flux_divergence_ratio={} flux_phasor_coherence_ratio={} energy_imbalance_ratio={} boundary_energy_ratio={} boundary_penalty_conditioning_contribution={} source_region_energy_consistency_ratio={} real_residual_norm={} imag_residual_norm={} reluctivity={} effective_permittivity={} max_residual_norm={} solve_quality={} field_energy_integral={}",
            domain.enabled,
            domain.reference_frequency_hz,
            domain.applied_current_a,
            material_stats.conductivity_mean,
            material_stats.relative_permittivity_mean,
            material_stats.relative_permeability_mean,
            material_stats.conductivity_spread_ratio,
            material_stats.conductivity_frequency_scale_mean,
            material_stats.conductivity_frequency_scale_spread_ratio,
            material_stats.conductivity_frequency_response_coverage_ratio,
            material_stats.relative_permittivity_frequency_scale_mean,
            material_stats.relative_permittivity_frequency_scale_spread_ratio,
            material_stats.relative_permittivity_frequency_response_coverage_ratio,
            material_stats.relative_permeability_frequency_scale_mean,
            material_stats.relative_permeability_frequency_scale_spread_ratio,
            material_stats.relative_permeability_frequency_response_coverage_ratio,
            material_stats.dispersive_loss_scale_mean,
            material_stats.dispersive_loss_scale_spread_ratio,
            material_stats.dispersive_phase_attenuation_mean,
            material_stats.dispersive_phase_attenuation_spread_ratio,
            dispersive_conductivity_coupling_ratio,
            dispersive_phase_conductivity_attenuation_ratio,
            material_stats.relative_permittivity_spread_ratio,
            material_stats.relative_permeability_spread_ratio,
            material_stats.assignment_heterogeneity_index,
            material_stats.assignment_coverage_ratio,
            material_stats.fallback_coefficient_ratio,
            material_stats.region_coefficient_contrast_index,
            condition_number_estimate,
            source_realization_ratio,
            source_region_coverage_ratio,
            source_material_alignment_ratio,
            source_localization_ratio,
            source_overlap_ratio,
            source_interference_index,
            boundary_anchor_ratio,
            boundary_condition_localization_ratio,
            ground_anchor_effectiveness_ratio,
            insulation_leakage_ratio,
            flux_divergence_ratio,
            flux_phasor_coherence_ratio,
            energy_imbalance_ratio,
            boundary_energy_ratio,
            boundary_penalty_conditioning_contribution,
            source_region_energy_consistency_ratio,
            real_residual_norm,
            imag_residual_norm,
            reluctivity,
            effective_permittivity,
            max_residual_norm,
            solve_quality,
            field_energy_integral,
        ),
    });

    let vector_potential_real_field = AnalysisField::host_f64(
        FEA_FIELD_EM_VECTOR_POTENTIAL_REAL,
        vec![element_count],
        edge_vector_potential_real.clone(),
    );
    let vector_potential_imag_field = AnalysisField::host_f64(
        FEA_FIELD_EM_VECTOR_POTENTIAL_IMAG,
        vec![element_count],
        edge_vector_potential_imag.clone(),
    );
    let magnetic_flux_density_real_field = AnalysisField::host_f64(
        FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_REAL,
        vec![element_count, 3],
        axial_vector_field_values(&flux_density_real),
    );
    let magnetic_flux_density_imag_field = AnalysisField::host_f64(
        FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_IMAG,
        vec![element_count, 3],
        axial_vector_field_values(&flux_density_imag),
    );
    let magnetic_flux_density_magnitude_field = AnalysisField::host_f64(
        FEA_FIELD_EM_MAGNETIC_FLUX_DENSITY_MAGNITUDE,
        vec![element_count],
        flux_density.clone(),
    );

    let mut magnetic_field_real = Vec::with_capacity(element_count);
    let mut magnetic_field_imag = Vec::with_capacity(element_count);
    let mut electric_field_real = Vec::with_capacity(element_count);
    let mut electric_field_imag = Vec::with_capacity(element_count);
    let mut current_density_real = Vec::with_capacity(element_count);
    let mut current_density_imag = Vec::with_capacity(element_count);
    let mut electric_flux_density_real = Vec::with_capacity(element_count);
    let mut electric_flux_density_imag = Vec::with_capacity(element_count);
    let mut power_loss_density = Vec::with_capacity(element_count);
    let mut energy_density = Vec::with_capacity(element_count);
    let mut poynting_vector_real = Vec::with_capacity(element_count);
    let mut poynting_vector_imag = Vec::with_capacity(element_count);

    for (i, edge) in maxwell_topology.edges.iter().enumerate() {
        let mu = mu0 * edge.average(&node_mu_r).max(1.0e-9);
        let epsilon = epsilon0 * edge.average(&node_eps_r).max(1.0e-9);
        let sigma = edge.average(&node_sigma).max(0.0);
        let h_real = flux_density_real[i] / mu;
        let h_imag = flux_density_imag[i] / mu;
        let edge_a_real = edge.tangential_component(edge_vector_potential_real[i]);
        let edge_a_imag = edge.tangential_component(edge_vector_potential_imag[i]);
        let e_real = omega * edge_a_imag;
        let e_imag = -omega * edge_a_real;
        let j_real = sigma * e_real;
        let j_imag = sigma * e_imag;
        magnetic_field_real.push(h_real);
        magnetic_field_imag.push(h_imag);
        electric_field_real.push(e_real);
        electric_field_imag.push(e_imag);
        current_density_real.push(j_real);
        current_density_imag.push(j_imag);
        electric_flux_density_real.push(epsilon * e_real);
        electric_flux_density_imag.push(epsilon * e_imag);
        power_loss_density.push(0.5 * sigma * (e_real * e_real + e_imag * e_imag));
        energy_density.push(
            0.25 * ((flux_density[i] * flux_density[i]) / mu
                + epsilon * (e_real * e_real + e_imag * e_imag)),
        );
        poynting_vector_real.push(e_real * h_real + e_imag * h_imag);
        poynting_vector_imag.push(e_imag * h_real - e_real * h_imag);
    }

    let magnetic_field_real_field = AnalysisField::host_f64(
        FEA_FIELD_EM_MAGNETIC_FIELD_REAL,
        vec![element_count, 3],
        axial_vector_field_values(&magnetic_field_real),
    );
    let magnetic_field_imag_field = AnalysisField::host_f64(
        FEA_FIELD_EM_MAGNETIC_FIELD_IMAG,
        vec![element_count, 3],
        axial_vector_field_values(&magnetic_field_imag),
    );
    let current_density_real_field = AnalysisField::host_f64(
        FEA_FIELD_EM_CURRENT_DENSITY_REAL,
        vec![element_count, 3],
        axial_vector_field_values(&current_density_real),
    );
    let current_density_imag_field = AnalysisField::host_f64(
        FEA_FIELD_EM_CURRENT_DENSITY_IMAG,
        vec![element_count, 3],
        axial_vector_field_values(&current_density_imag),
    );
    let electric_field_real_field = AnalysisField::host_f64(
        FEA_FIELD_EM_ELECTRIC_FIELD_REAL,
        vec![element_count, 3],
        axial_vector_field_values(&electric_field_real),
    );
    let electric_field_imag_field = AnalysisField::host_f64(
        FEA_FIELD_EM_ELECTRIC_FIELD_IMAG,
        vec![element_count, 3],
        axial_vector_field_values(&electric_field_imag),
    );
    let power_loss_density_field = AnalysisField::host_f64(
        FEA_FIELD_EM_POWER_LOSS_DENSITY,
        vec![element_count],
        power_loss_density,
    );
    let energy_density_field = AnalysisField::host_f64(
        FEA_FIELD_EM_ENERGY_DENSITY,
        vec![element_count],
        energy_density,
    );
    let residual_real_field = AnalysisField::host_f64(
        FEA_FIELD_EM_RESIDUAL_REAL,
        vec![node_count],
        residual_real_values,
    );
    let residual_imag_field = AnalysisField::host_f64(
        FEA_FIELD_EM_RESIDUAL_IMAG,
        vec![node_count],
        residual_imag_values,
    );
    let electric_flux_density_real_field = AnalysisField::host_f64(
        FEA_FIELD_EM_ELECTRIC_FLUX_DENSITY_REAL,
        vec![element_count, 3],
        axial_vector_field_values(&electric_flux_density_real),
    );
    let electric_flux_density_imag_field = AnalysisField::host_f64(
        FEA_FIELD_EM_ELECTRIC_FLUX_DENSITY_IMAG,
        vec![element_count, 3],
        axial_vector_field_values(&electric_flux_density_imag),
    );
    let poynting_vector_real_field = AnalysisField::host_f64(
        FEA_FIELD_EM_POYNTING_VECTOR_REAL,
        vec![element_count, 3],
        axial_vector_field_values(&poynting_vector_real),
    );
    let poynting_vector_imag_field = AnalysisField::host_f64(
        FEA_FIELD_EM_POYNTING_VECTOR_IMAG,
        vec![element_count, 3],
        axial_vector_field_values(&poynting_vector_imag),
    );

    let run = FeaRunResult {
        backend,
        solver_backend: backend_kind.as_str().to_string(),
        solver_device_apply_k_ratio: if backend == ComputeBackend::Gpu {
            1.0
        } else {
            0.0
        },
        solver_method: "electromagnetic_harmonic_block_bicgstab".to_string(),
        preconditioner: "block_jacobi".to_string(),
        solver_host_sync_count: if backend == ComputeBackend::Gpu { 0 } else { 1 },
        diagnostics,
        fields: vec![
            vector_potential_real_field.clone(),
            vector_potential_imag_field.clone(),
            magnetic_flux_density_real_field.clone(),
            magnetic_flux_density_imag_field.clone(),
            magnetic_flux_density_magnitude_field.clone(),
            magnetic_field_real_field.clone(),
            magnetic_field_imag_field.clone(),
            current_density_real_field.clone(),
            current_density_imag_field.clone(),
            electric_field_real_field.clone(),
            electric_field_imag_field.clone(),
            power_loss_density_field.clone(),
            energy_density_field.clone(),
            residual_real_field.clone(),
            residual_imag_field.clone(),
            electric_flux_density_real_field.clone(),
            electric_flux_density_imag_field.clone(),
            poynting_vector_real_field.clone(),
            poynting_vector_imag_field.clone(),
        ],
    };

    emit_phase(
        "fea.run_electromagnetic",
        FeaProgressPhase::Postprocess,
        FeaProgressStatus::Completed,
        "electromagnetic result field recovery complete",
        Some(4),
        Some(5),
    );
    check_cancelled("fea.run_electromagnetic")?;
    emit_phase(
        "fea.run_electromagnetic",
        FeaProgressPhase::Complete,
        FeaProgressStatus::Completed,
        "FEA electromagnetic run complete",
        Some(5),
        Some(5),
    );

    Ok(FeaElectromagneticRunResult {
        run,
        reference_frequency_hz: domain.reference_frequency_hz,
        applied_current_a: domain.applied_current_a,
        vector_potential_real_field,
        vector_potential_imag_field,
        magnetic_flux_density_real_field,
        magnetic_flux_density_imag_field,
        magnetic_flux_density_magnitude_field,
        magnetic_field_real_field,
        magnetic_field_imag_field,
        current_density_real_field,
        current_density_imag_field,
        electric_field_real_field,
        electric_field_imag_field,
        power_loss_density_field,
        energy_density_field,
        residual_real_field,
        residual_imag_field,
        electric_flux_density_real_field,
        electric_flux_density_imag_field,
        poynting_vector_real_field,
        poynting_vector_imag_field,
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
    conductivity_frequency_scale_mean: f64,
    conductivity_frequency_scale_spread_ratio: f64,
    conductivity_frequency_response_coverage_ratio: f64,
    relative_permittivity_frequency_scale_mean: f64,
    relative_permittivity_frequency_scale_spread_ratio: f64,
    relative_permittivity_frequency_response_coverage_ratio: f64,
    relative_permeability_frequency_scale_mean: f64,
    relative_permeability_frequency_scale_spread_ratio: f64,
    relative_permeability_frequency_response_coverage_ratio: f64,
    dispersive_loss_scale_mean: f64,
    dispersive_loss_scale_spread_ratio: f64,
    dispersive_phase_attenuation_mean: f64,
    dispersive_phase_attenuation_spread_ratio: f64,
    relative_permittivity_spread_ratio: f64,
    relative_permeability_spread_ratio: f64,
    assignment_heterogeneity_index: f64,
    assignment_coverage_ratio: f64,
    fallback_coefficient_ratio: f64,
    region_coefficient_contrast_index: f64,
}

#[derive(Debug, Clone)]
struct RegionElectromagneticCoefficients {
    region_id: String,
    conductivity_s_per_m: f64,
    conductivity_frequency_scale: f64,
    relative_permittivity_frequency_scale: f64,
    relative_permeability_frequency_scale: f64,
    dispersive_loss_scale: f64,
    dispersive_phase_attenuation: f64,
    relative_permittivity: f64,
    relative_permeability: f64,
    weight: f64,
    covered: bool,
    used_fallback: bool,
    has_frequency_response: bool,
    has_relative_permittivity_frequency_response: bool,
    has_relative_permeability_frequency_response: bool,
}

#[derive(Debug, Clone)]
struct ElectromagneticCoefficientProfile {
    region_coefficients: Vec<RegionElectromagneticCoefficients>,
    stats: ElectromagneticMaterialStats,
}

fn electromagnetic_coefficient_profile(
    model: &AnalysisModel,
    reference_frequency_hz: f64,
) -> ElectromagneticCoefficientProfile {
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
        let (
            conductivity,
            conductivity_frequency_scale,
            relative_permittivity_frequency_scale,
            relative_permeability_frequency_scale,
            dispersive_loss_scale,
            dispersive_phase_attenuation,
            permittivity,
            permeability,
            covered,
            used_fallback,
            has_frequency_response,
            has_relative_permittivity_frequency_response,
            has_relative_permeability_frequency_response,
        ) = if let Some(electrical) = assigned {
            let freq_sample = conductivity_frequency_sample(electrical, reference_frequency_hz);
            (
                (electrical.conductivity_s_per_m * freq_sample.scale).max(1.0e-9),
                freq_sample.scale,
                freq_sample.relative_permittivity_scale,
                freq_sample.relative_permeability_scale,
                freq_sample.dispersive_loss_scale,
                freq_sample.phase_attenuation,
                (electrical.relative_permittivity * freq_sample.relative_permittivity_scale)
                    .max(1.0e-9),
                (electrical.relative_permeability * freq_sample.relative_permeability_scale)
                    .max(1.0e-9),
                true,
                false,
                freq_sample.has_frequency_response,
                freq_sample.has_relative_permittivity_frequency_response,
                freq_sample.has_relative_permeability_frequency_response,
            )
        } else if let Some(electrical) = expected {
            let freq_sample = conductivity_frequency_sample(electrical, reference_frequency_hz);
            (
                (electrical.conductivity_s_per_m * freq_sample.scale).max(1.0e-9),
                freq_sample.scale,
                freq_sample.relative_permittivity_scale,
                freq_sample.relative_permeability_scale,
                freq_sample.dispersive_loss_scale,
                freq_sample.phase_attenuation,
                (electrical.relative_permittivity * freq_sample.relative_permittivity_scale)
                    .max(1.0e-9),
                (electrical.relative_permeability * freq_sample.relative_permeability_scale)
                    .max(1.0e-9),
                true,
                true,
                freq_sample.has_frequency_response,
                freq_sample.has_relative_permittivity_frequency_response,
                freq_sample.has_relative_permeability_frequency_response,
            )
        } else {
            (
                1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, false, true, false, false, false,
            )
        };
        covered_assignments += usize::from(covered);
        fallback_coefficients += usize::from(used_fallback);
        samples.push(RegionElectromagneticCoefficients {
            region_id: assignment.region_id.clone(),
            conductivity_s_per_m: conductivity,
            conductivity_frequency_scale,
            relative_permittivity_frequency_scale,
            relative_permeability_frequency_scale,
            dispersive_loss_scale,
            dispersive_phase_attenuation,
            relative_permittivity: permittivity,
            relative_permeability: permeability,
            weight: confidence_weight(assignment.confidence),
            covered,
            used_fallback,
            has_frequency_response,
            has_relative_permittivity_frequency_response,
            has_relative_permeability_frequency_response,
        });
    }

    if samples.is_empty() {
        for material in &model.materials {
            let Some(electrical) = material.electrical.as_ref() else {
                continue;
            };
            let freq_sample = conductivity_frequency_sample(electrical, reference_frequency_hz);
            samples.push(RegionElectromagneticCoefficients {
                region_id: format!("material_region_{}", material.material_id),
                conductivity_s_per_m: (electrical.conductivity_s_per_m * freq_sample.scale)
                    .max(1.0e-9),
                conductivity_frequency_scale: freq_sample.scale,
                relative_permittivity_frequency_scale: freq_sample.relative_permittivity_scale,
                relative_permeability_frequency_scale: freq_sample.relative_permeability_scale,
                dispersive_loss_scale: freq_sample.dispersive_loss_scale,
                dispersive_phase_attenuation: freq_sample.phase_attenuation,
                relative_permittivity: (electrical.relative_permittivity
                    * freq_sample.relative_permittivity_scale)
                    .max(1.0e-9),
                relative_permeability: (electrical.relative_permeability
                    * freq_sample.relative_permeability_scale)
                    .max(1.0e-9),
                weight: 1.0,
                covered: true,
                used_fallback: false,
                has_frequency_response: freq_sample.has_frequency_response,
                has_relative_permittivity_frequency_response: freq_sample
                    .has_relative_permittivity_frequency_response,
                has_relative_permeability_frequency_response: freq_sample
                    .has_relative_permeability_frequency_response,
            });
        }
    }

    if samples.is_empty() {
        samples.push(RegionElectromagneticCoefficients {
            region_id: "default_region_0".to_string(),
            conductivity_s_per_m: 1.0,
            conductivity_frequency_scale: 1.0,
            relative_permittivity_frequency_scale: 1.0,
            relative_permeability_frequency_scale: 1.0,
            dispersive_loss_scale: 0.0,
            dispersive_phase_attenuation: 1.0,
            relative_permittivity: 1.0,
            relative_permeability: 1.0,
            weight: 1.0,
            covered: false,
            used_fallback: true,
            has_frequency_response: false,
            has_relative_permittivity_frequency_response: false,
            has_relative_permeability_frequency_response: false,
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
    let conductivity_frequency_scale_values = samples
        .iter()
        .map(|sample| sample.conductivity_frequency_scale)
        .collect::<Vec<_>>();
    let relative_permittivity_frequency_scale_values = samples
        .iter()
        .map(|sample| sample.relative_permittivity_frequency_scale)
        .collect::<Vec<_>>();
    let relative_permeability_frequency_scale_values = samples
        .iter()
        .map(|sample| sample.relative_permeability_frequency_scale)
        .collect::<Vec<_>>();
    let dispersive_loss_scale_values = samples
        .iter()
        .map(|sample| sample.dispersive_loss_scale)
        .collect::<Vec<_>>();
    let dispersive_phase_attenuation_values = samples
        .iter()
        .map(|sample| sample.dispersive_phase_attenuation)
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
    let conductivity_frequency_scale_mean = weighted_mean(&conductivity_frequency_scale_values);
    let conductivity_frequency_scale_spread_ratio =
        spread_ratio(&conductivity_frequency_scale_values);
    let conductivity_frequency_response_coverage_ratio = samples
        .iter()
        .filter(|sample| sample.has_frequency_response)
        .count() as f64
        / samples.len().max(1) as f64;
    let relative_permittivity_frequency_scale_mean =
        weighted_mean(&relative_permittivity_frequency_scale_values);
    let relative_permittivity_frequency_scale_spread_ratio =
        spread_ratio(&relative_permittivity_frequency_scale_values);
    let relative_permittivity_frequency_response_coverage_ratio = samples
        .iter()
        .filter(|sample| sample.has_relative_permittivity_frequency_response)
        .count() as f64
        / samples.len().max(1) as f64;
    let relative_permeability_frequency_scale_mean =
        weighted_mean(&relative_permeability_frequency_scale_values);
    let relative_permeability_frequency_scale_spread_ratio =
        spread_ratio(&relative_permeability_frequency_scale_values);
    let relative_permeability_frequency_response_coverage_ratio = samples
        .iter()
        .filter(|sample| sample.has_relative_permeability_frequency_response)
        .count() as f64
        / samples.len().max(1) as f64;
    let dispersive_loss_scale_mean = weighted_mean(&dispersive_loss_scale_values);
    let dispersive_loss_scale_spread_ratio = spread_ratio(&dispersive_loss_scale_values);
    let dispersive_phase_attenuation_mean = weighted_mean(&dispersive_phase_attenuation_values);
    let dispersive_phase_attenuation_spread_ratio =
        spread_ratio(&dispersive_phase_attenuation_values);
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
            conductivity_frequency_scale_mean,
            conductivity_frequency_scale_spread_ratio,
            conductivity_frequency_response_coverage_ratio,
            relative_permittivity_frequency_scale_mean,
            relative_permittivity_frequency_scale_spread_ratio,
            relative_permittivity_frequency_response_coverage_ratio,
            relative_permeability_frequency_scale_mean,
            relative_permeability_frequency_scale_spread_ratio,
            relative_permeability_frequency_response_coverage_ratio,
            dispersive_loss_scale_mean,
            dispersive_loss_scale_spread_ratio,
            dispersive_phase_attenuation_mean,
            dispersive_phase_attenuation_spread_ratio,
            relative_permittivity_spread_ratio,
            relative_permeability_spread_ratio,
            assignment_heterogeneity_index,
            assignment_coverage_ratio,
            fallback_coefficient_ratio,
            region_coefficient_contrast_index,
        },
    }
}

#[derive(Debug, Clone, Copy)]
struct ConductivityFrequencySample {
    scale: f64,
    relative_permittivity_scale: f64,
    relative_permeability_scale: f64,
    dispersive_loss_scale: f64,
    phase_attenuation: f64,
    has_frequency_response: bool,
    has_relative_permittivity_frequency_response: bool,
    has_relative_permeability_frequency_response: bool,
}

fn conductivity_frequency_sample(
    electrical: &runmat_analysis_core::MaterialElectricalModel,
    reference_frequency_hz: f64,
) -> ConductivityFrequencySample {
    let scale = conductivity_scale_at_frequency(
        &electrical.conductivity_frequency_response,
        reference_frequency_hz,
    )
    .unwrap_or(1.0)
    .clamp(1.0e-4, 1.0e4);
    let dispersive_loss_scale = dispersive_loss_scale_at_frequency(
        &electrical.conductivity_frequency_response,
        reference_frequency_hz,
    )
    .unwrap_or(0.0)
    .clamp(0.0, 10.0);
    let relative_permittivity_scale = relative_permittivity_scale_at_frequency(
        &electrical.conductivity_frequency_response,
        reference_frequency_hz,
    )
    .unwrap_or(1.0)
    .clamp(1.0e-4, 1.0e4);
    let relative_permeability_scale = relative_permeability_scale_at_frequency(
        &electrical.conductivity_frequency_response,
        reference_frequency_hz,
    )
    .unwrap_or(1.0)
    .clamp(1.0e-4, 1.0e4);
    let phase_attenuation = dispersive_phase_attenuation_for_loss_scale(dispersive_loss_scale);
    ConductivityFrequencySample {
        scale,
        relative_permittivity_scale,
        relative_permeability_scale,
        dispersive_loss_scale,
        phase_attenuation,
        has_frequency_response: !electrical.conductivity_frequency_response.is_empty(),
        has_relative_permittivity_frequency_response: electrical
            .conductivity_frequency_response
            .iter()
            .any(|point| point.relative_permittivity_scale.is_some()),
        has_relative_permeability_frequency_response: electrical
            .conductivity_frequency_response
            .iter()
            .any(|point| point.relative_permeability_scale.is_some()),
    }
}

fn dispersive_phase_attenuation_for_loss_scale(dispersive_loss_scale: f64) -> f64 {
    let bounded = dispersive_loss_scale.clamp(0.0, 10.0);
    (1.0 / (1.0 + bounded * bounded)).clamp(0.05, 1.0)
}

fn conductivity_scale_at_frequency(
    response: &[runmat_analysis_core::ConductivityFrequencyPoint],
    reference_frequency_hz: f64,
) -> Option<f64> {
    frequency_response_value_at_frequency(response, reference_frequency_hz, |point| {
        Some(point.conductivity_scale)
    })
}

fn dispersive_loss_scale_at_frequency(
    response: &[runmat_analysis_core::ConductivityFrequencyPoint],
    reference_frequency_hz: f64,
) -> Option<f64> {
    frequency_response_value_at_frequency(response, reference_frequency_hz, |point| {
        point.dispersive_loss_scale
    })
}

fn relative_permittivity_scale_at_frequency(
    response: &[runmat_analysis_core::ConductivityFrequencyPoint],
    reference_frequency_hz: f64,
) -> Option<f64> {
    frequency_response_value_at_frequency(response, reference_frequency_hz, |point| {
        point.relative_permittivity_scale
    })
}

fn relative_permeability_scale_at_frequency(
    response: &[runmat_analysis_core::ConductivityFrequencyPoint],
    reference_frequency_hz: f64,
) -> Option<f64> {
    frequency_response_value_at_frequency(response, reference_frequency_hz, |point| {
        point.relative_permeability_scale
    })
}

fn frequency_response_value_at_frequency<F>(
    response: &[runmat_analysis_core::ConductivityFrequencyPoint],
    reference_frequency_hz: f64,
    value: F,
) -> Option<f64>
where
    F: Fn(&runmat_analysis_core::ConductivityFrequencyPoint) -> Option<f64>,
{
    if !reference_frequency_hz.is_finite() || reference_frequency_hz <= 0.0 {
        return None;
    }
    let mut points = response
        .iter()
        .filter_map(|point| {
            let value = value(point)?;
            if point.frequency_hz.is_finite()
                && point.frequency_hz > 0.0
                && value.is_finite()
                && value >= 0.0
            {
                Some((point.frequency_hz, value))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if points.is_empty() {
        return None;
    }
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    if reference_frequency_hz <= points[0].0 {
        return Some(points[0].1);
    }
    if reference_frequency_hz >= points[points.len() - 1].0 {
        return Some(points[points.len() - 1].1);
    }
    for window in points.windows(2) {
        let (f0, s0) = window[0];
        let (f1, s1) = window[1];
        if reference_frequency_hz >= f0 && reference_frequency_hz <= f1 {
            if (f1 - f0).abs() <= f64::EPSILON {
                return Some(s1);
            }
            let t = (reference_frequency_hz.ln() - f0.ln()) / (f1.ln() - f0.ln()).max(f64::EPSILON);
            return Some((s0 + (s1 - s0) * t.clamp(0.0, 1.0)).max(0.0));
        }
    }
    Some(points[points.len() - 1].1)
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

fn region_span_for_index(
    region_index: usize,
    region_count: usize,
    node_count: usize,
) -> (usize, usize) {
    if region_count == 0 || node_count == 0 {
        return (0, 0);
    }
    let start = (region_index * node_count) / region_count;
    let mut end = ((region_index + 1) * node_count) / region_count;
    if end <= start {
        end = (start + 1).min(node_count);
    }
    (start.min(node_count), end.min(node_count))
}

fn regional_source_gain(
    region: &RegionElectromagneticCoefficients,
    stats: &ElectromagneticMaterialStats,
    coil_source: bool,
) -> f64 {
    let sigma_norm =
        (region.conductivity_s_per_m / stats.conductivity_mean.max(1.0e-9)).clamp(0.2, 5.0);
    let eps_norm = (region.relative_permittivity / stats.relative_permittivity_mean.max(1.0e-9))
        .clamp(0.2, 5.0);
    let mu_norm = (region.relative_permeability / stats.relative_permeability_mean.max(1.0e-9))
        .clamp(0.2, 5.0);
    if coil_source {
        (0.65 * mu_norm.sqrt() + 0.35 * sigma_norm.sqrt()).clamp(0.3, 3.0)
    } else {
        (0.70 * sigma_norm.sqrt() + 0.30 * eps_norm.sqrt()).clamp(0.3, 3.0)
    }
}

fn source_overlap_ratio(
    per_source_real_profiles: &[Vec<f64>],
    per_source_imag_profiles: &[Vec<f64>],
) -> f64 {
    if per_source_real_profiles.is_empty() || per_source_imag_profiles.is_empty() {
        return 0.0;
    }
    let node_count = per_source_real_profiles[0].len();
    if node_count == 0 {
        return 0.0;
    }
    let mut active_nodes = 0usize;
    let mut phase_conflict_nodes = 0usize;
    for node_index in 0..node_count {
        let mut contributors = 0usize;
        let mut sum_real = 0.0_f64;
        let mut sum_imag = 0.0_f64;
        let mut sum_mag = 0.0_f64;
        for (real_profile, imag_profile) in per_source_real_profiles
            .iter()
            .zip(per_source_imag_profiles.iter())
        {
            let real = real_profile[node_index];
            let imag = imag_profile[node_index];
            let magnitude = (real * real + imag * imag).sqrt();
            if magnitude > 1.0e-12 {
                contributors += 1;
                sum_real += real;
                sum_imag += imag;
                sum_mag += magnitude;
            }
        }
        if contributors > 0 {
            active_nodes += 1;
            let net_mag = (sum_real * sum_real + sum_imag * sum_imag).sqrt();
            let cancellation = if sum_mag <= 1.0e-12 {
                0.0
            } else {
                (1.0 - net_mag / sum_mag).clamp(0.0, 1.0)
            };
            if contributors > 1 && cancellation > 0.15 {
                phase_conflict_nodes += 1;
            }
        }
    }
    if active_nodes == 0 {
        0.0
    } else {
        phase_conflict_nodes as f64 / active_nodes as f64
    }
}

fn source_interference_index(
    per_source_real_profiles: &[Vec<f64>],
    per_source_imag_profiles: &[Vec<f64>],
) -> f64 {
    if per_source_real_profiles.is_empty() || per_source_imag_profiles.is_empty() {
        return 0.0;
    }
    let node_count = per_source_real_profiles[0].len();
    if node_count == 0 {
        return 0.0;
    }
    let mut cancellation_sum = 0.0_f64;
    let mut active_nodes = 0usize;
    for node_index in 0..node_count {
        let mut sum_real = 0.0_f64;
        let mut sum_imag = 0.0_f64;
        let mut sum_mag = 0.0_f64;
        for (real_profile, imag_profile) in per_source_real_profiles
            .iter()
            .zip(per_source_imag_profiles.iter())
        {
            let real = real_profile[node_index];
            let imag = imag_profile[node_index];
            let magnitude = (real * real + imag * imag).sqrt();
            sum_real += real;
            sum_imag += imag;
            sum_mag += magnitude;
        }
        if sum_mag > 1.0e-12 {
            let net_mag = (sum_real * sum_real + sum_imag * sum_imag).sqrt();
            cancellation_sum += (1.0 - net_mag / sum_mag).clamp(0.0, 1.0);
            active_nodes += 1;
        }
    }
    if active_nodes == 0 {
        0.0
    } else {
        (cancellation_sum / active_nodes as f64).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone)]
struct HarmonicBlockSolveResult {
    real_solution: Vec<f64>,
    imag_solution: Vec<f64>,
    residual_norm: f64,
    iterations: usize,
    converged: bool,
}

fn solve_harmonic_block_system(
    operator: &OperatorSystem,
    coupling_terms: &[f64],
    rhs_real: &[f64],
    rhs_imag: &[f64],
    max_iters: usize,
    tol: f64,
) -> HarmonicBlockSolveResult {
    let n = operator.dof_count;
    let size = n * 2;
    let mut b = vec![0.0_f64; size];
    for i in 0..n {
        if operator.constrained[i] {
            b[i] = 0.0;
            b[n + i] = 0.0;
        } else {
            b[i] = rhs_real[i];
            b[n + i] = rhs_imag[i];
        }
    }
    let b_norm = block_dot(&b, &b).sqrt().max(1.0e-9);
    let mut x = vec![0.0_f64; size];
    let mut r = b.clone();
    let r_hat = r.clone();
    let mut p = vec![0.0_f64; size];
    let mut v = vec![0.0_f64; size];
    let mut rho_old = 1.0_f64;
    let mut alpha = 1.0_f64;
    let mut omega = 1.0_f64;
    let mut converged = false;
    let mut iterations = 0usize;
    let mut residual_norm = block_dot(&r, &r).sqrt() / b_norm;
    if residual_norm <= tol {
        converged = true;
    } else {
        for iter in 0..max_iters {
            let rho_new = block_dot(&r_hat, &r);
            if rho_new.abs() <= 1.0e-30 {
                break;
            }
            let beta = (rho_new / rho_old) * (alpha / omega);
            for i in 0..size {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
            let p_hat = apply_block_jacobi_preconditioner(operator, coupling_terms, &p);
            v = apply_harmonic_block_operator(operator, coupling_terms, &p_hat);
            let denom = block_dot(&r_hat, &v);
            if denom.abs() <= 1.0e-30 {
                break;
            }
            alpha = rho_new / denom;
            let mut s = vec![0.0_f64; size];
            for i in 0..size {
                s[i] = r[i] - alpha * v[i];
            }
            let s_norm = block_dot(&s, &s).sqrt() / b_norm;
            if s_norm <= tol {
                for i in 0..size {
                    x[i] += alpha * p_hat[i];
                }
                converged = true;
                iterations = iter + 1;
                residual_norm = s_norm;
                break;
            }
            let s_hat = apply_block_jacobi_preconditioner(operator, coupling_terms, &s);
            let t = apply_harmonic_block_operator(operator, coupling_terms, &s_hat);
            let tt = block_dot(&t, &t);
            if tt.abs() <= 1.0e-30 {
                break;
            }
            omega = block_dot(&t, &s) / tt;
            for i in 0..size {
                x[i] += alpha * p_hat[i] + omega * s_hat[i];
            }
            for i in 0..size {
                r[i] = s[i] - omega * t[i];
            }
            residual_norm = block_dot(&r, &r).sqrt() / b_norm;
            iterations = iter + 1;
            if residual_norm <= tol {
                converged = true;
                break;
            }
            if omega.abs() <= 1.0e-30 {
                break;
            }
            rho_old = rho_new;
        }
    }

    HarmonicBlockSolveResult {
        real_solution: x[..n].to_vec(),
        imag_solution: x[n..].to_vec(),
        residual_norm,
        iterations,
        converged,
    }
}

fn apply_harmonic_block_operator(
    operator: &OperatorSystem,
    coupling_terms: &[f64],
    vector: &[f64],
) -> Vec<f64> {
    let n = operator.dof_count;
    let real = &vector[..n];
    let imag = &vector[n..];
    let k_real = apply_k(operator, real);
    let k_imag = apply_k(operator, imag);
    let mut out = vec![0.0_f64; n * 2];
    for i in 0..n {
        if operator.constrained[i] {
            out[i] = real[i];
            out[n + i] = imag[i];
        } else {
            let c = coupling_terms[i];
            out[i] = k_real[i] - c * imag[i];
            out[n + i] = k_imag[i] + c * real[i];
        }
    }
    out
}

fn apply_block_jacobi_preconditioner(
    operator: &OperatorSystem,
    coupling_terms: &[f64],
    vector: &[f64],
) -> Vec<f64> {
    let n = operator.dof_count;
    let real = &vector[..n];
    let imag = &vector[n..];
    let mut out = vec![0.0_f64; n * 2];
    for i in 0..n {
        if operator.constrained[i] {
            out[i] = real[i];
            out[n + i] = imag[i];
            continue;
        }
        let k = operator.stiffness_diag[i].max(1.0e-12);
        let c = coupling_terms[i];
        let denom = (k * k + c * c).max(1.0e-18);
        out[i] = (k * real[i] + c * imag[i]) / denom;
        out[n + i] = (-c * real[i] + k * imag[i]) / denom;
    }
    out
}

fn block_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[derive(Debug, Clone, PartialEq)]
struct MaxwellEdgeTopology {
    edges: Vec<MaxwellEdge>,
    gauge_anchor_nodes: Vec<usize>,
}

impl MaxwellEdgeTopology {
    fn line_graph(node_count: usize, spacing: f64, constrained: &[bool]) -> Self {
        let edge_count = node_count.saturating_sub(1);
        let length = if spacing.is_finite() && spacing > 0.0 {
            spacing
        } else {
            1.0
        };
        let edges = (0..edge_count)
            .map(|from_node| {
                let to_node = from_node + 1;
                MaxwellEdge {
                    from_node,
                    to_node,
                    orientation: 1.0,
                    length,
                    constrained: constrained.get(from_node).copied().unwrap_or(false)
                        || constrained.get(to_node).copied().unwrap_or(false),
                }
            })
            .collect::<Vec<_>>();
        let gauge_anchor_nodes = constrained
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value).then_some(index))
            .collect::<Vec<_>>();
        Self {
            edges,
            gauge_anchor_nodes,
        }
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn element_count(&self) -> usize {
        self.edges.len()
    }

    fn oriented_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| edge.orientation.abs() > 0.0)
            .count()
    }

    fn constrained_edge_count(&self) -> usize {
        self.edges.iter().filter(|edge| edge.constrained).count()
    }

    fn gauge_anchor_count(&self) -> usize {
        self.gauge_anchor_nodes.len()
    }

    fn curl_curl_energy_scale(&self) -> f64 {
        self.edges
            .iter()
            .map(|edge| edge.curl_weight() * edge.curl_weight())
            .sum::<f64>()
    }

    fn edge_line_integrals(&self, nodal_values: &[f64]) -> Vec<f64> {
        self.edges
            .iter()
            .map(|edge| {
                let from = nodal_values.get(edge.from_node).copied().unwrap_or(0.0);
                let to = nodal_values.get(edge.to_node).copied().unwrap_or(0.0);
                0.5 * (from + to) * edge.length * edge.orientation
            })
            .collect()
    }

    fn curl_from_nodal_vector_potential(
        &self,
        real: &[f64],
        imag: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut flux_density_real = Vec::with_capacity(self.edges.len());
        let mut flux_density_imag = Vec::with_capacity(self.edges.len());
        let mut flux_density_magnitude = Vec::with_capacity(self.edges.len());
        for edge in &self.edges {
            let real_curl = edge.oriented_difference(real);
            let imag_curl = edge.oriented_difference(imag);
            flux_density_real.push(real_curl);
            flux_density_imag.push(imag_curl);
            flux_density_magnitude.push((real_curl * real_curl + imag_curl * imag_curl).sqrt());
        }
        (flux_density_real, flux_density_imag, flux_density_magnitude)
    }

    fn curl_from_nodal_magnitude(&self, magnitude: &[f64]) -> Vec<f64> {
        self.edges
            .iter()
            .map(|edge| edge.oriented_difference(magnitude).abs())
            .collect()
    }

    fn gauge_anchor_residual_ratio(&self, real: &[f64], imag: &[f64]) -> f64 {
        let mut anchor_energy = 0.0_f64;
        let mut total_energy = 0.0_f64;
        for node_index in 0..real.len().max(imag.len()) {
            let real_value = real.get(node_index).copied().unwrap_or(0.0);
            let imag_value = imag.get(node_index).copied().unwrap_or(0.0);
            let energy = real_value * real_value + imag_value * imag_value;
            total_energy += energy;
            if self.gauge_anchor_nodes.binary_search(&node_index).is_ok() {
                anchor_energy += energy;
            }
        }
        if total_energy <= 1.0e-18 {
            0.0
        } else {
            (anchor_energy / total_energy).sqrt()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MaxwellEdge {
    from_node: usize,
    to_node: usize,
    orientation: f64,
    length: f64,
    constrained: bool,
}

impl MaxwellEdge {
    fn curl_weight(&self) -> f64 {
        self.orientation / self.length.max(1.0e-12)
    }

    fn oriented_difference(&self, values: &[f64]) -> f64 {
        let from = values.get(self.from_node).copied().unwrap_or(0.0);
        let to = values.get(self.to_node).copied().unwrap_or(0.0);
        (to - from) * self.curl_weight()
    }

    fn average(&self, values: &[f64]) -> f64 {
        let from = values.get(self.from_node).copied().unwrap_or(0.0);
        let to = values.get(self.to_node).copied().unwrap_or(from);
        0.5 * (from + to)
    }

    fn tangential_component(&self, line_integral: f64) -> f64 {
        line_integral * self.orientation / self.length.max(1.0e-12)
    }
}

#[cfg(test)]
fn flux_density_from_complex_vector_potential(
    real: &[f64],
    imag: &[f64],
    h: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let node_count = real.len().min(imag.len());
    let mut flux_density_real = vec![0.0_f64; node_count];
    let mut flux_density_imag = vec![0.0_f64; node_count];
    let mut flux_density_magnitude = vec![0.0_f64; node_count];
    if node_count < 2 || !h.is_finite() || h <= 0.0 {
        return (flux_density_real, flux_density_imag, flux_density_magnitude);
    }

    let inv_two_h = 0.5 / h;
    for i in 1..(node_count - 1) {
        let grad_real = (real[i + 1] - real[i - 1]) * inv_two_h;
        let grad_imag = (imag[i + 1] - imag[i - 1]) * inv_two_h;
        flux_density_real[i] = grad_real;
        flux_density_imag[i] = grad_imag;
        flux_density_magnitude[i] = (grad_real * grad_real + grad_imag * grad_imag).sqrt();
    }
    flux_density_real[0] = flux_density_real[1];
    flux_density_real[node_count - 1] = flux_density_real[node_count - 2];
    flux_density_imag[0] = flux_density_imag[1];
    flux_density_imag[node_count - 1] = flux_density_imag[node_count - 2];
    flux_density_magnitude[0] = flux_density_magnitude[1];
    flux_density_magnitude[node_count - 1] = flux_density_magnitude[node_count - 2];
    (flux_density_real, flux_density_imag, flux_density_magnitude)
}

fn axial_vector_field_values(values: &[f64]) -> Vec<f64> {
    let mut vector_values = Vec::with_capacity(values.len() * 3);
    for value in values {
        vector_values.extend_from_slice(&[*value, 0.0, 0.0]);
    }
    vector_values
}

#[cfg(test)]
fn flux_density_from_magnitude_vector_potential(magnitude: &[f64], h: f64) -> Vec<f64> {
    let node_count = magnitude.len();
    let mut flux_density = vec![0.0_f64; node_count];
    if node_count < 2 || !h.is_finite() || h <= 0.0 {
        return flux_density;
    }
    let inv_two_h = 0.5 / h;
    for i in 1..(node_count - 1) {
        flux_density[i] = ((magnitude[i + 1] - magnitude[i - 1]) * inv_two_h).abs();
    }
    flux_density[0] = flux_density[1];
    flux_density[node_count - 1] = flux_density[node_count - 2];
    flux_density
}

fn flux_phasor_coherence_ratio(complex_flux: &[f64], magnitude_estimate_flux: &[f64]) -> f64 {
    let pair_count = complex_flux.len().min(magnitude_estimate_flux.len());
    if pair_count == 0 {
        return 1.0;
    }
    let mut complex_energy = 0.0_f64;
    let mut estimate_energy = 0.0_f64;
    for i in 0..pair_count {
        complex_energy += complex_flux[i] * complex_flux[i];
        estimate_energy += magnitude_estimate_flux[i] * magnitude_estimate_flux[i];
    }
    if complex_energy <= 1.0e-18 {
        1.0
    } else {
        (estimate_energy / complex_energy).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{ConductivityFrequencyPoint, MaterialElectricalModel};

    use super::{
        conductivity_frequency_sample, conductivity_scale_at_frequency,
        dispersive_loss_scale_at_frequency, dispersive_phase_attenuation_for_loss_scale,
        flux_density_from_complex_vector_potential, flux_density_from_magnitude_vector_potential,
        flux_phasor_coherence_ratio, relative_permeability_scale_at_frequency,
        relative_permittivity_scale_at_frequency, MaxwellEdgeTopology,
    };

    #[test]
    fn conductivity_scale_interpolates_on_log_frequency_axis() {
        let response = vec![
            ConductivityFrequencyPoint {
                frequency_hz: 10.0,
                conductivity_scale: 2.0,
                dispersive_loss_scale: Some(0.1),
                relative_permittivity_scale: None,
                relative_permeability_scale: None,
            },
            ConductivityFrequencyPoint {
                frequency_hz: 1_000.0,
                conductivity_scale: 0.5,
                dispersive_loss_scale: Some(0.2),
                relative_permittivity_scale: None,
                relative_permeability_scale: None,
            },
        ];
        let scale = conductivity_scale_at_frequency(&response, 100.0).expect("scale should exist");
        assert!((scale - 1.25).abs() < 1.0e-9, "expected 1.25, got {scale}");
    }

    #[test]
    fn conductivity_scale_clamps_to_edge_points() {
        let response = vec![
            ConductivityFrequencyPoint {
                frequency_hz: 20.0,
                conductivity_scale: 1.2,
                dispersive_loss_scale: None,
                relative_permittivity_scale: None,
                relative_permeability_scale: None,
            },
            ConductivityFrequencyPoint {
                frequency_hz: 200.0,
                conductivity_scale: 0.8,
                dispersive_loss_scale: None,
                relative_permittivity_scale: None,
                relative_permeability_scale: None,
            },
        ];
        let low = conductivity_scale_at_frequency(&response, 5.0).expect("low edge");
        let high = conductivity_scale_at_frequency(&response, 500.0).expect("high edge");
        assert!((low - 1.2).abs() < 1.0e-12, "expected low edge");
        assert!((high - 0.8).abs() < 1.0e-12, "expected high edge");
    }

    #[test]
    fn conductivity_frequency_sample_defaults_when_response_is_empty() {
        let electrical = MaterialElectricalModel::default();
        let sample = conductivity_frequency_sample(&electrical, 60.0);
        assert!((sample.scale - 1.0).abs() < 1.0e-12);
        assert!((sample.relative_permittivity_scale - 1.0).abs() < 1.0e-12);
        assert!((sample.relative_permeability_scale - 1.0).abs() < 1.0e-12);
        assert!((sample.dispersive_loss_scale - 0.0).abs() < 1.0e-12);
        assert!((sample.phase_attenuation - 1.0).abs() < 1.0e-12);
        assert!(!sample.has_frequency_response);
        assert!(!sample.has_relative_permittivity_frequency_response);
        assert!(!sample.has_relative_permeability_frequency_response);
    }

    #[test]
    fn dispersive_loss_scale_interpolates_on_log_frequency_axis() {
        let response = vec![
            ConductivityFrequencyPoint {
                frequency_hz: 20.0,
                conductivity_scale: 1.1,
                dispersive_loss_scale: Some(0.08),
                relative_permittivity_scale: None,
                relative_permeability_scale: None,
            },
            ConductivityFrequencyPoint {
                frequency_hz: 500.0,
                conductivity_scale: 0.9,
                dispersive_loss_scale: Some(0.2),
                relative_permittivity_scale: None,
                relative_permeability_scale: None,
            },
        ];
        let value =
            dispersive_loss_scale_at_frequency(&response, 100.0).expect("loss scale should exist");
        assert!(
            (value - 0.14).abs() < 1.0e-9,
            "unexpected interpolated value: {value}"
        );
    }

    #[test]
    fn dispersive_phase_attenuation_decreases_with_loss_scale() {
        let low = dispersive_phase_attenuation_for_loss_scale(0.05);
        let high = dispersive_phase_attenuation_for_loss_scale(0.8);
        assert!(
            low > high,
            "expected attenuation to decrease for higher loss scale"
        );
        assert!((0.0..=1.0).contains(&low));
        assert!((0.0..=1.0).contains(&high));
    }

    #[test]
    fn relative_permittivity_scale_interpolates_on_log_frequency_axis() {
        let response = vec![
            ConductivityFrequencyPoint {
                frequency_hz: 10.0,
                conductivity_scale: 1.0,
                dispersive_loss_scale: None,
                relative_permittivity_scale: Some(1.2),
                relative_permeability_scale: None,
            },
            ConductivityFrequencyPoint {
                frequency_hz: 1_000.0,
                conductivity_scale: 1.0,
                dispersive_loss_scale: None,
                relative_permittivity_scale: Some(0.8),
                relative_permeability_scale: None,
            },
        ];
        let scale = relative_permittivity_scale_at_frequency(&response, 100.0)
            .expect("permittivity scale should exist");
        assert!((scale - 1.0).abs() < 1.0e-9, "expected 1.0, got {scale}");
    }

    #[test]
    fn relative_permeability_scale_interpolates_on_log_frequency_axis() {
        let response = vec![
            ConductivityFrequencyPoint {
                frequency_hz: 10.0,
                conductivity_scale: 1.0,
                dispersive_loss_scale: None,
                relative_permittivity_scale: None,
                relative_permeability_scale: Some(2.0),
            },
            ConductivityFrequencyPoint {
                frequency_hz: 1_000.0,
                conductivity_scale: 1.0,
                dispersive_loss_scale: None,
                relative_permittivity_scale: None,
                relative_permeability_scale: Some(0.5),
            },
        ];
        let scale = relative_permeability_scale_at_frequency(&response, 100.0)
            .expect("permeability scale should exist");
        assert!((scale - 1.25).abs() < 1.0e-9, "expected 1.25, got {scale}");
    }

    #[test]
    fn conductivity_frequency_sample_tracks_relative_response_coverage() {
        let electrical = MaterialElectricalModel {
            conductivity_frequency_response: vec![
                ConductivityFrequencyPoint {
                    frequency_hz: 10.0,
                    conductivity_scale: 1.1,
                    dispersive_loss_scale: Some(0.05),
                    relative_permittivity_scale: Some(0.9),
                    relative_permeability_scale: None,
                },
                ConductivityFrequencyPoint {
                    frequency_hz: 1_000.0,
                    conductivity_scale: 0.9,
                    dispersive_loss_scale: Some(0.2),
                    relative_permittivity_scale: Some(1.1),
                    relative_permeability_scale: Some(1.2),
                },
            ],
            ..MaterialElectricalModel::default()
        };
        let sample = conductivity_frequency_sample(&electrical, 100.0);
        assert!(sample.has_frequency_response);
        assert!(sample.has_relative_permittivity_frequency_response);
        assert!(sample.has_relative_permeability_frequency_response);
        assert!(sample.relative_permittivity_scale > 0.0);
        assert!(sample.relative_permeability_scale > 0.0);
    }

    #[test]
    fn maxwell_edge_topology_recovers_oriented_edge_curl() {
        let topology = MaxwellEdgeTopology::line_graph(4, 0.25, &[true, false, false, true]);
        let real = vec![0.0, 0.25, 0.50, 0.75];
        let imag = vec![0.0, 0.0, 0.25, 0.25];

        let edge_real = topology.edge_line_integrals(&real);
        let (curl_real, curl_imag, curl_mag) =
            topology.curl_from_nodal_vector_potential(&real, &imag);

        assert_eq!(topology.edge_count(), 3);
        assert_eq!(topology.element_count(), 3);
        assert_eq!(topology.oriented_edge_count(), 3);
        assert_eq!(topology.constrained_edge_count(), 2);
        assert_eq!(topology.gauge_anchor_count(), 2);
        assert!((edge_real[0] - 0.03125).abs() < 1.0e-12);
        assert!(curl_real.iter().all(|value| (*value - 1.0).abs() < 1.0e-12));
        assert!((curl_imag[1] - 1.0).abs() < 1.0e-12);
        assert!(curl_mag[1] > curl_mag[0]);
    }

    #[test]
    fn maxwell_edge_topology_reports_zero_gauge_residual_when_anchors_are_zero() {
        let topology = MaxwellEdgeTopology::line_graph(5, 0.25, &[true, false, false, false, true]);
        let real = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let imag = vec![0.0, 0.2, 0.4, 0.2, 0.0];

        let residual = topology.gauge_anchor_residual_ratio(&real, &imag);

        assert!(residual <= 1.0e-12);
    }

    #[test]
    fn complex_flux_density_reconstruction_preserves_quadrature_gradients() {
        let real = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let imag = vec![1.0, 0.0, -1.0, 0.0, 1.0];
        let (_, _, flux_density) = flux_density_from_complex_vector_potential(&real, &imag, 0.25);

        assert!(flux_density[2] > 3.5);
        assert!((flux_density[0] - flux_density[1]).abs() < 1.0e-12);
        assert!((flux_density[4] - flux_density[3]).abs() < 1.0e-12);
    }

    #[test]
    fn complex_flux_density_reconstruction_zero_for_uniform_phasor() {
        let real = vec![2.0; 8];
        let imag = vec![-1.0; 8];
        let (_, _, flux_density) = flux_density_from_complex_vector_potential(&real, &imag, 0.1);

        assert!(flux_density.iter().all(|value| value.abs() <= 1.0e-12));
    }

    #[test]
    fn flux_phasor_coherence_ratio_drops_for_quadrature_variation() {
        let real = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let imag = vec![1.0, 0.0, -1.0, 0.0, 1.0];
        let (_, _, complex_flux) = flux_density_from_complex_vector_potential(&real, &imag, 0.25);
        let magnitude = real
            .iter()
            .zip(imag.iter())
            .map(|(re, im)| (re * re + im * im).sqrt())
            .collect::<Vec<_>>();
        let magnitude_estimate = flux_density_from_magnitude_vector_potential(&magnitude, 0.25);
        let coherence = flux_phasor_coherence_ratio(&complex_flux, &magnitude_estimate);

        assert!(
            coherence < 0.25,
            "expected strong quadrature loss, got {coherence}"
        );
    }

    #[test]
    fn flux_phasor_coherence_ratio_is_unity_for_in_phase_signal() {
        let real = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let imag = vec![0.0; 5];
        let (_, _, complex_flux) = flux_density_from_complex_vector_potential(&real, &imag, 0.5);
        let magnitude = real
            .iter()
            .zip(imag.iter())
            .map(|(re, im)| (re * re + im * im).sqrt())
            .collect::<Vec<_>>();
        let magnitude_estimate = flux_density_from_magnitude_vector_potential(&magnitude, 0.5);
        let coherence = flux_phasor_coherence_ratio(&complex_flux, &magnitude_estimate);

        assert!(
            (coherence - 1.0).abs() < 1.0e-12,
            "expected unity coherence, got {coherence}"
        );
    }
}
