use runmat_analysis_core::AnalysisModel;
use serde::{Deserialize, Serialize};

use crate::operator::OperatorSystem;
use crate::{FeaPrepCalibrationProfile, FeaPrepContext, FeaThermoMechanicalContext};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblySummary {
    pub dof_count: usize,
    pub constrained_dof_count: usize,
    pub load_count: usize,
    pub prep_assembly: Option<PrepAssemblySummary>,
    pub prep_operator_topology: Option<PrepOperatorTopologySummary>,
    pub prep_region_topology: Option<PrepRegionTopologySummary>,
    pub prep_element_assembly: Option<PrepElementAssemblySummary>,
    pub prep_element_connectivity: Option<PrepElementConnectivitySummary>,
    pub prep_graph_assembly: Option<PrepGraphAssemblySummary>,
    pub prep_calibration: Option<PrepCalibrationSummary>,
    pub prep_acceptance: Option<PrepAcceptanceSummary>,
    pub thermo_mechanical: Option<ThermoMechanicalAssemblySummary>,
    pub operator: OperatorSystem,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepAssemblySummary {
    pub active_region_count: usize,
    pub mapped_load_count: usize,
    pub mapped_bc_count: usize,
    pub mapped_load_ratio: f64,
    pub constrained_prep_ratio: f64,
    pub layout_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepOperatorTopologySummary {
    pub stiffness_scale: f64,
    pub mass_scale: f64,
    pub damping_scale: f64,
    pub rhs_scale: f64,
    pub coupling_nonzero_ratio: f64,
    pub stiffness_spread_ratio: f64,
    pub topology_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepRegionTopologySummary {
    pub region_block_count: usize,
    pub inter_block_edge_count: usize,
    pub coupling_nonzero_ratio: f64,
    pub block_size_min: usize,
    pub block_size_max: usize,
    pub block_size_mean: f64,
    pub region_topology_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepElementAssemblySummary {
    pub assembled_element_count: usize,
    pub triangle_element_count: usize,
    pub quad_element_count: usize,
    pub tet_element_count: usize,
    pub hex_element_count: usize,
    pub mixed_element_count: usize,
    pub scatter_nnz_count: usize,
    pub assembly_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepElementConnectivitySummary {
    pub assembled_element_count: usize,
    pub stiffness_offdiag_nnz_count: usize,
    pub mass_offdiag_proxy_nnz_count: usize,
    pub damping_offdiag_proxy_nnz_count: usize,
    pub triangle_contrib_share: f64,
    pub quad_contrib_share: f64,
    pub tet_contrib_share: f64,
    pub hex_contrib_share: f64,
    pub mixed_contrib_share: f64,
    pub mean_connectivity_hop: f64,
    pub connectivity_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepGraphAssemblySummary {
    pub node_count: usize,
    pub edge_count: usize,
    pub degree_min: usize,
    pub degree_max: usize,
    pub degree_mean: f64,
    pub degree_p95: f64,
    pub fill_ratio: f64,
    pub connected_component_count: usize,
    pub ordering_bandwidth_before: usize,
    pub ordering_bandwidth_after: usize,
    pub ordering_reduction_ratio: f64,
    pub ordering_fingerprint: u64,
    pub recommend_ilu0: bool,
    pub graph_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepCalibrationSummary {
    pub profile: String,
    pub triangle_weight: f64,
    pub quad_weight: f64,
    pub tet_weight: f64,
    pub hex_weight: f64,
    pub mixed_weight: f64,
    pub stiffness_calibration_scale: f64,
    pub mass_calibration_scale: f64,
    pub damping_calibration_scale: f64,
    pub calibration_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepAcceptanceSummary {
    pub profile: String,
    pub accepted: bool,
    pub bounded_displacement_scale: bool,
    pub bounded_stress_scale: bool,
    pub bounded_connectivity_fill: bool,
    pub acceptance_score: f64,
    pub acceptance_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoMechanicalAssemblySummary {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_temperature_delta_k: f64,
    pub thermal_expansion_coefficient: f64,
    pub thermal_strain_scale: f64,
    pub thermal_load_scale: f64,
    pub constitutive_temperature_factor: f64,
    pub constitutive_poisson_coupling: f64,
    pub effective_modulus_scale: f64,
    pub coupling_fingerprint: u64,
}

pub fn assemble_linear_system(
    model: &AnalysisModel,
    prep_context: Option<FeaPrepContext>,
    thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
) -> AssemblySummary {
    let base_dof_count = (model.loads.len() * 3).max(3);
    let dof_count = if let Some(prep) = prep_context {
        let prep_dof = ((prep.prepared_node_count as f64) * prep.topology_dof_multiplier)
            .round()
            .max(base_dof_count as f64) as usize;
        prep_dof.clamp(base_dof_count, base_dof_count.saturating_mul(6).max(3))
    } else {
        base_dof_count
    };

    let avg_youngs_modulus = if model.materials.is_empty() {
        1.0e9
    } else {
        model
            .materials
            .iter()
            .map(|material| material.youngs_modulus_pa.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let avg_poisson_ratio = if model.materials.is_empty() {
        0.3
    } else {
        model
            .materials
            .iter()
            .map(|material| material.poisson_ratio.clamp(0.0, 0.49))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let stiffness_base = (avg_youngs_modulus / 2.0e3).max(1.0e5);

    let mut stiffness_diag = vec![0.0; dof_count];
    let mut stiffness_upper = vec![0.0; dof_count.saturating_sub(1)];
    let mut mass_diag = vec![0.0; dof_count];
    let mut damping_diag = vec![0.0; dof_count];
    for i in 0..dof_count {
        let factor = 1.0 + (i as f64) * 0.05;
        stiffness_diag[i] = stiffness_base * factor;
        mass_diag[i] = 1.0 + (i as f64) * 0.01;
        damping_diag[i] = 0.05 * factor;
    }

    let mut rhs = vec![0.0; dof_count];
    let load_base_index = |i: usize, prep: Option<FeaPrepContext>| -> usize {
        if let Some(prep) = prep {
            let stride = (1 + prep.mapped_load_count.max(1)).min(dof_count.max(1));
            let offset = (prep.layout_seed as usize) % dof_count.max(1);
            (offset + i.saturating_mul(stride)) % dof_count.max(1)
        } else {
            (i * 3) % dof_count
        }
    };
    for (i, load) in model.loads.iter().enumerate() {
        let base = load_base_index(i, prep_context);
        match &load.kind {
            runmat_analysis_core::LoadKind::Force { fx, fy, fz } => {
                rhs[base] += *fx;
                if base + 1 < dof_count {
                    rhs[base + 1] += *fy;
                }
                if base + 2 < dof_count {
                    rhs[base + 2] += *fz;
                }
            }
            runmat_analysis_core::LoadKind::Pressure { magnitude_pa } => {
                rhs[base] += magnitude_pa * 1.0e-3;
                if base + 1 < dof_count {
                    rhs[base + 1] -= magnitude_pa * 1.0e-3;
                }
            }
            runmat_analysis_core::LoadKind::BodyForce { gx, gy, gz } => {
                rhs[base] += *gx;
                if base + 1 < dof_count {
                    rhs[base + 1] += *gy;
                }
                if base + 2 < dof_count {
                    rhs[base + 2] += *gz;
                }
            }
        }
    }

    let constrained_dof_count = model.boundary_conditions.len().min(dof_count);
    let mut constrained = vec![false; dof_count];
    let constraint_offset = prep_context
        .map(|prep| (prep.layout_seed as usize) % dof_count.max(1))
        .unwrap_or(0);
    for idx in 0..constrained_dof_count {
        let dof = (constraint_offset + idx * 2) % dof_count.max(1);
        constrained[dof] = true;
        rhs[dof] = 0.0;
    }

    let mut prep_load_bonus = 0usize;
    let mut prep_assembly = None;
    let mut prep_operator_topology = None;
    let mut prep_region_topology = None;
    let mut prep_element_assembly = None;
    let mut prep_element_connectivity = None;
    let mut prep_graph_assembly = None;
    let mut prep_calibration = None;
    let mut prep_acceptance = None;
    let mut thermo_mechanical = None;
    let mut topology_stiffness_scale = 1.0;
    let mut topology_mass_scale = 1.0;
    let mut topology_damping_scale = 1.0;
    let mut topology_rhs_scale = 1.0;
    let mut topology_coupling_scale = 1.0;
    let mut topology_coupling_anisotropy = 1.0;
    let mut region_block_sizes = Vec::new();
    let mut region_boundary_positions = Vec::new();
    let mut region_coupling_weight = 1.0;
    if let Some(prep) = prep_context {
        let mesh_scale = 1.0 + (prep.prepared_mesh_count.min(32) as f64) * 0.01;
        let density = if prep.prepared_node_count == 0 {
            1.0
        } else {
            prep.prepared_element_count as f64 / prep.prepared_node_count as f64
        };
        let quality_scale = (prep.min_scaled_jacobian.clamp(0.5, 1.0)
            / prep.mean_aspect_ratio.max(1.0).min(4.0))
        .clamp(0.25, 1.0);
        let stiffness_scale =
            (mesh_scale * (1.0 + 0.05 * density.min(2.0)) * quality_scale).clamp(0.5, 1.5);
        let rhs_scale = (1.0 / quality_scale).clamp(1.0, 2.0);
        let region_span_norm = (prep.topology_region_span_mean / 12.0).clamp(0.0, 1.0);
        topology_stiffness_scale = (1.0
            + 0.12 * prep.topology_volume_core_ratio
            + 0.06 * prep.topology_surface_patch_ratio
            + 0.05 * prep.mapped_region_participation_ratio
            - 0.08 * prep.topology_mixed_family_ratio)
            .clamp(0.85, 1.25);
        topology_mass_scale =
            (1.0 + 0.05 * prep.topology_surface_patch_ratio + 0.04 * region_span_norm
                - 0.04 * prep.topology_mixed_family_ratio)
                .clamp(0.9, 1.2);
        topology_damping_scale = (1.0
            + 0.03 * prep.topology_volume_core_ratio
            + 0.05 * prep.topology_mixed_family_ratio
            + 0.02 * prep.mapped_region_participation_ratio)
            .clamp(0.9, 1.2);
        topology_rhs_scale = (1.0
            + 0.04 * prep.topology_surface_patch_ratio
            + 0.03 * prep.topology_region_span_mean.clamp(1.0, 24.0) / 24.0)
            .clamp(1.0, 1.18);
        topology_coupling_scale = (1.0
            + 0.12 * prep.topology_volume_core_ratio
            + 0.08 * prep.mapped_region_participation_ratio
            - 0.06 * prep.topology_mixed_family_ratio)
            .clamp(0.8, 1.3);
        topology_coupling_anisotropy =
            (1.0 - 0.18 * prep.topology_mixed_family_ratio).clamp(0.78, 1.0);
        region_coupling_weight = (1.0
            + 0.10 * prep.mapped_region_participation_ratio
            + 0.05 * prep.topology_region_mesh_variance.clamp(0.0, 6.0) / 6.0
            - 0.04 * prep.topology_mixed_family_ratio)
            .clamp(0.85, 1.2);
        let region_block_count = prep.topology_region_block_count.clamp(1, dof_count.max(1));
        region_block_sizes = build_region_block_sizes(
            dof_count,
            region_block_count,
            prep.layout_seed,
            prep.topology_region_mesh_mean,
            prep.topology_region_mesh_variance,
            prep.mapped_region_participation_ratio,
        );
        let region_block_offsets = block_offsets(&region_block_sizes);
        region_boundary_positions = region_block_offsets
            .iter()
            .skip(1)
            .map(|offset| offset.saturating_sub(1))
            .filter(|index| *index < dof_count.saturating_sub(1))
            .collect::<Vec<_>>();

        for (block_index, offset) in region_block_offsets.iter().enumerate() {
            let block_size = region_block_sizes[block_index];
            let block_end = offset.saturating_add(block_size).min(dof_count);
            let block_bias = block_bias(prep.layout_seed, block_index)
                * (0.04 + 0.02 * prep.topology_region_mesh_mean.clamp(1.0, 6.0) / 6.0);
            let stiffness_block_scale = (1.0 + block_bias).clamp(0.9, 1.1);
            let mass_block_scale = (1.0 + 0.6 * block_bias).clamp(0.92, 1.08);
            let damping_block_scale = (1.0 + 0.8 * block_bias).clamp(0.9, 1.1);
            for idx in *offset..block_end {
                stiffness_diag[idx] *= stiffness_block_scale;
                mass_diag[idx] *= mass_block_scale;
                damping_diag[idx] *= damping_block_scale;
            }
        }

        for value in &mut stiffness_diag {
            *value *= stiffness_scale * topology_stiffness_scale;
        }
        for value in &mut mass_diag {
            *value *= (1.0 + 0.02 * density.min(2.0)).clamp(1.0, 1.04) * topology_mass_scale;
        }
        for value in &mut damping_diag {
            *value *= (1.0 + 0.03 * prep.mean_aspect_ratio.min(3.0)).clamp(1.0, 1.09)
                * topology_damping_scale;
        }
        for value in &mut rhs {
            *value *= rhs_scale * topology_rhs_scale;
        }
        prep_element_assembly = Some(apply_prep_native_element_assembly(
            prep,
            dof_count,
            &constrained,
            &mut stiffness_diag,
            &mut mass_diag,
            &mut damping_diag,
            &mut rhs,
        ));
        prep_load_bonus = prep
            .mapped_region_count
            .saturating_add(prep.inverted_element_count.min(8));

        prep_assembly = Some(PrepAssemblySummary {
            active_region_count: prep.mapped_region_count,
            mapped_load_count: prep.mapped_load_count,
            mapped_bc_count: prep.mapped_bc_count,
            mapped_load_ratio: if model.loads.is_empty() {
                0.0
            } else {
                prep.mapped_load_count as f64 / model.loads.len() as f64
            },
            constrained_prep_ratio: if model.boundary_conditions.is_empty() {
                0.0
            } else {
                prep.mapped_bc_count as f64 / model.boundary_conditions.len() as f64
            },
            layout_seed: prep.layout_seed,
        });
    }

    let bandwidth_stride = prep_context
        .map(|prep| prep.topology_bandwidth_proxy.max(1) as usize)
        .unwrap_or(1);
    for i in 0..stiffness_upper.len() {
        let at_region_boundary = region_boundary_positions.binary_search(&i).is_ok();
        let local_coupling_scale = if i % 2 == 0 {
            topology_coupling_scale
        } else {
            topology_coupling_scale * topology_coupling_anisotropy
        };
        let region_boundary_scale = if at_region_boundary {
            region_coupling_weight * 0.75
        } else {
            region_coupling_weight
        };
        let coupling = 0.05
            * stiffness_diag[i].min(stiffness_diag[i + 1])
            * local_coupling_scale
            * region_boundary_scale;
        let in_sparse_band = bandwidth_stride <= 1 || (i % bandwidth_stride != 0);
        stiffness_upper[i] = if constrained[i] || constrained[i + 1] || !in_sparse_band {
            0.0
        } else {
            coupling
        };
    }

    if let Some(prep) = prep_context {
        if let Some(element_summary) = prep_element_assembly.as_ref() {
            let (connectivity_summary, graph_summary) = apply_prep_element_connectivity_scatter(
                prep,
                &constrained,
                &mut stiffness_upper,
                &mut mass_diag,
                &mut damping_diag,
                element_summary,
            );
            prep_element_connectivity = Some(connectivity_summary);
            prep_graph_assembly = Some(graph_summary);
        }
        if let Some(calibration) = apply_prep_calibration(
            prep,
            avg_youngs_modulus,
            prep_graph_assembly.as_ref(),
            &mut stiffness_diag,
            &mut mass_diag,
            &mut damping_diag,
            &mut rhs,
        ) {
            let acceptance = evaluate_prep_acceptance(
                prep,
                &calibration,
                prep_graph_assembly.as_ref(),
                &stiffness_diag,
            );
            prep_calibration = Some(calibration);
            prep_acceptance = Some(acceptance);
        }
        let coupling_nonzero_ratio = if stiffness_upper.is_empty() {
            0.0
        } else {
            stiffness_upper
                .iter()
                .filter(|value| value.abs() > 0.0)
                .count() as f64
                / stiffness_upper.len() as f64
        };
        let max_stiffness = stiffness_diag.iter().copied().fold(0.0_f64, f64::max);
        let min_stiffness = stiffness_diag
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .fold(f64::INFINITY, f64::min);
        let stiffness_spread_ratio = if min_stiffness.is_finite() && min_stiffness > 0.0 {
            max_stiffness / min_stiffness
        } else {
            0.0
        };
        prep_operator_topology = Some(PrepOperatorTopologySummary {
            stiffness_scale: topology_stiffness_scale,
            mass_scale: topology_mass_scale,
            damping_scale: topology_damping_scale,
            rhs_scale: topology_rhs_scale,
            coupling_nonzero_ratio,
            stiffness_spread_ratio,
            topology_fingerprint: topology_fingerprint(
                prep,
                topology_stiffness_scale,
                topology_mass_scale,
                topology_damping_scale,
                topology_rhs_scale,
                coupling_nonzero_ratio,
                stiffness_spread_ratio,
            ),
        });

        if !region_block_sizes.is_empty() {
            let block_size_min = region_block_sizes.iter().copied().min().unwrap_or(0);
            let block_size_max = region_block_sizes.iter().copied().max().unwrap_or(0);
            let block_size_mean =
                region_block_sizes.iter().sum::<usize>() as f64 / region_block_sizes.len() as f64;
            let inter_block_edge_count = region_boundary_positions.len();
            prep_region_topology = Some(PrepRegionTopologySummary {
                region_block_count: region_block_sizes.len(),
                inter_block_edge_count,
                coupling_nonzero_ratio,
                block_size_min,
                block_size_max,
                block_size_mean,
                region_topology_fingerprint: region_topology_fingerprint(
                    prep,
                    &region_block_sizes,
                    inter_block_edge_count,
                    coupling_nonzero_ratio,
                ),
            });
        }
    }

    if let Some(context) = thermo_mechanical_context {
        if context.enabled {
            let thermal_strain_scale = (context.thermal_expansion_coefficient
                * context.applied_temperature_delta_k.abs())
            .clamp(0.0, 0.05);
            let thermal_load_scale = (context.applied_temperature_delta_k / 50.0).clamp(-2.0, 2.0);
            let constitutive_temperature_factor =
                (context.applied_temperature_delta_k / 600.0).clamp(-0.25, 0.25);
            let constitutive_poisson_coupling =
                (0.6 + avg_poisson_ratio.clamp(0.0, 0.49)).clamp(0.6, 1.2);
            let modulus_temperature_scale = (1.0
                - constitutive_temperature_factor * constitutive_poisson_coupling)
                .clamp(0.75, 1.15);
            let thermal_stiffening_scale = (1.0 + 0.35 * thermal_strain_scale).clamp(1.0, 1.06);
            let effective_modulus_scale =
                (modulus_temperature_scale * thermal_stiffening_scale).clamp(0.75, 1.2);
            for i in 0..dof_count {
                let thermal_bias = 1.0 + thermal_strain_scale * (1.0 + (i % 3) as f64 * 0.1);
                stiffness_diag[i] *= thermal_bias * effective_modulus_scale;
                if !constrained[i] {
                    rhs[i] += thermal_load_scale * (1.0 + (i % 5) as f64 * 0.05);
                }
            }
            for value in &mut stiffness_upper {
                *value *= effective_modulus_scale;
            }
            thermo_mechanical = Some(ThermoMechanicalAssemblySummary {
                enabled: true,
                reference_temperature_k: context.reference_temperature_k,
                applied_temperature_delta_k: context.applied_temperature_delta_k,
                thermal_expansion_coefficient: context.thermal_expansion_coefficient,
                thermal_strain_scale,
                thermal_load_scale,
                constitutive_temperature_factor,
                constitutive_poisson_coupling,
                effective_modulus_scale,
                coupling_fingerprint: thermo_mechanical_fingerprint(
                    context,
                    dof_count,
                    constitutive_temperature_factor,
                    constitutive_poisson_coupling,
                    effective_modulus_scale,
                ),
            });
        }
    }

    AssemblySummary {
        dof_count,
        constrained_dof_count,
        load_count: model.loads.len().saturating_add(prep_load_bonus),
        prep_assembly,
        prep_operator_topology,
        prep_region_topology,
        prep_element_assembly,
        prep_element_connectivity,
        prep_graph_assembly,
        prep_calibration,
        prep_acceptance,
        thermo_mechanical,
        operator: OperatorSystem {
            dof_count,
            constrained,
            stiffness_diag,
            stiffness_upper,
            mass_diag,
            damping_diag,
            rhs,
        },
    }
}

fn topology_fingerprint(
    prep: FeaPrepContext,
    stiffness_scale: f64,
    mass_scale: f64,
    damping_scale: f64,
    rhs_scale: f64,
    coupling_nonzero_ratio: f64,
    stiffness_spread_ratio: f64,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        stiffness_scale.to_bits(),
        mass_scale.to_bits(),
        damping_scale.to_bits(),
        rhs_scale.to_bits(),
        coupling_nonzero_ratio.to_bits(),
        stiffness_spread_ratio.to_bits(),
        prep.topology_surface_patch_ratio.to_bits(),
        prep.topology_volume_core_ratio.to_bits(),
        prep.topology_mixed_family_ratio.to_bits(),
        prep.topology_region_span_mean.to_bits(),
        prep.mapped_region_participation_ratio.to_bits(),
        prep.topology_dof_multiplier.to_bits(),
        prep.topology_bandwidth_proxy as u64,
        prep.layout_seed,
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

fn apply_prep_native_element_assembly(
    prep: FeaPrepContext,
    dof_count: usize,
    constrained: &[bool],
    stiffness_diag: &mut [f64],
    mass_diag: &mut [f64],
    damping_diag: &mut [f64],
    rhs: &mut [f64],
) -> PrepElementAssemblySummary {
    let element_count = prep
        .prepared_element_count
        .max(prep.prepared_mesh_count)
        .max(1);
    let triangle_count = ((element_count as f64)
        * prep.topology_triangle_family_ratio.clamp(0.0, 1.0))
    .round() as usize;
    let quad_count =
        ((element_count as f64) * prep.topology_quad_family_ratio.clamp(0.0, 1.0)).round() as usize;
    let tet_count =
        ((element_count as f64) * prep.topology_tet_family_ratio.clamp(0.0, 1.0)).round() as usize;
    let mut hex_count =
        ((element_count as f64) * prep.topology_hex_family_ratio.clamp(0.0, 1.0)).round() as usize;
    let assigned = triangle_count + quad_count + tet_count + hex_count;
    if assigned > element_count {
        let overflow = assigned - element_count;
        hex_count = hex_count.saturating_sub(overflow);
    }
    let mixed_count =
        element_count.saturating_sub(triangle_count + quad_count + tet_count + hex_count);

    let mut touched_diag = vec![false; dof_count];
    let mut touched_rhs = vec![false; dof_count];
    let mut element_cursor = 0usize;
    let stride = (prep.topology_bandwidth_proxy.max(1) as usize + 1)
        .saturating_add(prep.topology_region_block_count.saturating_sub(1));
    let span_scale = 1.0 + 0.08 * prep.topology_region_span_mean.clamp(1.0, 24.0) / 24.0;

    let mut apply_family = |count: usize, stiffness_factor: f64, mass_factor: f64| {
        for _ in 0..count {
            let base = ((prep.layout_seed as usize)
                .wrapping_add(element_cursor.saturating_mul(stride.max(1))))
                % dof_count.max(1);
            let wave = 1.0 + ((element_cursor % 17) as f64) / 80.0;
            let assembly_scale = stiffness_factor * span_scale * wave;
            let mass_scale = mass_factor * (1.0 + 0.04 * prep.topology_surface_patch_ratio);
            let damping_scale = (0.9 + 0.2 * prep.mapped_region_participation_ratio)
                * (1.0 + 0.05 * prep.topology_mixed_family_ratio);

            stiffness_diag[base] += 7.5e4 * assembly_scale;
            mass_diag[base] += 0.25 * mass_scale;
            damping_diag[base] += 0.012 * damping_scale;
            touched_diag[base] = true;

            if !constrained[base] {
                rhs[base] += 0.5 * assembly_scale;
                touched_rhs[base] = true;
            }
            if base + 1 < dof_count {
                stiffness_diag[base + 1] += 2.0e4 * assembly_scale;
                mass_diag[base + 1] += 0.08 * mass_scale;
                damping_diag[base + 1] += 0.004 * damping_scale;
                touched_diag[base + 1] = true;
            }
            element_cursor = element_cursor.saturating_add(1);
        }
    };

    apply_family(triangle_count, 0.92, 0.95);
    apply_family(quad_count, 1.00, 1.00);
    apply_family(tet_count, 1.07, 1.05);
    apply_family(hex_count, 1.15, 1.12);
    apply_family(mixed_count, 0.98, 1.02);

    let scatter_nnz_count = touched_diag.iter().filter(|&&hit| hit).count()
        + touched_rhs.iter().filter(|&&hit| hit).count();
    PrepElementAssemblySummary {
        assembled_element_count: element_count,
        triangle_element_count: triangle_count,
        quad_element_count: quad_count,
        tet_element_count: tet_count,
        hex_element_count: hex_count,
        mixed_element_count: mixed_count,
        scatter_nnz_count,
        assembly_fingerprint: element_assembly_fingerprint(
            prep,
            element_count,
            triangle_count,
            quad_count,
            tet_count,
            hex_count,
            mixed_count,
            scatter_nnz_count,
        ),
    }
}

fn apply_prep_element_connectivity_scatter(
    prep: FeaPrepContext,
    constrained: &[bool],
    stiffness_upper: &mut [f64],
    mass_diag: &mut [f64],
    damping_diag: &mut [f64],
    element_summary: &PrepElementAssemblySummary,
) -> (PrepElementConnectivitySummary, PrepGraphAssemblySummary) {
    if stiffness_upper.is_empty() {
        let connectivity_summary = PrepElementConnectivitySummary {
            assembled_element_count: element_summary.assembled_element_count,
            stiffness_offdiag_nnz_count: 0,
            mass_offdiag_proxy_nnz_count: 0,
            damping_offdiag_proxy_nnz_count: 0,
            triangle_contrib_share: 0.0,
            quad_contrib_share: 0.0,
            tet_contrib_share: 0.0,
            hex_contrib_share: 0.0,
            mixed_contrib_share: 0.0,
            mean_connectivity_hop: 0.0,
            connectivity_fingerprint: element_connectivity_fingerprint(
                prep,
                element_summary,
                0,
                0,
                0,
                0.0,
                [0.0; 5],
                0,
            ),
        };
        let graph_summary = PrepGraphAssemblySummary {
            node_count: constrained.len(),
            edge_count: 0,
            degree_min: 0,
            degree_max: 0,
            degree_mean: 0.0,
            degree_p95: 0.0,
            fill_ratio: 0.0,
            connected_component_count: constrained.len().max(1),
            ordering_bandwidth_before: 0,
            ordering_bandwidth_after: 0,
            ordering_reduction_ratio: 0.0,
            ordering_fingerprint: 0,
            recommend_ilu0: false,
            graph_fingerprint: graph_fingerprint(prep, 0, constrained.len().max(1), 0.0, 0.0),
        };
        return (connectivity_summary, graph_summary);
    }

    let node_count = constrained.len().max(1);
    let edges = build_prep_graph_edges(prep, node_count, element_summary);
    let (degree_min, degree_max, degree_mean, degree_p95, component_count) =
        graph_degree_stats(node_count, &edges);
    let max_edges = node_count.saturating_mul(node_count.saturating_sub(1)) / 2;
    let fill_ratio = if max_edges == 0 {
        0.0
    } else {
        edges.len() as f64 / max_edges as f64
    };

    let mut touched_stiffness = vec![false; stiffness_upper.len()];
    let mut touched_mass = vec![false; mass_diag.len()];
    let mut touched_damping = vec![false; damping_diag.len()];
    let mut family_contrib = [0.0_f64; 5];
    let mut hops = Vec::new();
    let family_stiffness = [0.85_f64, 0.95_f64, 1.05_f64, 1.15_f64, 0.9_f64];
    let family_mass = [0.10_f64, 0.11_f64, 0.12_f64, 0.14_f64, 0.105_f64];
    let family_damping = [0.004_f64, 0.0045_f64, 0.005_f64, 0.0055_f64, 0.0042_f64];
    let region_bias = 1.0 + 0.05 * prep.topology_region_mesh_mean.clamp(1.0, 8.0) / 8.0;

    for (edge_cursor, (left, right, family_index)) in edges.iter().copied().enumerate() {
        if constrained[left] || constrained[right] {
            continue;
        }
        let hop = right.abs_diff(left).max(1);
        hops.push(hop as f64);
        let wave = 1.0 + ((edge_cursor % 23) as f64) / 100.0;
        let stiffness_add = 0.012
            * family_stiffness[family_index]
            * region_bias
            * wave
            * prep.topology_dof_multiplier.clamp(1.0, 4.0)
            / hop as f64;
        let lo = left.min(right);
        let hi = left.max(right);
        for band in lo..hi {
            if band >= stiffness_upper.len() {
                continue;
            }
            let attenuation = 1.0 / (1.0 + (band - lo) as f64);
            stiffness_upper[band] +=
                stiffness_add * attenuation * (stiffness_upper[band].abs() + 1.0);
            touched_stiffness[band] = true;
        }
        family_contrib[family_index] += stiffness_add.abs();

        let mass_add = family_mass[family_index] * wave;
        mass_diag[left] += mass_add;
        mass_diag[right] += mass_add * 0.8;
        touched_mass[left] = true;
        touched_mass[right] = true;

        let damping_add =
            family_damping[family_index] * (1.0 + 0.5 * prep.topology_mixed_family_ratio);
        damping_diag[left] += damping_add;
        damping_diag[right] += damping_add * 0.85;
        touched_damping[left] = true;
        touched_damping[right] = true;
    }

    let stiffness_offdiag_nnz_count = touched_stiffness.iter().filter(|&&hit| hit).count();
    let mass_offdiag_proxy_nnz_count = touched_mass.iter().filter(|&&hit| hit).count();
    let damping_offdiag_proxy_nnz_count = touched_damping.iter().filter(|&&hit| hit).count();
    let total_contrib = family_contrib.iter().sum::<f64>().max(1.0e-12);
    let shares = [
        family_contrib[0] / total_contrib,
        family_contrib[1] / total_contrib,
        family_contrib[2] / total_contrib,
        family_contrib[3] / total_contrib,
        family_contrib[4] / total_contrib,
    ];
    let mean_connectivity_hop = if hops.is_empty() {
        0.0
    } else {
        hops.iter().sum::<f64>() / hops.len() as f64
    };

    let graph_fingerprint_value =
        graph_fingerprint(prep, edges.len(), component_count, degree_mean, degree_p95);
    let ordering_permutation = graph_ordering_permutation(node_count, &edges);
    let ordering_bandwidth_before = graph_bandwidth(&edges, None);
    let ordering_bandwidth_after = graph_bandwidth(&edges, Some(&ordering_permutation));
    let ordering_reduction_ratio = if ordering_bandwidth_before == 0 {
        0.0
    } else {
        1.0 - (ordering_bandwidth_after as f64 / ordering_bandwidth_before as f64)
    };
    let ordering_fingerprint = graph_ordering_fingerprint(prep, &ordering_permutation);
    let recommend_ilu0 = degree_p95 >= 4.0 || fill_ratio >= 0.02 || component_count <= 3;
    let graph_summary = PrepGraphAssemblySummary {
        node_count,
        edge_count: edges.len(),
        degree_min,
        degree_max,
        degree_mean,
        degree_p95,
        fill_ratio,
        connected_component_count: component_count,
        ordering_bandwidth_before,
        ordering_bandwidth_after,
        ordering_reduction_ratio,
        ordering_fingerprint,
        recommend_ilu0,
        graph_fingerprint: graph_fingerprint_value,
    };

    let connectivity_summary = PrepElementConnectivitySummary {
        assembled_element_count: element_summary.assembled_element_count,
        stiffness_offdiag_nnz_count,
        mass_offdiag_proxy_nnz_count,
        damping_offdiag_proxy_nnz_count,
        triangle_contrib_share: shares[0],
        quad_contrib_share: shares[1],
        tet_contrib_share: shares[2],
        hex_contrib_share: shares[3],
        mixed_contrib_share: shares[4],
        mean_connectivity_hop,
        connectivity_fingerprint: element_connectivity_fingerprint(
            prep,
            element_summary,
            stiffness_offdiag_nnz_count,
            mass_offdiag_proxy_nnz_count,
            damping_offdiag_proxy_nnz_count,
            mean_connectivity_hop,
            shares,
            graph_fingerprint_value,
        ),
    };

    (connectivity_summary, graph_summary)
}

fn apply_prep_calibration(
    prep: FeaPrepContext,
    avg_youngs_modulus: f64,
    graph_summary: Option<&PrepGraphAssemblySummary>,
    stiffness_diag: &mut [f64],
    mass_diag: &mut [f64],
    damping_diag: &mut [f64],
    rhs: &mut [f64],
) -> Option<PrepCalibrationSummary> {
    if stiffness_diag.is_empty() {
        return None;
    }
    let profile = select_calibration_profile(prep, avg_youngs_modulus, graph_summary);
    let (profile_gain, profile_name) = match profile {
        CalibrationProfile::Fast => (0.92, "fast"),
        CalibrationProfile::Balanced => (1.0, "balanced"),
        CalibrationProfile::Conservative => (1.08, "conservative"),
    };

    let triangle_weight = (0.95 + 0.08 * prep.topology_triangle_family_ratio) * profile_gain;
    let quad_weight = (1.0 + 0.06 * prep.topology_quad_family_ratio) * profile_gain;
    let tet_weight = (1.04 + 0.10 * prep.topology_tet_family_ratio) * profile_gain;
    let hex_weight = (1.08 + 0.12 * prep.topology_hex_family_ratio) * profile_gain;
    let mixed_weight = (0.9 + 0.05 * prep.topology_mixed_family_ratio) * profile_gain;

    let stiffness_calibration_scale = (triangle_weight * prep.topology_triangle_family_ratio
        + quad_weight * prep.topology_quad_family_ratio
        + tet_weight * prep.topology_tet_family_ratio
        + hex_weight * prep.topology_hex_family_ratio
        + mixed_weight * prep.topology_mixed_family_ratio.max(0.01))
    .clamp(0.8, 1.3);
    let mass_calibration_scale = (0.96
        + 0.03 * prep.topology_surface_patch_ratio
        + 0.04 * prep.topology_region_mesh_mean.clamp(1.0, 6.0) / 6.0)
        .clamp(0.9, 1.2)
        * profile_gain;
    let damping_calibration_scale = (0.94
        + 0.05 * prep.topology_mixed_family_ratio
        + 0.03 * prep.mapped_region_participation_ratio)
        .clamp(0.9, 1.2)
        * profile_gain;

    for value in stiffness_diag.iter_mut() {
        *value *= stiffness_calibration_scale;
    }
    for value in mass_diag.iter_mut() {
        *value *= mass_calibration_scale;
    }
    for value in damping_diag.iter_mut() {
        *value *= damping_calibration_scale;
    }
    for value in rhs.iter_mut() {
        *value *= (2.0 - stiffness_calibration_scale).clamp(0.8, 1.2);
    }

    Some(PrepCalibrationSummary {
        profile: profile_name.to_string(),
        triangle_weight,
        quad_weight,
        tet_weight,
        hex_weight,
        mixed_weight,
        stiffness_calibration_scale,
        mass_calibration_scale,
        damping_calibration_scale,
        calibration_fingerprint: calibration_fingerprint(
            prep,
            profile_name,
            stiffness_calibration_scale,
            mass_calibration_scale,
            damping_calibration_scale,
        ),
    })
}

fn evaluate_prep_acceptance(
    prep: FeaPrepContext,
    calibration: &PrepCalibrationSummary,
    graph_summary: Option<&PrepGraphAssemblySummary>,
    stiffness_diag: &[f64],
) -> PrepAcceptanceSummary {
    let bounded_displacement_scale = (0.8..=1.3).contains(&calibration.stiffness_calibration_scale);
    let bounded_stress_scale = (0.9..=1.25).contains(&calibration.damping_calibration_scale);
    let bounded_connectivity_fill = graph_summary
        .map(|graph| graph.fill_ratio <= 0.25 && graph.connected_component_count <= 64)
        .unwrap_or(true);
    let stiffness_max = stiffness_diag.iter().copied().fold(0.0_f64, f64::max);
    let stiffness_min = stiffness_diag
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    let spread = if stiffness_min.is_finite() && stiffness_min > 0.0 {
        stiffness_max / stiffness_min
    } else {
        0.0
    };
    let spread_penalty = (spread / 100.0).clamp(0.0, 1.0);
    let mut acceptance_score = 1.0 - spread_penalty;
    if !bounded_displacement_scale {
        acceptance_score -= 0.2;
    }
    if !bounded_stress_scale {
        acceptance_score -= 0.2;
    }
    if !bounded_connectivity_fill {
        acceptance_score -= 0.3;
    }
    acceptance_score = acceptance_score.clamp(0.0, 1.0);
    let accepted = bounded_displacement_scale
        && bounded_stress_scale
        && bounded_connectivity_fill
        && acceptance_score >= 0.4
        && prep.min_scaled_jacobian >= 0.45;

    PrepAcceptanceSummary {
        profile: calibration.profile.clone(),
        accepted,
        bounded_displacement_scale,
        bounded_stress_scale,
        bounded_connectivity_fill,
        acceptance_score,
        acceptance_fingerprint: acceptance_fingerprint(
            prep,
            &calibration.profile,
            accepted,
            acceptance_score,
            spread,
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CalibrationProfile {
    Fast,
    Balanced,
    Conservative,
}

fn select_calibration_profile(
    prep: FeaPrepContext,
    avg_youngs_modulus: f64,
    graph_summary: Option<&PrepGraphAssemblySummary>,
) -> CalibrationProfile {
    if let Some(profile) = prep.calibration_profile_override {
        return match profile {
            FeaPrepCalibrationProfile::Fast => CalibrationProfile::Fast,
            FeaPrepCalibrationProfile::Balanced => CalibrationProfile::Balanced,
            FeaPrepCalibrationProfile::Conservative => CalibrationProfile::Conservative,
        };
    }
    let stiffness_regime = avg_youngs_modulus;
    let ordering_gain = graph_summary
        .map(|graph| graph.ordering_reduction_ratio)
        .unwrap_or(0.0);
    if prep.min_scaled_jacobian < 0.65 || prep.topology_mixed_family_ratio > 0.25 {
        CalibrationProfile::Conservative
    } else if stiffness_regime < 5.0e10 || ordering_gain > 0.2 {
        CalibrationProfile::Fast
    } else {
        CalibrationProfile::Balanced
    }
}

fn build_prep_graph_edges(
    prep: FeaPrepContext,
    node_count: usize,
    element_summary: &PrepElementAssemblySummary,
) -> Vec<(usize, usize, usize)> {
    use std::collections::BTreeSet;

    let family_counts = [
        element_summary.triangle_element_count,
        element_summary.quad_element_count,
        element_summary.tet_element_count,
        element_summary.hex_element_count,
        element_summary.mixed_element_count,
    ];
    let family_valence = [2usize, 3, 4, 5, 3];
    let stride = (prep.topology_bandwidth_proxy.max(1) as usize)
        .saturating_add(prep.topology_region_block_count.max(1));
    let max_hop = prep
        .topology_region_span_mean
        .round()
        .clamp(1.0, node_count as f64) as usize;
    let mut edges = BTreeSet::new();
    let mut cursor = 0usize;
    for family_index in 0..family_counts.len() {
        for _ in 0..family_counts[family_index] {
            let base = ((prep.layout_seed as usize)
                .wrapping_add(family_index.saturating_mul(31))
                .wrapping_add(cursor.saturating_mul(stride.max(1))))
                % node_count.max(1);
            for k in 0..family_valence[family_index] {
                let hop = 1
                    + ((prep.layout_seed as usize)
                        .wrapping_add(k)
                        .wrapping_add(cursor)
                        .wrapping_add(family_index.saturating_mul(7))
                        % max_hop.max(1));
                let target = (base + hop) % node_count.max(1);
                if base == target {
                    continue;
                }
                let lo = base.min(target);
                let hi = base.max(target);
                edges.insert((lo, hi, family_index));
            }
            cursor = cursor.saturating_add(1);
        }
    }
    edges.into_iter().collect()
}

fn graph_degree_stats(
    node_count: usize,
    edges: &[(usize, usize, usize)],
) -> (usize, usize, f64, f64, usize) {
    if node_count == 0 {
        return (0, 0, 0.0, 0.0, 0);
    }
    let mut degree = vec![0usize; node_count];
    let mut parent = (0..node_count).collect::<Vec<_>>();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            let root = find(parent, parent[x]);
            parent[x] = root;
        }
        parent[x]
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    for (a, b, _) in edges {
        degree[*a] = degree[*a].saturating_add(1);
        degree[*b] = degree[*b].saturating_add(1);
        union(&mut parent, *a, *b);
    }
    let degree_min = degree.iter().copied().min().unwrap_or(0);
    let degree_max = degree.iter().copied().max().unwrap_or(0);
    let degree_mean = degree.iter().sum::<usize>() as f64 / degree.len() as f64;
    let mut sorted_degree = degree.iter().map(|d| *d as f64).collect::<Vec<_>>();
    sorted_degree.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let degree_p95 = if sorted_degree.is_empty() {
        0.0
    } else {
        let index = ((sorted_degree.len() - 1) as f64 * 0.95).round() as usize;
        sorted_degree[index]
    };

    let mut roots = std::collections::BTreeSet::new();
    for idx in 0..node_count {
        roots.insert(find(&mut parent, idx));
    }
    (degree_min, degree_max, degree_mean, degree_p95, roots.len())
}

fn graph_ordering_permutation(node_count: usize, edges: &[(usize, usize, usize)]) -> Vec<usize> {
    if node_count == 0 {
        return Vec::new();
    }
    let mut degree = vec![0usize; node_count];
    for (a, b, _) in edges {
        degree[*a] = degree[*a].saturating_add(1);
        degree[*b] = degree[*b].saturating_add(1);
    }
    let mut order = (0..node_count).collect::<Vec<_>>();
    order.sort_by(|a, b| degree[*a].cmp(&degree[*b]).then_with(|| a.cmp(b)));
    let mut permutation = vec![0usize; node_count];
    for (new_idx, old_idx) in order.iter().copied().enumerate() {
        permutation[old_idx] = new_idx;
    }
    permutation
}

fn graph_bandwidth(edges: &[(usize, usize, usize)], permutation: Option<&[usize]>) -> usize {
    let mut max_bw = 0usize;
    for (a, b, _) in edges {
        let lhs = permutation.map(|perm| perm[*a]).unwrap_or(*a);
        let rhs = permutation.map(|perm| perm[*b]).unwrap_or(*b);
        max_bw = max_bw.max(lhs.abs_diff(rhs));
    }
    max_bw
}

fn graph_ordering_fingerprint(prep: FeaPrepContext, permutation: &[usize]) -> u64 {
    let mut hash = 1469598103934665603_u64;
    hash ^= prep.layout_seed;
    hash = hash.wrapping_mul(1099511628211_u64);
    for value in permutation.iter().take(256) {
        hash ^= *value as u64;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

fn build_region_block_sizes(
    dof_count: usize,
    block_count: usize,
    layout_seed: u64,
    region_mesh_mean: f64,
    region_mesh_variance: f64,
    mapped_region_participation_ratio: f64,
) -> Vec<usize> {
    let mut sizes = vec![dof_count / block_count; block_count];
    for size in &mut sizes {
        if *size == 0 {
            *size = 1;
        }
    }
    let assigned = sizes.iter().sum::<usize>();
    let mut remainder = dof_count.saturating_sub(assigned);
    let seed_bias = ((layout_seed % 13) as usize).max(1);
    let participation_bias =
        (mapped_region_participation_ratio.clamp(0.0, 1.0) * 7.0).round() as usize;
    let variance_bias = region_mesh_variance.clamp(0.0, 16.0).round() as usize;
    let stride = (seed_bias + participation_bias + variance_bias).max(1);
    let mut cursor = (region_mesh_mean.round() as usize + seed_bias) % block_count.max(1);
    while remainder > 0 {
        sizes[cursor % block_count] = sizes[cursor % block_count].saturating_add(1);
        cursor = cursor.saturating_add(stride);
        remainder -= 1;
    }
    sizes
}

fn block_offsets(sizes: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut current = 0usize;
    for size in sizes {
        offsets.push(current);
        current = current.saturating_add(*size);
    }
    offsets
}

fn block_bias(layout_seed: u64, block_index: usize) -> f64 {
    let mut hash = layout_seed ^ ((block_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    let normalized = (hash % 1000) as f64 / 999.0;
    normalized * 2.0 - 1.0
}

fn region_topology_fingerprint(
    prep: FeaPrepContext,
    block_sizes: &[usize],
    inter_block_edge_count: usize,
    coupling_nonzero_ratio: f64,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        prep.layout_seed,
        prep.topology_region_block_count as u64,
        prep.topology_region_mesh_mean.to_bits(),
        prep.topology_region_mesh_variance.to_bits(),
        prep.topology_region_span_mean.to_bits(),
        prep.mapped_region_participation_ratio.to_bits(),
        inter_block_edge_count as u64,
        coupling_nonzero_ratio.to_bits(),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    for size in block_sizes {
        hash ^= *size as u64;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

fn element_assembly_fingerprint(
    prep: FeaPrepContext,
    element_count: usize,
    triangle_count: usize,
    quad_count: usize,
    tet_count: usize,
    hex_count: usize,
    mixed_count: usize,
    scatter_nnz_count: usize,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        prep.layout_seed,
        prep.prepared_element_count as u64,
        element_count as u64,
        triangle_count as u64,
        quad_count as u64,
        tet_count as u64,
        hex_count as u64,
        mixed_count as u64,
        scatter_nnz_count as u64,
        prep.topology_triangle_family_ratio.to_bits(),
        prep.topology_quad_family_ratio.to_bits(),
        prep.topology_tet_family_ratio.to_bits(),
        prep.topology_hex_family_ratio.to_bits(),
        prep.topology_mixed_family_ratio.to_bits(),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

fn element_connectivity_fingerprint(
    prep: FeaPrepContext,
    element_summary: &PrepElementAssemblySummary,
    stiffness_offdiag_nnz_count: usize,
    mass_offdiag_proxy_nnz_count: usize,
    damping_offdiag_proxy_nnz_count: usize,
    mean_connectivity_hop: f64,
    shares: [f64; 5],
    graph_fingerprint: u64,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        prep.layout_seed,
        element_summary.assembly_fingerprint,
        element_summary.assembled_element_count as u64,
        stiffness_offdiag_nnz_count as u64,
        mass_offdiag_proxy_nnz_count as u64,
        damping_offdiag_proxy_nnz_count as u64,
        mean_connectivity_hop.to_bits(),
        shares[0].to_bits(),
        shares[1].to_bits(),
        shares[2].to_bits(),
        shares[3].to_bits(),
        shares[4].to_bits(),
        prep.topology_bandwidth_proxy as u64,
        graph_fingerprint,
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

fn graph_fingerprint(
    prep: FeaPrepContext,
    edge_count: usize,
    connected_component_count: usize,
    degree_mean: f64,
    degree_p95: f64,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        prep.layout_seed,
        prep.topology_bandwidth_proxy as u64,
        prep.topology_region_block_count as u64,
        edge_count as u64,
        connected_component_count as u64,
        degree_mean.to_bits(),
        degree_p95.to_bits(),
        prep.topology_region_span_mean.to_bits(),
        prep.topology_mixed_family_ratio.to_bits(),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

fn calibration_fingerprint(
    prep: FeaPrepContext,
    profile: &str,
    stiffness_scale: f64,
    mass_scale: f64,
    damping_scale: f64,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for byte in profile.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    for value in [
        prep.layout_seed,
        prep.prepared_element_count as u64,
        stiffness_scale.to_bits(),
        mass_scale.to_bits(),
        damping_scale.to_bits(),
        prep.topology_triangle_family_ratio.to_bits(),
        prep.topology_quad_family_ratio.to_bits(),
        prep.topology_tet_family_ratio.to_bits(),
        prep.topology_hex_family_ratio.to_bits(),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

fn acceptance_fingerprint(
    prep: FeaPrepContext,
    profile: &str,
    accepted: bool,
    score: f64,
    spread: f64,
) -> u64 {
    let mut hash = calibration_fingerprint(
        prep,
        profile,
        score,
        spread,
        if accepted { 1.0 } else { 0.0 },
    );
    hash ^= accepted as u64;
    hash = hash.wrapping_mul(1099511628211_u64);
    hash
}

fn thermo_mechanical_fingerprint(
    context: FeaThermoMechanicalContext,
    dof_count: usize,
    constitutive_temperature_factor: f64,
    constitutive_poisson_coupling: f64,
    effective_modulus_scale: f64,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        dof_count as u64,
        context.reference_temperature_k.to_bits(),
        context.applied_temperature_delta_k.to_bits(),
        context.thermal_expansion_coefficient.to_bits(),
        constitutive_temperature_factor.to_bits(),
        constitutive_poisson_coupling.to_bits(),
        effective_modulus_scale.to_bits(),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}
