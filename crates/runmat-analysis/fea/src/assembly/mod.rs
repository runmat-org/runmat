use runmat_analysis_core::AnalysisModel;
use serde::{Deserialize, Serialize};

use crate::operator::OperatorSystem;
use crate::FeaPrepContext;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblySummary {
    pub dof_count: usize,
    pub constrained_dof_count: usize,
    pub load_count: usize,
    pub prep_assembly: Option<PrepAssemblySummary>,
    pub prep_operator_topology: Option<PrepOperatorTopologySummary>,
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

pub fn assemble_linear_system(
    model: &AnalysisModel,
    prep_context: Option<FeaPrepContext>,
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
    let mut topology_stiffness_scale = 1.0;
    let mut topology_mass_scale = 1.0;
    let mut topology_damping_scale = 1.0;
    let mut topology_rhs_scale = 1.0;
    let mut topology_coupling_scale = 1.0;
    let mut topology_coupling_anisotropy = 1.0;
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
        let local_coupling_scale = if i % 2 == 0 {
            topology_coupling_scale
        } else {
            topology_coupling_scale * topology_coupling_anisotropy
        };
        let coupling = 0.05 * stiffness_diag[i].min(stiffness_diag[i + 1]) * local_coupling_scale;
        let in_sparse_band = bandwidth_stride <= 1 || (i % bandwidth_stride != 0);
        stiffness_upper[i] = if constrained[i] || constrained[i + 1] || !in_sparse_band {
            0.0
        } else {
            coupling
        };
    }

    if let Some(prep) = prep_context {
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
    }

    AssemblySummary {
        dof_count,
        constrained_dof_count,
        load_count: model.loads.len().saturating_add(prep_load_bonus),
        prep_assembly,
        prep_operator_topology,
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
