pub mod dofs;
pub mod elements;
mod solid_matrix;
pub mod solver_solid;
mod solver_solid_boundary;

use std::{collections::BTreeMap, fmt};

use runmat_analysis_core::{
    AnalysisModel, BeamSectionModel, BoundaryConditionKind, LoadKind, ShellSectionModel,
    StructuralElementKind, StructuralModel,
};
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use self::elements::solid::SolidMaterial;
use self::{
    dofs::{StructuralDofKind, StructuralDofLayout, StructuralNodeDofSet},
    elements::beam::{
        global_stiffness_matrix as beam_global_stiffness_matrix, transformation_matrix,
        BeamElementGeometry, BeamMaterial, BeamSection, BeamTransform12, BEAM_ELEMENT_DOF_COUNT,
    },
    elements::shell::{
        global_stiffness_matrix as shell_global_stiffness_matrix, ShellElementGeometry,
        ShellMaterial, ShellSection, SHELL_ELEMENT_DOF_COUNT, SHELL_NODE_DOF_COUNT,
    },
    solver_solid::{
        assemble_solver_solid_stiffness_csr, solver_solid_topology, SolverSolidAssemblyError,
    },
    solver_solid_boundary::apply_solver_mesh_structural_regions,
};

use crate::operator::{CsrMatrix, OperatorSystem};
use crate::physics::coupling::thermo_mechanical;
use crate::{FeaElectroThermalContext, FeaThermoMechanicalContext};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblySummary {
    pub dof_count: usize,
    #[serde(default)]
    pub structural_node_count: usize,
    #[serde(default)]
    pub structural_translational_dof_count: usize,
    #[serde(default)]
    pub structural_rotational_dof_count: usize,
    #[serde(default)]
    pub structural_rotation_node_count: usize,
    #[serde(default)]
    pub structural_moment_load_count: usize,
    #[serde(default)]
    pub structural_direct_rotational_moment_load_count: usize,
    #[serde(default)]
    pub structural_wrench_lowering: Vec<WrenchLoweringSummary>,
    #[serde(default)]
    pub structural_rotational_constraint_count: usize,
    #[serde(default)]
    pub structural_beam_element_count: usize,
    #[serde(default)]
    pub structural_shell_element_count: usize,
    #[serde(default)]
    pub structural_solid_element_count: usize,
    #[serde(default)]
    pub structural_solid_recovery: Vec<SolidRecoveryElementSummary>,
    #[serde(default)]
    pub structural_dof_layout: StructuralDofLayout,
    #[serde(default)]
    pub structural_beam_recovery: Vec<BeamRecoveryElementSummary>,
    #[serde(default)]
    pub structural_shell_recovery: Vec<ShellRecoveryElementSummary>,
    pub constrained_dof_count: usize,
    pub load_count: usize,
    pub structural_material: StructuralMaterialSummary,
    pub thermo_mechanical: Option<ThermoMechanicalAssemblySummary>,
    pub electro_thermal: Option<ElectroThermalAssemblySummary>,
    pub operator: OperatorSystem,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearAssemblyError {
    SolidStiffness(SolverSolidAssemblyError),
    SolverMeshRegionMapping(SolverMeshRegionMappingError),
}

impl fmt::Display for LinearAssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinearAssemblyError::SolidStiffness(err) => {
                write!(f, "solid stiffness assembly failed: {err:?}")
            }
            LinearAssemblyError::SolverMeshRegionMapping(err) => {
                write!(f, "{err}")
            }
        }
    }
}

impl std::error::Error for LinearAssemblyError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverMeshRegionMappingError {
    UnmappedLoadRegion {
        load_id: String,
        region_id: String,
        load_kind: &'static str,
    },
    UnmappedBoundaryConditionRegion {
        bc_id: String,
        region_id: String,
        boundary_condition_kind: &'static str,
    },
}

impl fmt::Display for SolverMeshRegionMappingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverMeshRegionMappingError::UnmappedLoadRegion {
                load_id,
                region_id,
                load_kind,
            } => write!(
                f,
                "solver mesh load region did not resolve to persistent boundary provenance: load_id={load_id} region_id={region_id} load_kind={load_kind}"
            ),
            SolverMeshRegionMappingError::UnmappedBoundaryConditionRegion {
                bc_id,
                region_id,
                boundary_condition_kind,
            } => write!(
                f,
                "solver mesh boundary condition region did not resolve to persistent boundary provenance: bc_id={bc_id} region_id={region_id} boundary_condition_kind={boundary_condition_kind}"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralMaterialSummary {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    #[serde(default)]
    pub density_kg_per_m3: f64,
    pub lame_lambda_pa: f64,
    pub shear_modulus_pa: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub node_i_index: usize,
    pub node_j_index: usize,
    pub length_m: f64,
    pub section: BeamSection,
    pub material: BeamMaterial,
    pub transform_global_to_local: BeamTransform12,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub node_indices: [usize; 3],
    pub area_m2: f64,
    pub section: ShellSection,
    pub material: ShellMaterial,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolidRecoveryElementSummary {
    pub element_id: String,
    pub region_id: String,
    pub node_indices: [usize; 4],
    pub coordinates_m: [[f64; 3]; 4],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WrenchLoweringSummary {
    pub load_id: String,
    pub region_id: String,
    pub target_node_count: usize,
    pub applied_force: [f64; 3],
    pub applied_moment_at_point: [f64; 3],
    pub force_residual: [f64; 3],
    pub moment_residual: [f64; 3],
    pub moment_couple_applied: bool,
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
    pub constitutive_material_spread_ratio: f64,
    pub assignment_heterogeneity_index: f64,
    pub spatial_gradient_index: f64,
    pub spatial_coverage_ratio: f64,
    pub temporal_profile_variation: f64,
    pub region_delta_count: usize,
    pub coupling_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectroThermalAssemblySummary {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_voltage_v: f64,
    pub base_electrical_conductivity_s_per_m: f64,
    pub resistive_heating_coefficient: f64,
    pub joule_heating_scale: f64,
    pub conductivity_spread_ratio: f64,
    pub temporal_profile_variation: f64,
    pub region_scale_count: usize,
    pub coupling_fingerprint: u64,
}

pub fn assemble_linear_system(
    model: &AnalysisModel,
    solver_mesh: Option<SolverMeshArtifact>,
    thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    electro_thermal_context: Option<FeaElectroThermalContext>,
) -> AssemblySummary {
    assemble_linear_system_impl(
        model,
        solver_mesh,
        thermo_mechanical_context,
        electro_thermal_context,
        false,
    )
    .expect("non-strict assembly should build operator topology")
}

pub fn try_assemble_linear_system(
    model: &AnalysisModel,
    solver_mesh: Option<SolverMeshArtifact>,
    thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    electro_thermal_context: Option<FeaElectroThermalContext>,
) -> Result<AssemblySummary, LinearAssemblyError> {
    assemble_linear_system_impl(
        model,
        solver_mesh,
        thermo_mechanical_context,
        electro_thermal_context,
        true,
    )
}

fn assemble_linear_system_impl(
    model: &AnalysisModel,
    solver_mesh: Option<SolverMeshArtifact>,
    thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    electro_thermal_context: Option<FeaElectroThermalContext>,
    strict_solver_mesh_stiffness: bool,
) -> Result<AssemblySummary, LinearAssemblyError> {
    if solver_mesh.is_none() {
        if let Some(summary) = assemble_beam_system(model) {
            return Ok(summary);
        }
    }

    let base_dof_count = (model.loads.len() * 3).max(3);
    let solid_topology = solver_mesh
        .as_ref()
        .and_then(|mesh| solver_solid_topology(mesh, base_dof_count).ok());
    let dof_count = solid_topology
        .as_ref()
        .map(|topology| topology.dof_count)
        .unwrap_or(base_dof_count);
    let structural_solid_recovery = solver_mesh
        .as_ref()
        .map(solid_recovery_from_solver_mesh)
        .unwrap_or_default();

    let avg_youngs_modulus = if model.materials.is_empty() {
        1.0e9
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.youngs_modulus_pa.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let avg_poisson_ratio = if model.materials.is_empty() {
        0.3
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.poisson_ratio.clamp(0.0, 0.49))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let avg_reference_temperature_k = if model.materials.is_empty() {
        293.15
    } else {
        model
            .materials
            .iter()
            .map(|material| material.thermal.reference_temperature_k)
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let shear_modulus_pa = avg_youngs_modulus / (2.0 * (1.0 + avg_poisson_ratio)).max(1.0e-9);
    let avg_density_kg_per_m3 = if model.materials.is_empty() {
        7850.0
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.density_kg_per_m3.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let lame_lambda_pa = avg_youngs_modulus * avg_poisson_ratio
        / ((1.0 + avg_poisson_ratio) * (1.0 - 2.0 * avg_poisson_ratio)).max(1.0e-9);
    let structural_material = StructuralMaterialSummary {
        youngs_modulus_pa: avg_youngs_modulus,
        poisson_ratio: avg_poisson_ratio,
        density_kg_per_m3: avg_density_kg_per_m3,
        lame_lambda_pa,
        shear_modulus_pa,
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
    for (i, load) in model.loads.iter().enumerate() {
        let base = (i * 3) % dof_count;
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
            runmat_analysis_core::LoadKind::Moment { .. } => {}
            runmat_analysis_core::LoadKind::Wrench { fx, fy, fz, .. } => {
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
            runmat_analysis_core::LoadKind::CurrentDensity { jx, jy, jz, .. } => {
                rhs[base] += *jx * 1.0e-3;
                if base + 1 < dof_count {
                    rhs[base + 1] += *jy * 1.0e-3;
                }
                if base + 2 < dof_count {
                    rhs[base + 2] += *jz * 1.0e-3;
                }
            }
            runmat_analysis_core::LoadKind::CoilCurrent { current_a, .. } => {
                rhs[base] += *current_a * 1.0e-2;
            }
            runmat_analysis_core::LoadKind::HeatSource { .. } => {}
        }
    }

    let structural_dof_layout = StructuralDofLayout::legacy_translational_rows(dof_count);
    let structural_moment_load_count = model
        .loads
        .iter()
        .filter(|load| matches!(load.kind, runmat_analysis_core::LoadKind::Moment { .. }))
        .count();
    let structural_direct_rotational_moment_load_count =
        if structural_dof_layout.has_rotational_dofs() {
            structural_moment_load_count
        } else {
            0
        };

    let legacy_constrained_dof_count = model.boundary_conditions.len().min(dof_count);
    let mut constrained = vec![false; dof_count];
    let constraint_offset = 0;
    for idx in 0..legacy_constrained_dof_count {
        let dof = (constraint_offset + idx * 2) % dof_count.max(1);
        constrained[dof] = true;
        rhs[dof] = 0.0;
    }
    let mut structural_wrench_lowering = Vec::new();
    if let (Some(mesh), Some(_)) = (solver_mesh.as_ref(), solid_topology.as_ref()) {
        structural_wrench_lowering = apply_solver_mesh_structural_regions(
            model,
            mesh,
            &structural_dof_layout,
            &mut constrained,
            &mut rhs,
            strict_solver_mesh_stiffness,
        )
        .map_err(LinearAssemblyError::SolverMeshRegionMapping)?;
    }

    let mut thermo_mechanical = None;
    let mut electro_thermal = None;
    for i in 0..stiffness_upper.len() {
        let coupling = 0.05 * stiffness_diag[i].min(stiffness_diag[i + 1]);
        stiffness_upper[i] = if constrained[i] || constrained[i + 1] {
            0.0
        } else {
            coupling
        };
    }

    if let Some(context) = thermo_mechanical_context {
        if context.enabled {
            let thermal_strain_scale = (context.thermal_expansion_coefficient
                * context.applied_temperature_delta_k.abs())
            .clamp(0.0, 0.05);
            let thermal_load_scale = (context.applied_temperature_delta_k / 50.0).clamp(-2.0, 2.0);
            let constitutive_temperature_factor = if model.materials.is_empty() {
                (-(2.5e-4) * context.applied_temperature_delta_k).clamp(-0.25, 0.25)
            } else {
                let response = model
                    .materials
                    .iter()
                    .map(|material| {
                        let adjusted_delta = context.applied_temperature_delta_k
                            + (context.reference_temperature_k
                                - material.thermal.reference_temperature_k)
                            + (avg_reference_temperature_k
                                - material.thermal.reference_temperature_k)
                                * 0.1;
                        material.thermal.modulus_temp_coeff_per_k * adjusted_delta
                    })
                    .sum::<f64>()
                    / model.materials.len() as f64;
                response.clamp(-0.25, 0.25)
            };
            let constitutive_poisson_coupling =
                (0.6 + avg_poisson_ratio.clamp(0.0, 0.49)).clamp(0.6, 1.2);
            let modulus_temperature_scale = (1.0
                + constitutive_temperature_factor * constitutive_poisson_coupling)
                .clamp(0.72, 1.15);
            let thermal_stiffening_scale = (1.0 + 0.35 * thermal_strain_scale).clamp(1.0, 1.06);
            let effective_modulus_scale =
                (modulus_temperature_scale * thermal_stiffening_scale).clamp(0.75, 1.2);
            let mut dof_adjustments = vec![0.0_f64; dof_count];
            let assignment_heterogeneity_index = apply_thermo_material_heterogeneity(
                model,
                dof_count,
                constitutive_temperature_factor,
                context.reference_temperature_k,
                context.applied_temperature_delta_k,
                &mut dof_adjustments,
            );
            let spatial_field =
                apply_thermo_spatial_field(&context, dof_count, &mut dof_adjustments);
            let temporal_profile_variation =
                thermo_mechanical::temporal_profile_variation(Some(&context));
            let mut local_modulus_scales = vec![effective_modulus_scale; dof_count];
            for i in 0..dof_count {
                let thermal_bias = 1.0 + thermal_strain_scale * (1.0 + (i % 3) as f64 * 0.1);
                let local_scale =
                    (effective_modulus_scale * (1.0 + dof_adjustments[i])).clamp(0.75, 1.2);
                local_modulus_scales[i] = local_scale;
                stiffness_diag[i] *= thermal_bias * local_scale;
                if !constrained[i] {
                    rhs[i] += thermal_load_scale * (1.0 + (i % 5) as f64 * 0.05);
                }
            }
            for i in 0..stiffness_upper.len() {
                let edge_scale = 0.5 * (local_modulus_scales[i] + local_modulus_scales[i + 1]);
                stiffness_upper[i] *= edge_scale;
            }
            let min_modulus_scale = local_modulus_scales
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max_modulus_scale = local_modulus_scales.iter().copied().fold(0.0_f64, f64::max);
            let constitutive_material_spread_ratio =
                if min_modulus_scale.is_finite() && min_modulus_scale > 0.0 {
                    max_modulus_scale / min_modulus_scale
                } else {
                    1.0
                };
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
                constitutive_material_spread_ratio,
                assignment_heterogeneity_index,
                spatial_gradient_index: spatial_field.gradient_index,
                spatial_coverage_ratio: spatial_field.coverage_ratio,
                temporal_profile_variation,
                region_delta_count: context.region_temperature_deltas.len(),
                coupling_fingerprint: thermo_mechanical_fingerprint(
                    &context,
                    ThermoMechanicalFingerprintInputs {
                        dof_count,
                        constitutive_temperature_factor,
                        constitutive_poisson_coupling,
                        effective_modulus_scale,
                        constitutive_material_spread_ratio,
                        assignment_heterogeneity_index,
                        spatial_gradient_index: spatial_field.gradient_index,
                        temporal_profile_variation,
                    },
                ),
            });
        }
    }

    if let Some(context) = electro_thermal_context {
        if context.enabled {
            let temporal_variation = if context.time_profile.len() < 2 {
                0.0
            } else {
                let min_scale = context
                    .time_profile
                    .iter()
                    .map(|point| point.current_scale)
                    .fold(f64::INFINITY, f64::min);
                let max_scale = context
                    .time_profile
                    .iter()
                    .map(|point| point.current_scale)
                    .fold(-f64::INFINITY, f64::max);
                ((max_scale - min_scale).abs() / 2.0).clamp(0.0, 1.0)
            };
            let mut conductivity_scales = vec![1.0_f64; dof_count];
            for (idx, scale) in context.region_conductivity_scales.iter().enumerate() {
                let cursor = (idx * 5 + scale.region_id.len()) % dof_count.max(1);
                conductivity_scales[cursor] = scale.conductivity_scale.clamp(0.2, 2.5);
            }
            let min_scale = conductivity_scales
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
                .max(1.0e-6);
            let max_scale = conductivity_scales.iter().copied().fold(0.0_f64, f64::max);
            let conductivity_spread_ratio = (max_scale / min_scale).clamp(1.0, 8.0);
            let joule_heating_scale = (context.applied_voltage_v.powi(2)
                * context.base_electrical_conductivity_s_per_m.max(1.0e-9)
                * context.resistive_heating_coefficient.max(0.0)
                / 1.0e6)
                .clamp(0.0, 10.0);

            for i in 0..dof_count {
                let local = conductivity_scales[i];
                damping_diag[i] *= (1.0 + 0.02 * local).clamp(1.0, 1.1);
                if !constrained[i] {
                    rhs[i] += joule_heating_scale * local * (1.0 + (i % 7) as f64 * 0.01);
                }
            }

            electro_thermal = Some(ElectroThermalAssemblySummary {
                enabled: true,
                reference_temperature_k: context.reference_temperature_k,
                applied_voltage_v: context.applied_voltage_v,
                base_electrical_conductivity_s_per_m: context.base_electrical_conductivity_s_per_m,
                resistive_heating_coefficient: context.resistive_heating_coefficient,
                joule_heating_scale,
                conductivity_spread_ratio,
                temporal_profile_variation: temporal_variation,
                region_scale_count: context.region_conductivity_scales.len(),
                coupling_fingerprint: electro_thermal_fingerprint(
                    &context,
                    dof_count,
                    joule_heating_scale,
                    conductivity_spread_ratio,
                    temporal_variation,
                ),
            });
        }
    }

    let stiffness_csr = match solver_mesh.as_ref() {
        Some(mesh) => match assemble_solver_solid_stiffness_csr(
            mesh,
            (model.materials.len() == 1).then_some(SolidMaterial {
                youngs_modulus_pa: structural_material.youngs_modulus_pa,
                poisson_ratio: structural_material.poisson_ratio,
            }),
            &solid_materials_by_region(model),
            base_dof_count,
        ) {
            Ok(dense) => Some(dense),
            Err(err) if strict_solver_mesh_stiffness => {
                return Err(LinearAssemblyError::SolidStiffness(err));
            }
            Err(_) => None,
        },
        None => None,
    };
    if let Some(csr) = stiffness_csr.as_ref() {
        apply_csr_constraints(csr, &constrained, &mut rhs, dof_count);
        for (i, diagonal) in stiffness_diag.iter_mut().enumerate().take(dof_count) {
            let start = csr.row_offsets[i];
            let end = csr.row_offsets[i + 1];
            *diagonal = csr.column_indices[start..end]
                .iter()
                .zip(csr.values[start..end].iter())
                .find_map(|(&column, &value)| (column == i).then_some(value.abs()))
                .unwrap_or(1.0e-12)
                .max(1.0e-12);
        }
        stiffness_upper.fill(0.0);
    }

    let constrained_dof_count = constrained.iter().filter(|value| **value).count();

    Ok(AssemblySummary {
        dof_count,
        structural_node_count: structural_dof_layout.node_count(),
        structural_translational_dof_count: structural_dof_layout.translational_dof_count(),
        structural_rotational_dof_count: structural_dof_layout.rotational_dof_count(),
        structural_rotation_node_count: structural_dof_layout.rotation_node_count(),
        structural_moment_load_count,
        structural_direct_rotational_moment_load_count,
        structural_wrench_lowering,
        structural_rotational_constraint_count: 0,
        structural_beam_element_count: 0,
        structural_shell_element_count: 0,
        structural_solid_element_count: solid_topology
            .as_ref()
            .map(|topology| topology.volume_element_count)
            .unwrap_or(0),
        structural_solid_recovery,
        structural_dof_layout,
        structural_beam_recovery: Vec::new(),
        structural_shell_recovery: Vec::new(),
        constrained_dof_count,
        load_count: model.loads.len(),
        structural_material,
        thermo_mechanical,
        electro_thermal,
        operator: OperatorSystem {
            dof_count,
            constrained,
            stiffness_dense: None,
            stiffness_csr,
            stiffness_diag,
            stiffness_upper,
            mass_diag,
            damping_diag,
            rhs,
        },
    })
}

fn solid_recovery_from_solver_mesh(mesh: &SolverMeshArtifact) -> Vec<SolidRecoveryElementSummary> {
    let node_indices = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();

    mesh.topology
        .volume_elements
        .iter()
        .filter(|element| element.node_ids.len() == 4)
        .filter_map(|element| {
            let mut indices = [0_usize; 4];
            let mut coordinates_m = [[0.0_f64; 3]; 4];
            for (local, node_id) in element.node_ids.iter().copied().enumerate() {
                let node_index = *node_indices.get(&node_id)?;
                indices[local] = node_index;
                coordinates_m[local] = mesh.topology.nodes.get(node_index)?.coordinates_m;
            }
            Some(SolidRecoveryElementSummary {
                element_id: element.element_id.to_string(),
                region_id: element.region_id.source_topology_id.clone(),
                node_indices: indices,
                coordinates_m,
            })
        })
        .collect()
}

fn electro_thermal_fingerprint(
    context: &FeaElectroThermalContext,
    dof_count: usize,
    joule_heating_scale: f64,
    conductivity_spread_ratio: f64,
    temporal_profile_variation: f64,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        context.reference_temperature_k.to_bits(),
        context.applied_voltage_v.to_bits(),
        context.base_electrical_conductivity_s_per_m.to_bits(),
        context.resistive_heating_coefficient.to_bits(),
        joule_heating_scale.to_bits(),
        conductivity_spread_ratio.to_bits(),
        temporal_profile_variation.to_bits(),
        dof_count as u64,
        context.region_conductivity_scales.len() as u64,
        context.time_profile.len() as u64,
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

fn assemble_beam_system(model: &AnalysisModel) -> Option<AssemblySummary> {
    let structural = model.structural.as_ref()?;
    let beam_elements = structural
        .elements
        .iter()
        .filter(|element| matches!(element.kind, StructuralElementKind::Beam(_)))
        .collect::<Vec<_>>();
    let shell_elements = structural
        .elements
        .iter()
        .filter(|element| matches!(element.kind, StructuralElementKind::Shell(_)))
        .collect::<Vec<_>>();
    if (beam_elements.is_empty() && shell_elements.is_empty()) || structural.nodes.is_empty() {
        return None;
    }

    let structural_material = structural_material_summary(model);
    let beam_material = BeamMaterial {
        youngs_modulus_pa: structural_material.youngs_modulus_pa,
        shear_modulus_pa: structural_material.shear_modulus_pa,
    };
    let shell_material = ShellMaterial {
        youngs_modulus_pa: structural_material.youngs_modulus_pa,
        poisson_ratio: structural_material.poisson_ratio,
        shear_modulus_pa: structural_material.shear_modulus_pa,
    };
    let node_count = structural.nodes.len();
    let structural_dof_layout = StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational_rotational();
        node_count
    ]);
    let dof_count = structural_dof_layout.total_dof_count();
    let mut dense = vec![0.0_f64; dof_count * dof_count];
    let mut rhs = vec![0.0_f64; dof_count];
    let mut mass_diag = vec![0.0_f64; dof_count];
    let mut damping_diag = vec![0.0_f64; dof_count];
    let mut structural_beam_recovery = Vec::new();
    let mut structural_shell_recovery = Vec::new();

    for element in &beam_elements {
        let StructuralElementKind::Beam(beam) = &element.kind else {
            continue;
        };
        let node_i_index = structural_node_index(structural, beam.node_ids[0])?;
        let node_j_index = structural_node_index(structural, beam.node_ids[1])?;
        let section = structural_beam_section(structural, &beam.section_id)?;
        let geometry = BeamElementGeometry {
            node_i_m: structural.nodes[node_i_index].coordinates_m,
            node_j_m: structural.nodes[node_j_index].coordinates_m,
            reference_axis: beam.reference_axis,
        };
        let frame = geometry.local_frame().ok()?;
        let transform_global_to_local = transformation_matrix(frame);
        let stiffness = beam_global_stiffness_matrix(section, beam_material, geometry).ok()?;
        let element_dofs =
            beam_element_dof_indices(&structural_dof_layout, node_i_index, node_j_index)?;
        for (local_row, &global_row) in element_dofs.iter().enumerate() {
            for (local_col, &global_col) in element_dofs.iter().enumerate() {
                dense[global_row * dof_count + global_col] += stiffness[local_row][local_col];
            }
        }
        structural_beam_recovery.push(BeamRecoveryElementSummary {
            element_id: element.element_id.clone(),
            region_id: element.region_id.clone(),
            node_i_index,
            node_j_index,
            length_m: frame.length_m,
            section,
            material: beam_material,
            transform_global_to_local,
        });

        add_lumped_beam_mass_and_damping(
            &mut mass_diag,
            &mut damping_diag,
            &element_dofs,
            section,
            structural_material.density_kg_per_m3,
            frame.length_m,
            &transform_global_to_local,
        );
    }

    for element in &shell_elements {
        let StructuralElementKind::Shell(shell) = &element.kind else {
            continue;
        };
        let node_indices = [
            structural_node_index(structural, shell.node_ids[0])?,
            structural_node_index(structural, shell.node_ids[1])?,
            structural_node_index(structural, shell.node_ids[2])?,
        ];
        let section = structural_shell_section(structural, &shell.section_id)?;
        let geometry = ShellElementGeometry {
            nodes_m: [
                structural.nodes[node_indices[0]].coordinates_m,
                structural.nodes[node_indices[1]].coordinates_m,
                structural.nodes[node_indices[2]].coordinates_m,
            ],
            reference_axis: shell.reference_axis,
        };
        let frame = geometry.local_frame().ok()?;
        let stiffness = shell_global_stiffness_matrix(section, shell_material, geometry).ok()?;
        let element_dofs = shell_element_dof_indices(&structural_dof_layout, node_indices)?;
        for (local_row, &global_row) in element_dofs.iter().enumerate() {
            for (local_col, &global_col) in element_dofs.iter().enumerate() {
                dense[global_row * dof_count + global_col] += stiffness[local_row][local_col];
            }
        }
        add_lumped_shell_mass_and_damping(
            &mut mass_diag,
            &mut damping_diag,
            &element_dofs,
            section,
            structural_material.density_kg_per_m3,
            frame.area_m2,
        );
        structural_shell_recovery.push(ShellRecoveryElementSummary {
            element_id: element.element_id.clone(),
            region_id: element.region_id.clone(),
            node_indices,
            area_m2: frame.area_m2,
            section,
            material: shell_material,
        });
    }

    let mut direct_rotational_moment_load_count = 0usize;
    let mut structural_wrench_lowering = Vec::new();
    for load in &model.loads {
        let target_nodes = structural_target_nodes(structural, &load.region_id);
        if target_nodes.is_empty() {
            continue;
        }
        let scale = 1.0 / target_nodes.len() as f64;
        match load.kind {
            LoadKind::Force { fx, fy, fz } => {
                for node_index in target_nodes {
                    add_rhs(
                        &structural_dof_layout,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Ux,
                        fx * scale,
                    );
                    add_rhs(
                        &structural_dof_layout,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Uy,
                        fy * scale,
                    );
                    add_rhs(
                        &structural_dof_layout,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Uz,
                        fz * scale,
                    );
                }
            }
            LoadKind::Moment { mx, my, mz } => {
                direct_rotational_moment_load_count += 1;
                for node_index in target_nodes {
                    add_rhs(
                        &structural_dof_layout,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Rx,
                        mx * scale,
                    );
                    add_rhs(
                        &structural_dof_layout,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Ry,
                        my * scale,
                    );
                    add_rhs(
                        &structural_dof_layout,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Rz,
                        mz * scale,
                    );
                }
            }
            LoadKind::Wrench {
                fx,
                fy,
                fz,
                mx,
                my,
                mz,
                px,
                py,
                pz,
            } => {
                let summary = add_wrench_rhs(
                    structural,
                    &structural_dof_layout,
                    &mut rhs,
                    &target_nodes,
                    [fx, fy, fz],
                    [mx, my, mz],
                    [px, py, pz],
                );
                structural_wrench_lowering.push(WrenchLoweringSummary {
                    load_id: load.load_id.clone(),
                    region_id: load.region_id.clone(),
                    ..summary
                });
            }
            _ => {}
        }
    }

    let mut constrained = vec![false; dof_count];
    for bc in &model.boundary_conditions {
        let target_nodes = structural_target_nodes(structural, &bc.region_id);
        for node_index in target_nodes {
            match bc.kind {
                BoundaryConditionKind::Fixed => {
                    for kind in StructuralDofKind::ORDER {
                        constrain_dof(
                            &structural_dof_layout,
                            &mut constrained,
                            &mut rhs,
                            node_index,
                            kind,
                            0.0,
                        );
                    }
                }
                BoundaryConditionKind::PrescribedDisplacement => {
                    for kind in [
                        StructuralDofKind::Ux,
                        StructuralDofKind::Uy,
                        StructuralDofKind::Uz,
                    ] {
                        constrain_dof(
                            &structural_dof_layout,
                            &mut constrained,
                            &mut rhs,
                            node_index,
                            kind,
                            0.0,
                        );
                    }
                }
                BoundaryConditionKind::PrescribedRotation { rx, ry, rz } => {
                    constrain_dof(
                        &structural_dof_layout,
                        &mut constrained,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Rx,
                        rx,
                    );
                    constrain_dof(
                        &structural_dof_layout,
                        &mut constrained,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Ry,
                        ry,
                    );
                    constrain_dof(
                        &structural_dof_layout,
                        &mut constrained,
                        &mut rhs,
                        node_index,
                        StructuralDofKind::Rz,
                        rz,
                    );
                }
                _ => {}
            }
        }
    }
    apply_dense_constraints(&dense, &constrained, &mut rhs, dof_count);

    let stiffness_diag = (0..dof_count)
        .map(|idx| {
            if constrained[idx] {
                1.0
            } else {
                dense[idx * dof_count + idx].abs().max(1.0e-12)
            }
        })
        .collect::<Vec<_>>();
    let stiffness_upper = vec![0.0; dof_count.saturating_sub(1)];
    let constrained_dof_count = constrained.iter().filter(|value| **value).count();
    let rotational_constraint_count = constrained
        .iter()
        .enumerate()
        .filter(|(_, is_constrained)| **is_constrained)
        .filter(|(row, _)| {
            structural_dof_layout
                .address(*row)
                .is_some_and(|address| address.kind.is_rotational())
        })
        .count();

    Some(AssemblySummary {
        dof_count,
        structural_node_count: structural_dof_layout.node_count(),
        structural_translational_dof_count: structural_dof_layout.translational_dof_count(),
        structural_rotational_dof_count: structural_dof_layout.rotational_dof_count(),
        structural_rotation_node_count: structural_dof_layout.rotation_node_count(),
        structural_moment_load_count: model
            .loads
            .iter()
            .filter(|load| matches!(load.kind, LoadKind::Moment { .. }))
            .count(),
        structural_direct_rotational_moment_load_count: direct_rotational_moment_load_count,
        structural_wrench_lowering,
        structural_rotational_constraint_count: rotational_constraint_count,
        structural_beam_element_count: beam_elements.len(),
        structural_shell_element_count: shell_elements.len(),
        structural_solid_element_count: 0,
        structural_solid_recovery: Vec::new(),
        structural_dof_layout,
        structural_beam_recovery,
        structural_shell_recovery,
        constrained_dof_count,
        load_count: model.loads.len(),
        structural_material,
        thermo_mechanical: None,
        electro_thermal: None,
        operator: OperatorSystem {
            dof_count,
            constrained,
            stiffness_dense: Some(dense),
            stiffness_csr: None,
            stiffness_diag,
            stiffness_upper,
            mass_diag,
            damping_diag,
            rhs,
        },
    })
}

fn structural_material_summary(model: &AnalysisModel) -> StructuralMaterialSummary {
    let avg_youngs_modulus = if model.materials.is_empty() {
        1.0e9
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.youngs_modulus_pa.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let avg_poisson_ratio = if model.materials.is_empty() {
        0.3
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.poisson_ratio.clamp(0.0, 0.49))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let shear_modulus_pa = avg_youngs_modulus / (2.0 * (1.0 + avg_poisson_ratio)).max(1.0e-9);
    let avg_density_kg_per_m3 = if model.materials.is_empty() {
        7850.0
    } else {
        model
            .materials
            .iter()
            .map(|material| material.mechanical.density_kg_per_m3.max(1.0))
            .sum::<f64>()
            / model.materials.len() as f64
    };
    let lame_lambda_pa = avg_youngs_modulus * avg_poisson_ratio
        / ((1.0 + avg_poisson_ratio) * (1.0 - 2.0 * avg_poisson_ratio)).max(1.0e-9);
    StructuralMaterialSummary {
        youngs_modulus_pa: avg_youngs_modulus,
        poisson_ratio: avg_poisson_ratio,
        density_kg_per_m3: avg_density_kg_per_m3,
        lame_lambda_pa,
        shear_modulus_pa,
    }
}

fn solid_materials_by_region(model: &AnalysisModel) -> BTreeMap<String, SolidMaterial> {
    let materials = model
        .materials
        .iter()
        .map(|material| {
            (
                material.material_id.clone(),
                SolidMaterial {
                    youngs_modulus_pa: material.mechanical.youngs_modulus_pa.max(1.0),
                    poisson_ratio: material.mechanical.poisson_ratio.clamp(0.0, 0.49),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    model
        .material_assignments
        .iter()
        .filter_map(|assignment| {
            materials
                .get(&assignment.assigned_material_id)
                .copied()
                .map(|material| (assignment.region_id.clone(), material))
        })
        .collect()
}

fn structural_node_index(structural: &StructuralModel, node_id: u32) -> Option<usize> {
    structural
        .nodes
        .iter()
        .position(|node| node.node_id == node_id)
}

fn structural_beam_section(structural: &StructuralModel, section_id: &str) -> Option<BeamSection> {
    structural
        .beam_sections
        .iter()
        .find(|section| section.section_id == section_id)
        .map(section_from_model)
}

fn structural_shell_section(
    structural: &StructuralModel,
    section_id: &str,
) -> Option<ShellSection> {
    structural
        .shell_sections
        .iter()
        .find(|section| section.section_id == section_id)
        .map(shell_section_from_model)
}

fn section_from_model(section: &BeamSectionModel) -> BeamSection {
    BeamSection {
        area_m2: section.area_m2,
        iy_m4: section.iy_m4,
        iz_m4: section.iz_m4,
        torsion_j_m4: section.torsion_j_m4,
        outer_fiber_y_m: section.outer_fiber_y_m,
        outer_fiber_z_m: section.outer_fiber_z_m,
        torsion_outer_radius_m: section.torsion_outer_radius_m,
    }
}

fn shell_section_from_model(section: &ShellSectionModel) -> ShellSection {
    ShellSection {
        thickness_m: section.thickness_m,
        shear_correction: section.shear_correction,
        drilling_stiffness_scale: section.drilling_stiffness_scale,
    }
}

fn beam_element_dof_indices(
    layout: &StructuralDofLayout,
    node_i_index: usize,
    node_j_index: usize,
) -> Option<[usize; BEAM_ELEMENT_DOF_COUNT]> {
    let mut indices = [0usize; BEAM_ELEMENT_DOF_COUNT];
    for (local, kind) in StructuralDofKind::ORDER.iter().copied().enumerate() {
        indices[local] = layout.index(node_i_index, kind)?;
        indices[local + 6] = layout.index(node_j_index, kind)?;
    }
    Some(indices)
}

fn shell_element_dof_indices(
    layout: &StructuralDofLayout,
    node_indices: [usize; 3],
) -> Option<[usize; SHELL_ELEMENT_DOF_COUNT]> {
    let mut indices = [0usize; SHELL_ELEMENT_DOF_COUNT];
    for (node_offset, node_index) in node_indices.iter().enumerate() {
        for (component, kind) in StructuralDofKind::ORDER.iter().enumerate() {
            indices[node_offset * SHELL_NODE_DOF_COUNT + component] =
                layout.index(*node_index, *kind)?;
        }
    }
    Some(indices)
}

fn add_lumped_beam_mass_and_damping(
    mass_diag: &mut [f64],
    damping_diag: &mut [f64],
    element_dofs: &[usize; BEAM_ELEMENT_DOF_COUNT],
    section: BeamSection,
    density_kg_per_m3: f64,
    length_m: f64,
    transform_global_to_local: &BeamTransform12,
) {
    let density = density_kg_per_m3.max(1.0);
    let length = length_m.max(1.0e-12);
    let nodal_mass = density * section.area_m2.max(1.0e-18) * length / 2.0;
    let local_rotational_inertia = [
        density * section.torsion_j_m4.max(1.0e-24) * length / 2.0,
        density * section.iy_m4.max(1.0e-24) * length / 2.0,
        density * section.iz_m4.max(1.0e-24) * length / 2.0,
    ];

    for node_offset in [0usize, 6] {
        for component in 0..3 {
            let dof = element_dofs[node_offset + component];
            mass_diag[dof] += nodal_mass;
            damping_diag[dof] += 0.01;
        }
        for global_component in 0..3 {
            let local_col = node_offset + 3 + global_component;
            let inertia = local_rotational_inertia
                .iter()
                .enumerate()
                .map(|(local_component, local_inertia)| {
                    let row = node_offset + 3 + local_component;
                    local_inertia * transform_global_to_local[row][local_col].powi(2)
                })
                .sum::<f64>();
            let dof = element_dofs[node_offset + 3 + global_component];
            mass_diag[dof] += inertia.max(1.0e-18);
            damping_diag[dof] += 0.01;
        }
    }
}

fn add_lumped_shell_mass_and_damping(
    mass_diag: &mut [f64],
    damping_diag: &mut [f64],
    element_dofs: &[usize; SHELL_ELEMENT_DOF_COUNT],
    section: ShellSection,
    density_kg_per_m3: f64,
    area_m2: f64,
) {
    let density = density_kg_per_m3.max(1.0);
    let thickness = section.thickness_m.max(1.0e-12);
    let area = area_m2.max(1.0e-18);
    let nodal_mass = density * thickness * area / 3.0;
    let nodal_rotary_inertia = nodal_mass * thickness.powi(2) / 12.0;
    for node_offset in [0usize, 6, 12] {
        for component in 0..3 {
            let dof = element_dofs[node_offset + component];
            mass_diag[dof] += nodal_mass;
            damping_diag[dof] += 0.01;
        }
        for component in 3..6 {
            let dof = element_dofs[node_offset + component];
            mass_diag[dof] += nodal_rotary_inertia.max(1.0e-18);
            damping_diag[dof] += 0.01;
        }
    }
}

fn structural_target_nodes(structural: &StructuralModel, region_id: &str) -> Vec<usize> {
    if let Some(node_id) = structural_node_selector(region_id) {
        return structural_node_index(structural, node_id)
            .into_iter()
            .collect();
    }
    let mut nodes = Vec::new();
    for element in &structural.elements {
        if element.region_id != region_id {
            continue;
        }
        match &element.kind {
            StructuralElementKind::Beam(beam) => {
                for node_id in beam.node_ids {
                    if let Some(index) = structural_node_index(structural, node_id) {
                        if !nodes.contains(&index) {
                            nodes.push(index);
                        }
                    }
                }
            }
            StructuralElementKind::Shell(shell) => {
                for node_id in shell.node_ids {
                    if let Some(index) = structural_node_index(structural, node_id) {
                        if !nodes.contains(&index) {
                            nodes.push(index);
                        }
                    }
                }
            }
        }
    }
    nodes
}

fn structural_node_selector(region_id: &str) -> Option<u32> {
    region_id
        .strip_prefix("node:")
        .unwrap_or(region_id)
        .parse::<u32>()
        .ok()
}

fn add_rhs(
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    node_index: usize,
    kind: StructuralDofKind,
    value: f64,
) {
    if let Some(dof) = layout.index(node_index, kind) {
        rhs[dof] += value;
    }
}

fn add_wrench_rhs(
    structural: &StructuralModel,
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    target_nodes: &[usize],
    force: [f64; 3],
    moment_at_point: [f64; 3],
    point_m: [f64; 3],
) -> WrenchLoweringSummary {
    if target_nodes.is_empty() {
        return WrenchLoweringSummary {
            load_id: String::new(),
            region_id: String::new(),
            target_node_count: 0,
            applied_force: [0.0; 3],
            applied_moment_at_point: [0.0; 3],
            force_residual: force,
            moment_residual: moment_at_point,
            moment_couple_applied: false,
        };
    }

    let centroid = target_centroid(structural, target_nodes);
    let scale = 1.0 / target_nodes.len() as f64;
    let mut nodal_forces = Vec::with_capacity(target_nodes.len());
    for &node_index in target_nodes {
        let nodal_force = scale_vec(force, scale);
        add_translational_rhs(layout, rhs, node_index, nodal_force);
        nodal_forces.push(nodal_force);
    }

    let force_arm = [
        centroid[0] - point_m[0],
        centroid[1] - point_m[1],
        centroid[2] - point_m[2],
    ];
    let force_moment = cross(force_arm, force);
    let couple = [
        moment_at_point[0] - force_moment[0],
        moment_at_point[1] - force_moment[1],
        moment_at_point[2] - force_moment[2],
    ];
    let mut moment_couple_applied = false;

    if !couple
        .iter()
        .all(|component| component.abs() <= f64::EPSILON)
    {
        let mut coupling = [[0.0_f64; 3]; 3];
        let offsets = target_nodes
            .iter()
            .map(|&node_index| {
                let node = structural.nodes[node_index].coordinates_m;
                [
                    node[0] - centroid[0],
                    node[1] - centroid[1],
                    node[2] - centroid[2],
                ]
            })
            .collect::<Vec<_>>();
        for offset in &offsets {
            let r2 = dot(*offset, *offset);
            for row in 0..3 {
                coupling[row][row] += r2;
                for col in 0..3 {
                    coupling[row][col] -= offset[row] * offset[col];
                }
            }
        }

        if let Some(inv) = invert_3x3(coupling) {
            let lambda = mat_vec(inv, couple);
            for ((&node_index, offset), nodal_force) in target_nodes
                .iter()
                .zip(offsets.iter())
                .zip(nodal_forces.iter_mut())
            {
                let couple_force = cross(lambda, *offset);
                add_translational_rhs(layout, rhs, node_index, couple_force);
                nodal_force[0] += couple_force[0];
                nodal_force[1] += couple_force[1];
                nodal_force[2] += couple_force[2];
            }
            moment_couple_applied = true;
        }
    }

    let (applied_force, applied_moment_at_point) =
        wrench_resultants(structural, target_nodes, &nodal_forces, point_m);
    WrenchLoweringSummary {
        load_id: String::new(),
        region_id: String::new(),
        target_node_count: target_nodes.len(),
        applied_force,
        applied_moment_at_point,
        force_residual: [
            force[0] - applied_force[0],
            force[1] - applied_force[1],
            force[2] - applied_force[2],
        ],
        moment_residual: [
            moment_at_point[0] - applied_moment_at_point[0],
            moment_at_point[1] - applied_moment_at_point[1],
            moment_at_point[2] - applied_moment_at_point[2],
        ],
        moment_couple_applied,
    }
}

fn add_translational_rhs(
    layout: &StructuralDofLayout,
    rhs: &mut [f64],
    node_index: usize,
    force: [f64; 3],
) {
    add_rhs(layout, rhs, node_index, StructuralDofKind::Ux, force[0]);
    add_rhs(layout, rhs, node_index, StructuralDofKind::Uy, force[1]);
    add_rhs(layout, rhs, node_index, StructuralDofKind::Uz, force[2]);
}

fn target_centroid(structural: &StructuralModel, target_nodes: &[usize]) -> [f64; 3] {
    let mut centroid = [0.0_f64; 3];
    for &node_index in target_nodes {
        let node = structural.nodes[node_index].coordinates_m;
        centroid[0] += node[0];
        centroid[1] += node[1];
        centroid[2] += node[2];
    }
    scale_vec(centroid, 1.0 / target_nodes.len() as f64)
}

fn wrench_resultants(
    structural: &StructuralModel,
    target_nodes: &[usize],
    nodal_forces: &[[f64; 3]],
    point_m: [f64; 3],
) -> ([f64; 3], [f64; 3]) {
    let mut applied_force = [0.0_f64; 3];
    let mut applied_moment = [0.0_f64; 3];
    for (&node_index, &force) in target_nodes.iter().zip(nodal_forces.iter()) {
        applied_force[0] += force[0];
        applied_force[1] += force[1];
        applied_force[2] += force[2];
        let node = structural.nodes[node_index].coordinates_m;
        let arm = [
            node[0] - point_m[0],
            node[1] - point_m[1],
            node[2] - point_m[2],
        ];
        let moment = cross(arm, force);
        applied_moment[0] += moment[0];
        applied_moment[1] += moment[1];
        applied_moment[2] += moment[2];
    }
    (applied_force, applied_moment)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale_vec(value: [f64; 3], scale: f64) -> [f64; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn mat_vec(matrix: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    [
        dot(matrix[0], value),
        dot(matrix[1], value),
        dot(matrix[2], value),
    ]
}

fn invert_3x3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let m = matrix;
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det.abs() <= 1.0e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}

fn constrain_dof(
    layout: &StructuralDofLayout,
    constrained: &mut [bool],
    rhs: &mut [f64],
    node_index: usize,
    kind: StructuralDofKind,
    value: f64,
) {
    if let Some(dof) = layout.index(node_index, kind) {
        constrained[dof] = true;
        rhs[dof] = value;
    }
}

fn apply_dense_constraints(dense: &[f64], constrained: &[bool], rhs: &mut [f64], dof_count: usize) {
    for dof in 0..dof_count {
        if !constrained[dof] {
            continue;
        }
        for row in 0..dof_count {
            if !constrained[row] {
                rhs[row] -= dense[row * dof_count + dof] * rhs[dof];
            }
        }
    }
}

fn apply_csr_constraints(csr: &CsrMatrix, constrained: &[bool], rhs: &mut [f64], dof_count: usize) {
    for row in 0..dof_count {
        if constrained[row] {
            continue;
        }
        let start = csr.row_offsets[row];
        let end = csr.row_offsets[row + 1];
        for entry in start..end {
            let column = csr.column_indices[entry];
            if constrained[column] {
                rhs[row] -= csr.values[entry] * rhs[column];
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ThermoMechanicalFingerprintInputs {
    dof_count: usize,
    constitutive_temperature_factor: f64,
    constitutive_poisson_coupling: f64,
    effective_modulus_scale: f64,
    constitutive_material_spread_ratio: f64,
    assignment_heterogeneity_index: f64,
    spatial_gradient_index: f64,
    temporal_profile_variation: f64,
}

fn thermo_mechanical_fingerprint(
    context: &FeaThermoMechanicalContext,
    inputs: ThermoMechanicalFingerprintInputs,
) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for value in [
        inputs.dof_count as u64,
        context.reference_temperature_k.to_bits(),
        context.applied_temperature_delta_k.to_bits(),
        context.thermal_expansion_coefficient.to_bits(),
        inputs.constitutive_temperature_factor.to_bits(),
        inputs.constitutive_poisson_coupling.to_bits(),
        inputs.effective_modulus_scale.to_bits(),
        inputs.constitutive_material_spread_ratio.to_bits(),
        inputs.assignment_heterogeneity_index.to_bits(),
        inputs.spatial_gradient_index.to_bits(),
        inputs.temporal_profile_variation.to_bits(),
    ] {
        hash ^= value;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ThermoSpatialFieldSummary {
    gradient_index: f64,
    coverage_ratio: f64,
}

fn apply_thermo_spatial_field(
    context: &FeaThermoMechanicalContext,
    dof_count: usize,
    dof_adjustments: &mut [f64],
) -> ThermoSpatialFieldSummary {
    if dof_count == 0 || context.region_temperature_deltas.is_empty() {
        return ThermoSpatialFieldSummary {
            gradient_index: 0.0,
            coverage_ratio: 0.0,
        };
    }
    let mut touched = vec![false; dof_count];
    let mut min_delta = f64::INFINITY;
    let mut max_delta = -f64::INFINITY;
    for (idx, region_delta) in context.region_temperature_deltas.iter().enumerate() {
        min_delta = min_delta.min(region_delta.temperature_delta_k);
        max_delta = max_delta.max(region_delta.temperature_delta_k);
        let normalized = ((region_delta.temperature_delta_k - context.applied_temperature_delta_k)
            / 240.0)
            .clamp(-0.45, 0.45);
        let start =
            ((region_hash(&region_delta.region_id) as usize).wrapping_add(idx * 5)) % dof_count;
        let stride = context
            .region_temperature_deltas
            .len()
            .saturating_add(3)
            .max(2);
        let mut cursor = start;
        for hop in 0..dof_count {
            if hop > 0 && cursor == start {
                break;
            }
            let wave = 1.0 + ((hop + idx) % 7) as f64 * 0.02;
            dof_adjustments[cursor] += normalized * wave;
            touched[cursor] = true;
            cursor = (cursor + stride) % dof_count;
        }
    }
    if !min_delta.is_finite() || !max_delta.is_finite() {
        return ThermoSpatialFieldSummary {
            gradient_index: 0.0,
            coverage_ratio: 0.0,
        };
    }
    let touched_count = touched.iter().filter(|entry| **entry).count() as f64;
    ThermoSpatialFieldSummary {
        gradient_index: ((max_delta - min_delta).abs() / 240.0).clamp(0.0, 1.0),
        coverage_ratio: (touched_count / dof_count as f64).clamp(0.0, 1.0),
    }
}

fn apply_thermo_material_heterogeneity(
    model: &AnalysisModel,
    dof_count: usize,
    constitutive_temperature_factor: f64,
    reference_temperature_k: f64,
    applied_temperature_delta_k: f64,
    dof_adjustments: &mut [f64],
) -> f64 {
    if dof_count == 0 || model.material_assignments.is_empty() {
        return 0.0;
    }
    let base_amplitude = (constitutive_temperature_factor.abs() * 0.8).clamp(0.0, 0.15);
    if base_amplitude <= 0.0 {
        return 0.0;
    }
    let mut weighted_activity = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for (idx, assignment) in model.material_assignments.iter().enumerate() {
        let confidence_weight = match assignment.confidence {
            runmat_analysis_core::EvidenceConfidence::Verified => 1.0,
            runmat_analysis_core::EvidenceConfidence::Probable => 0.65,
            runmat_analysis_core::EvidenceConfidence::Inferred => 0.4,
        };
        let expected_modulus = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.expected_material_id)
            .map(|material| material.mechanical.youngs_modulus_pa)
            .unwrap_or(1.0e9)
            .max(1.0);
        let assigned_modulus = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.assigned_material_id)
            .map(|material| material.mechanical.youngs_modulus_pa)
            .unwrap_or(expected_modulus)
            .max(1.0);
        let modulus_delta_ratio =
            ((assigned_modulus - expected_modulus) / expected_modulus).clamp(-0.6, 0.6);
        let expected_temp_response = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.expected_material_id)
            .map(|material| {
                material.thermal.modulus_temp_coeff_per_k
                    * (applied_temperature_delta_k
                        + (reference_temperature_k - material.thermal.reference_temperature_k))
            })
            .unwrap_or(constitutive_temperature_factor)
            .clamp(-0.4, 0.2);
        let assigned_temp_response = model
            .materials
            .iter()
            .find(|material| material.material_id == assignment.assigned_material_id)
            .map(|material| {
                material.thermal.modulus_temp_coeff_per_k
                    * (applied_temperature_delta_k
                        + (reference_temperature_k - material.thermal.reference_temperature_k))
            })
            .unwrap_or(expected_temp_response)
            .clamp(-0.4, 0.2);
        let response_delta = (assigned_temp_response - expected_temp_response).clamp(-0.35, 0.35);
        let region_phase = ((region_hash(&assignment.region_id) % 11) as f64) / 10.0;
        let activity =
            (0.7 * modulus_delta_ratio.abs() + 0.3 * response_delta.abs()).clamp(0.0, 1.0);
        let signed_bias = base_amplitude
            * confidence_weight
            * (0.45 * modulus_delta_ratio
                + 0.35 * response_delta
                + 0.2 * modulus_delta_ratio.signum() * region_phase);
        let stride = model.material_assignments.len().saturating_add(1).max(2);
        let start = ((region_hash(&assignment.region_id) as usize).wrapping_add(idx * 3))
            % dof_count.max(1);
        let mut cursor = start;
        for hop in 0..dof_count {
            if hop > 0 && cursor == start {
                break;
            }
            let wave = 1.0 + ((hop + idx) % 5) as f64 * 0.03;
            dof_adjustments[cursor] += signed_bias * wave;
            cursor = (cursor + stride) % dof_count.max(1);
        }
        weighted_activity += activity * confidence_weight;
        weight_sum += confidence_weight;
    }
    for value in dof_adjustments.iter_mut() {
        *value = value.clamp(-0.18, 0.18);
    }
    if weight_sum > 0.0 {
        (weighted_activity / weight_sum).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn region_hash(region_id: &str) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for byte in region_id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211_u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};
    use runmat_meshing_core::{
        ElementOrder, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact, StableDigest,
    };

    #[test]
    fn solver_mesh_populates_sparse_solid_stiffness_operator() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let summary = assemble_linear_system(&model, Some(tetrahedron4_mesh()), None, None);

        assert_eq!(summary.dof_count, 12);
        assert_eq!(summary.structural_solid_element_count, 1);
        assert_eq!(summary.structural_solid_recovery.len(), 1);
        assert_eq!(
            summary.structural_solid_recovery[0].node_indices,
            [0, 1, 2, 3]
        );
        assert_eq!(
            summary.structural_solid_recovery[0].coordinates_m,
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        );
        assert!(summary.operator.stiffness_dense.is_none());
        let csr = summary
            .operator
            .stiffness_csr
            .as_ref()
            .expect("solver mesh should assemble a sparse solid stiffness matrix");
        assert_eq!(csr.row_offsets.len(), summary.dof_count + 1);
        assert_eq!(csr.row_offsets.last().copied(), Some(csr.values.len()));
        assert_eq!(csr.column_indices.len(), csr.values.len());
        assert!(summary
            .operator
            .stiffness_diag
            .iter()
            .all(|value| *value > 0.0));
        for row in 0..summary.dof_count {
            let start = csr.row_offsets[row];
            let end = csr.row_offsets[row + 1];
            let diagonal = csr.column_indices[start..end]
                .iter()
                .zip(csr.values[start..end].iter())
                .find_map(|(&column, &value)| (column == row).then_some(value.abs()))
                .expect("csr row should include diagonal");
            assert!((summary.operator.stiffness_diag[row] - diagonal) <= 1.0e-8);
        }
    }

    #[test]
    fn solver_mesh_preempts_explicit_beam_topology() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.structural = Some(runmat_analysis_core::StructuralModel {
            nodes: vec![
                runmat_analysis_core::StructuralNode {
                    node_id: 1,
                    coordinates_m: [0.0, 0.0, 0.0],
                },
                runmat_analysis_core::StructuralNode {
                    node_id: 2,
                    coordinates_m: [1.0, 0.0, 0.0],
                },
            ],
            elements: vec![runmat_analysis_core::StructuralElement {
                element_id: "beam_1".to_string(),
                region_id: "span".to_string(),
                kind: runmat_analysis_core::StructuralElementKind::Beam(
                    runmat_analysis_core::BeamElementModel {
                        node_ids: [1, 2],
                        section_id: "rect".to_string(),
                        reference_axis: [0.0, 1.0, 0.0],
                    },
                ),
            }],
            beam_sections: vec![runmat_analysis_core::BeamSectionModel {
                section_id: "rect".to_string(),
                area_m2: 1.0e-4,
                iy_m4: 1.0e-9,
                iz_m4: 1.0e-9,
                torsion_j_m4: 1.0e-9,
                outer_fiber_y_m: 0.01,
                outer_fiber_z_m: 0.01,
                torsion_outer_radius_m: 0.01,
            }],
            shell_sections: Vec::new(),
        });

        let summary = assemble_linear_system(&model, Some(tetrahedron4_mesh()), None, None);

        assert_eq!(summary.structural_solid_element_count, 1);
        assert_eq!(summary.structural_solid_recovery.len(), 1);
        assert_eq!(summary.structural_beam_element_count, 0);
        assert!(summary.operator.stiffness_csr.is_some());
        assert!(summary.operator.stiffness_dense.is_none());
    }

    #[test]
    fn solver_mesh_material_regions_select_assigned_solid_materials() {
        let mut soft_model = fixture_model(FixtureId::CantileverLinearStatic);
        let mut hard = soft_model.materials[0].clone();
        hard.material_id = "mat_hard".to_string();
        hard.mechanical.youngs_modulus_pa = 200.0e9;
        hard.mechanical.poisson_ratio = 0.3;
        let mut soft = hard.clone();
        soft.material_id = "mat_soft".to_string();
        soft.mechanical.youngs_modulus_pa = 20.0e9;
        soft_model.materials = vec![hard.clone(), soft.clone()];
        soft_model.material_assignments = vec![runmat_analysis_core::MaterialAssignment {
            region_id: "soft_region".to_string(),
            expected_material_id: "mat_hard".to_string(),
            assigned_material_id: "mat_soft".to_string(),
            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
        }];
        let mut soft_mesh = tetrahedron4_mesh();
        set_solver_mesh_region(&mut soft_mesh, "soft_region");

        let mut hard_model = soft_model.clone();
        hard_model.material_assignments = vec![runmat_analysis_core::MaterialAssignment {
            region_id: "soft_region".to_string(),
            expected_material_id: "mat_hard".to_string(),
            assigned_material_id: "mat_hard".to_string(),
            confidence: runmat_analysis_core::EvidenceConfidence::Verified,
        }];
        let hard_mesh = soft_mesh.clone();
        assert_eq!(soft_mesh.canonical_digest, hard_mesh.canonical_digest);

        let soft_summary = assemble_linear_system(&soft_model, Some(soft_mesh), None, None);
        let hard_summary = assemble_linear_system(&hard_model, Some(hard_mesh), None, None);

        assert!(
            first_csr_diagonal(&soft_summary) < first_csr_diagonal(&hard_summary) * 0.2,
            "soft material assignment should lower solid element stiffness"
        );
        assert_eq!(
            soft_summary.structural_solid_recovery[0].region_id,
            "soft_region"
        );
    }

    #[test]
    fn strict_solver_mesh_assembly_rejects_invalid_tetrahedron4_stiffness() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let mut mesh = tetrahedron4_mesh();
        mesh.topology.volume_elements[0].node_ids = vec![1, 3, 2, 4];
        set_boundary_regions(&mut mesh, 0, &["root"]);
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let err = try_assemble_linear_system(&model, Some(mesh), None, None).expect_err(
            "strict solver mesh assembly should reject inverted Tetrahedron4 stiffness",
        );

        assert!(matches!(err, LinearAssemblyError::SolidStiffness(_)));
    }

    #[test]
    fn solver_mesh_boundary_regions_drive_solid_loads_and_constraints() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.boundary_conditions = vec![runmat_analysis_core::BoundaryCondition {
            bc_id: "fixed_root".to_string(),
            region_id: "root".to_string(),
            kind: BoundaryConditionKind::Fixed,
        }];
        model.loads = vec![runmat_analysis_core::LoadCase {
            load_id: "load_tip".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Force {
                fx: 0.0,
                fy: -12.0,
                fz: 0.0,
            },
        }];
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 0, &["root"]);
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let summary = assemble_linear_system(&model, Some(mesh), None, None);

        assert_eq!(summary.constrained_dof_count, 9);
        assert!(summary.operator.constrained[0]);
        assert!(summary.operator.constrained[1]);
        assert!(summary.operator.constrained[2]);
        assert_eq!(summary.operator.rhs[10], -4.0);
    }

    #[test]
    fn strict_solver_mesh_assembly_rejects_unmapped_load_region() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.boundary_conditions = vec![runmat_analysis_core::BoundaryCondition {
            bc_id: "fixed_root".to_string(),
            region_id: "root".to_string(),
            kind: BoundaryConditionKind::Fixed,
        }];
        model.loads = vec![runmat_analysis_core::LoadCase {
            load_id: "load_tip".to_string(),
            region_id: "missing_tip".to_string(),
            kind: LoadKind::Force {
                fx: 0.0,
                fy: -12.0,
                fz: 0.0,
            },
        }];
        let mut mesh = tetrahedron4_mesh();
        clear_boundary_regions(&mut mesh);
        set_boundary_regions(&mut mesh, 0, &["root"]);
        reseal(&mut mesh);

        let err = try_assemble_linear_system(&model, Some(mesh), None, None)
            .expect_err("strict solver mesh assembly should reject unmapped loads");

        assert!(matches!(
            err,
            LinearAssemblyError::SolverMeshRegionMapping(
                SolverMeshRegionMappingError::UnmappedLoadRegion { .. }
            )
        ));
        let message = err.to_string();
        assert!(message.contains("load_id=load_tip"));
        assert!(message.contains("region_id=missing_tip"));
    }

    #[test]
    fn strict_solver_mesh_assembly_rejects_unmapped_constraint_region() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.boundary_conditions = vec![runmat_analysis_core::BoundaryCondition {
            bc_id: "fixed_root".to_string(),
            region_id: "missing_root".to_string(),
            kind: BoundaryConditionKind::Fixed,
        }];
        model.loads = vec![runmat_analysis_core::LoadCase {
            load_id: "load_tip".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Force {
                fx: 0.0,
                fy: -12.0,
                fz: 0.0,
            },
        }];
        let mut mesh = tetrahedron4_mesh();
        clear_boundary_regions(&mut mesh);
        set_boundary_regions(&mut mesh, 0, &["tip"]);
        reseal(&mut mesh);

        let err = try_assemble_linear_system(&model, Some(mesh), None, None)
            .expect_err("strict solver mesh assembly should reject unmapped constraints");

        assert!(matches!(
            err,
            LinearAssemblyError::SolverMeshRegionMapping(
                SolverMeshRegionMappingError::UnmappedBoundaryConditionRegion { .. }
            )
        ));
        let message = err.to_string();
        assert!(message.contains("bc_id=fixed_root"));
        assert!(message.contains("region_id=missing_root"));
    }

    #[test]
    fn solver_mesh_boundary_regions_integrate_pressure_loads() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.boundary_conditions = Vec::new();
        model.loads = vec![runmat_analysis_core::LoadCase {
            load_id: "pressure_tip".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Pressure { magnitude_pa: 12.0 },
        }];
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let summary = assemble_linear_system(&model, Some(mesh), None, None);

        assert_close(summary.operator.rhs[1], -2.0);
        assert_close(summary.operator.rhs[4], -2.0);
        assert_close(summary.operator.rhs[10], -2.0);
    }

    #[test]
    fn solver_mesh_boundary_regions_lower_wrench_moments() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.boundary_conditions = Vec::new();
        model.loads = vec![runmat_analysis_core::LoadCase {
            load_id: "wrench_tip".to_string(),
            region_id: "tip".to_string(),
            kind: LoadKind::Wrench {
                fx: 0.0,
                fy: 0.0,
                fz: 0.0,
                mx: 0.0,
                my: 6.0,
                mz: 0.0,
                px: 0.0,
                py: 0.0,
                pz: 0.0,
            },
        }];
        let mut mesh = tetrahedron4_mesh();
        set_boundary_regions(&mut mesh, 1, &["tip"]);
        reseal(&mut mesh);

        let summary = assemble_linear_system(&model, Some(mesh), None, None);

        assert_eq!(summary.structural_wrench_lowering.len(), 1);
        let lowering = &summary.structural_wrench_lowering[0];
        assert_eq!(lowering.load_id, "wrench_tip");
        assert_eq!(lowering.region_id, "tip");
        assert!(lowering.moment_couple_applied);
        assert_close(lowering.applied_moment_at_point[1], 6.0);
        assert_close(lowering.moment_residual[1], 0.0);
        assert!(summary.operator.rhs.iter().any(|value| value.abs() > 0.0));
    }

    fn tetrahedron4_mesh() -> SolverMeshArtifact {
        crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4)
    }

    fn set_boundary_regions(mesh: &mut SolverMeshArtifact, face_index: usize, regions: &[&str]) {
        mesh.topology.boundary_faces[face_index].provenance = regions
            .iter()
            .map(|region| PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: (*region).to_owned(),
                assembly_path: vec!["root".to_owned()],
            })
            .collect();
        mesh.topology.boundary_faces[face_index].provenance.sort();
    }

    fn clear_boundary_regions(mesh: &mut SolverMeshArtifact) {
        for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
            face.provenance = vec![PersistentEntityId {
                kind: PersistentEntityKind::Face,
                source_topology_id: format!("unselected:{index}"),
                assembly_path: vec!["root".to_owned()],
            }];
        }
    }

    fn set_solver_mesh_region(mesh: &mut SolverMeshArtifact, region_id: &str) {
        let region = PersistentEntityId {
            kind: PersistentEntityKind::Region,
            source_topology_id: region_id.to_owned(),
            assembly_path: vec!["root".to_owned()],
        };
        mesh.topology.volume_elements[0].region_id = region.clone();
        mesh.topology.regions[0].region_id = region;
        reseal(mesh);
    }

    fn reseal(mesh: &mut SolverMeshArtifact) {
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest()
            .expect("test solver mesh should remain canonical");
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-8,
            "expected {expected}, got {actual}"
        );
    }

    fn first_csr_diagonal(summary: &AssemblySummary) -> f64 {
        let csr = summary
            .operator
            .stiffness_csr
            .as_ref()
            .expect("solver mesh should assemble CSR stiffness");
        csr.column_indices[csr.row_offsets[0]..csr.row_offsets[1]]
            .iter()
            .zip(csr.values[csr.row_offsets[0]..csr.row_offsets[1]].iter())
            .find_map(|(&column, &value)| (column == 0).then_some(value.abs()))
            .expect("first row should contain diagonal")
    }
}
