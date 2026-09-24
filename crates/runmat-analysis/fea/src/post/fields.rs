use runmat_analysis_core::AnalysisField;

use crate::contracts::{
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
    FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN, FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
    FEA_FIELD_STRUCTURAL_NODAL_VON_MISES, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_REACTION_MOMENT, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
    FEA_FIELD_STRUCTURAL_ROTATION, FEA_FIELD_STRUCTURAL_STRAIN,
    FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY, FEA_FIELD_STRUCTURAL_VON_MISES,
};
use crate::{
    algebra::apply_k,
    assembly::AssemblySummary,
    planning::resource::LocalExecutionPlan,
    post::{
        error::StructuralFieldRecoveryError, solid::recover_solid_fields_with_plan,
        volume_weighted::recover_volume_weighted_nodal_scalar,
    },
    solve::linear::LinearSolveResult,
};

mod catalog;
mod formulation;
mod scalars;
mod state;

pub use catalog::{
    structural_observable_field_definitions, take_structural_observable_fields,
    StructuralLinearFieldQuantity,
};
use formulation::{recover_beam_result_fields, recover_shell_result_fields};
use scalars::{recover_strain_energy_density, recover_stress, recover_von_mises};
use state::{
    recover_displacement, recover_reaction_force, recover_reaction_moment,
    recover_residual_metrics, recover_rotation, recover_total_strain_energy, rotation_shape,
    structural_displacement_node_count,
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const TENSOR_COMPONENT_COUNT: usize = 6;

pub fn recover_result_fields(
    summary: &AssemblySummary,
    solve_result: &LinearSolveResult,
    local_execution: LocalExecutionPlan,
) -> Result<Vec<AnalysisField>, StructuralFieldRecoveryError> {
    if solve_result.solution.is_empty()
        || solve_result.solution.iter().any(|value| !value.is_finite())
    {
        return Ok(empty_structural_fields());
    }

    let node_count = structural_displacement_node_count(summary, summary.dof_count.max(3));
    let internal_force = apply_k(&summary.operator, &solve_result.solution);
    let reaction_values = recover_reaction_force(summary, &internal_force, node_count);
    let reaction_moment_values = recover_reaction_moment(summary, &internal_force);
    let strain_energy = recover_total_strain_energy(&solve_result.solution, &internal_force);
    let residual_metrics = recover_residual_metrics(summary, &solve_result.solution);
    let mut fields = recover_deformation_result_fields_with_plan(
        summary,
        &solve_result.solution,
        local_execution,
    )?;
    fields.extend([
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_REACTION_FORCE,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            reaction_values,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
            rotation_shape(summary),
            reaction_moment_values,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
            vec![1],
            vec![strain_energy],
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
            vec![1],
            vec![residual_metrics.normalized_residual_norm],
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
            vec![1],
            vec![residual_metrics.equation_scale],
        ),
    ]);
    Ok(fields)
}

/// Recovers only deformation-dependent physical fields from a structural state vector.
///
/// Modal shapes have no static load equilibrium, so reaction and residual fields would be
/// misleading. This shared path keeps their strain, stress, energy-density, beam, and shell
/// recovery identical to linear-static recovery without manufacturing equilibrium quantities.
pub fn recover_deformation_result_fields(
    summary: &AssemblySummary,
    solution: &[f64],
) -> Result<Vec<AnalysisField>, StructuralFieldRecoveryError> {
    recover_deformation_result_fields_with_plan(summary, solution, LocalExecutionPlan::serial())
}

fn recover_deformation_result_fields_with_plan(
    summary: &AssemblySummary,
    solution: &[f64],
    local_execution: LocalExecutionPlan,
) -> Result<Vec<AnalysisField>, StructuralFieldRecoveryError> {
    if solution.is_empty() || solution.iter().any(|value| !value.is_finite()) {
        return Ok(Vec::new());
    }
    if solution.len() != summary.dof_count {
        return Err(StructuralFieldRecoveryError::DisplacementShapeMismatch {
            expected: summary.dof_count,
            actual: solution.len(),
        });
    }
    let node_count = structural_displacement_node_count(summary, summary.dof_count.max(3));
    let displacement_values = recover_displacement(summary, solution, node_count);
    let continuum_fields = if publishes_continuum_fields(summary) {
        let strain_recovery =
            recover_structural_strain_with_plan(summary, &displacement_values, local_execution)?;
        let element_count = strain_recovery.element_count;
        let strain_values = strain_recovery.values;
        let stress_values = strain_recovery
            .stress
            .unwrap_or_else(|| recover_stress(&strain_values, summary.structural_material));
        let von_mises_values = recover_von_mises(&stress_values);
        let nodal_von_mises_values = recover_volume_weighted_nodal_scalar(
            &summary.structural_solid_recovery,
            &von_mises_values,
            node_count,
        )?;
        let strain_energy_density_values = strain_recovery
            .strain_energy_density
            .unwrap_or_else(|| recover_strain_energy_density(&strain_values, &stress_values));
        let integration_point_strain = strain_recovery.integration_point_strain;
        let integration_point_stress = strain_recovery.integration_point_stress;
        Some((
            element_count,
            strain_values,
            stress_values,
            von_mises_values,
            nodal_von_mises_values,
            strain_energy_density_values,
            integration_point_strain,
            integration_point_stress,
        ))
    } else {
        None
    };

    let mut fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            displacement_values,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_ROTATION,
            rotation_shape(summary),
            recover_rotation(summary, solution),
        ),
    ];
    if let Some((
        element_count,
        strain_values,
        stress_values,
        von_mises_values,
        nodal_von_mises_values,
        strain_energy_density_values,
        integration_point_strain,
        integration_point_stress,
    )) = continuum_fields
    {
        fields.extend([
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_VON_MISES,
                vec![element_count],
                von_mises_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_STRAIN,
                vec![element_count, TENSOR_COMPONENT_COUNT],
                strain_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_STRESS,
                vec![element_count, TENSOR_COMPONENT_COUNT],
                stress_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
                vec![element_count],
                strain_energy_density_values,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN,
                vec![element_count * 4, TENSOR_COMPONENT_COUNT],
                integration_point_strain,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
                vec![element_count * 4, TENSOR_COMPONENT_COUNT],
                integration_point_stress,
            ),
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
                vec![node_count],
                nodal_von_mises_values,
            ),
        ]);
    }
    if let Some(beam_fields) = recover_beam_result_fields(summary, solution) {
        fields.extend(beam_fields);
    }
    if let Some(shell_fields) = recover_shell_result_fields(summary, solution)? {
        fields.extend(shell_fields);
    }
    Ok(fields)
}

fn publishes_continuum_fields(summary: &AssemblySummary) -> bool {
    !summary.structural_solid_recovery.is_empty()
}

/// Moves formulation-owned spatial fields with stable physical semantics into the observation
/// domain.
///
/// Diagnostics, global summaries, derived display conveniences, and unsupported beam/shell
/// resultants remain in `fields`; they are not mislabeled merely to make them observable.
/// Constraint-group reactions also remain here because their canonical values are recovered from
/// [`crate::constraints::CanonicalStructuralConstraintResult`], which preserves authored
/// constraint ownership when constraints share nodes or degrees of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralFieldRecoveryMetrics {
    pub active_stiffness_edge_count: usize,
    pub constrained_edge_count: usize,
    pub recovery_element_count: usize,
    pub solver_mesh_node_count: usize,
    pub solver_mesh_element_count: usize,
    pub max_edge_displacement_jump: f64,
    pub max_edge_strain_norm: f64,
    pub mean_edge_stiffness_ratio: f64,
    pub mean_edge_length_m: f64,
    pub strain_component_coverage_ratio: f64,
    pub element_geometry_node_count: usize,
    pub element_geometry_edge_count: usize,
    pub element_geometry_coverage_ratio: f64,
    pub basis: &'static str,
}

pub fn structural_field_recovery_metrics(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Result<StructuralFieldRecoveryMetrics, StructuralFieldRecoveryError> {
    let recovery_edges = structural_recovery_edges(summary);
    let strain_recovery = recover_structural_strain(summary, displacement)?;
    let active_stiffness_edge_count = recovery_edges.len();
    let constrained_edge_count = constrained_recovery_edge_count(summary);
    let mut max_edge_displacement_jump = 0.0_f64;
    let mut max_edge_strain_norm = 0.0_f64;
    let mut stiffness_ratio_sum = 0.0_f64;
    let mut edge_length_sum = 0.0_f64;
    for edge in &recovery_edges {
        let jump = displacement
            .get(edge.to_dof)
            .zip(displacement.get(edge.from_dof))
            .map(|(right, left)| (right - left).abs())
            .unwrap_or(0.0);
        max_edge_displacement_jump = max_edge_displacement_jump.max(jump);
        stiffness_ratio_sum += edge.stiffness_ratio;
        edge_length_sum += edge.edge_length_m;
    }
    for strain_tensor in strain_recovery.values.chunks_exact(TENSOR_COMPONENT_COUNT) {
        let strain_norm = strain_tensor
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        max_edge_strain_norm = max_edge_strain_norm.max(strain_norm);
    }
    let recovery_edge_count = recovery_edges.len();
    let mean_edge_stiffness_ratio = if recovery_edge_count == 0 {
        0.0
    } else {
        stiffness_ratio_sum / recovery_edge_count as f64
    };
    let mean_edge_length_m = if recovery_edge_count == 0 {
        0.0
    } else {
        edge_length_sum / recovery_edge_count as f64
    };
    let strain_component_coverage_ratio =
        strain_component_coverage_ratio(displacement, &recovery_edges);
    let solver_mesh_element_count = summary.structural_solid_recovery.len();
    let solver_mesh_node_count = if solver_mesh_element_count == 0 {
        0
    } else {
        let mut nodes = summary
            .structural_solid_recovery
            .iter()
            .flat_map(|element| element.geometry.node_indices().iter().copied())
            .collect::<Vec<_>>();
        nodes.sort_unstable();
        nodes.dedup();
        nodes.len()
    };

    Ok(StructuralFieldRecoveryMetrics {
        active_stiffness_edge_count,
        constrained_edge_count,
        recovery_element_count: strain_recovery.element_count,
        solver_mesh_node_count,
        solver_mesh_element_count,
        max_edge_displacement_jump,
        max_edge_strain_norm,
        mean_edge_stiffness_ratio,
        mean_edge_length_m,
        strain_component_coverage_ratio,
        element_geometry_node_count: solver_mesh_node_count,
        element_geometry_edge_count: solver_mesh_element_count.saturating_mul(6),
        element_geometry_coverage_ratio: f64::from(solver_mesh_element_count > 0),
        basis: strain_recovery.basis,
    })
}

struct StructuralStrainRecovery {
    values: Vec<f64>,
    stress: Option<Vec<f64>>,
    strain_energy_density: Option<Vec<f64>>,
    integration_point_strain: Vec<f64>,
    integration_point_stress: Vec<f64>,
    element_count: usize,
    basis: &'static str,
}

fn empty_structural_fields() -> Vec<AnalysisField> {
    vec![
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_DISPLACEMENT, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_ROTATION, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![0], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRAIN, vec![0, 6], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRESS, vec![0, 6], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_NODAL_VON_MISES, vec![0], Vec::new()),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
            vec![0],
            Vec::new(),
        ),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_REACTION_FORCE, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_REACTION_MOMENT, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
            vec![0],
            Vec::new(),
        ),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, vec![0], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_EQUATION_SCALE, vec![0], Vec::new()),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct StructuralRecoveryEdge {
    from_dof: usize,
    to_dof: usize,
    component: usize,
    hop: usize,
    edge_length_m: f64,
    stiffness_ratio: f64,
    basis: StructuralRecoveryBasis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuralRecoveryBasis {
    OperatorConnectivity,
}

impl StructuralRecoveryBasis {
    const fn as_str(self) -> &'static str {
        match self {
            Self::OperatorConnectivity => "operator_connectivity",
        }
    }
}

fn structural_recovery_edges(summary: &AssemblySummary) -> Vec<StructuralRecoveryEdge> {
    let matrix = summary.operator.operators().stiffness();
    let constrained = summary.operator.constrained_dofs();
    let mut edges = Vec::new();
    for from_dof in 0..matrix.rows() {
        for entry in matrix.pattern().row_range(from_dof) {
            let to_dof = matrix.pattern().column(entry);
            let stiffness = matrix.values()[entry].abs();
            if to_dof <= from_dof
                || stiffness == 0.0
                || constrained[from_dof]
                || constrained[to_dof]
            {
                continue;
            }
            let left_diag = matrix.get(from_dof, from_dof);
            let right_diag = matrix.get(to_dof, to_dof);
            let diag_scale = (0.5 * (left_diag + right_diag)).abs().max(1.0);
            edges.push(StructuralRecoveryEdge {
                from_dof,
                to_dof,
                component: from_dof % VECTOR_COMPONENT_COUNT,
                hop: to_dof - from_dof,
                edge_length_m: 1.0,
                stiffness_ratio: stiffness / diag_scale,
                basis: StructuralRecoveryBasis::OperatorConnectivity,
            });
        }
    }
    edges
}

fn constrained_recovery_edge_count(summary: &AssemblySummary) -> usize {
    structural_recovery_edges_including_constraints(summary)
        .into_iter()
        .filter(|(left, right)| {
            summary.operator.constrained_dofs()[*left]
                || summary.operator.constrained_dofs()[*right]
        })
        .count()
}

fn structural_recovery_edges_including_constraints(
    summary: &AssemblySummary,
) -> Vec<(usize, usize)> {
    let matrix = summary.operator.operators().stiffness();
    let mut edges = Vec::new();
    for row in 0..matrix.rows() {
        for entry in matrix.pattern().row_range(row) {
            let column = matrix.pattern().column(entry);
            if column > row && matrix.values()[entry] != 0.0 {
                edges.push((row, column));
            }
        }
    }
    edges
}

fn recover_structural_strain(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Result<StructuralStrainRecovery, StructuralFieldRecoveryError> {
    recover_structural_strain_with_plan(summary, displacement, LocalExecutionPlan::serial())
}

fn recover_structural_strain_with_plan(
    summary: &AssemblySummary,
    displacement: &[f64],
    local_execution: LocalExecutionPlan,
) -> Result<StructuralStrainRecovery, StructuralFieldRecoveryError> {
    if !summary.structural_solid_recovery.is_empty() {
        let recovered = recover_solid_fields_with_plan(
            &summary.structural_solid_recovery,
            displacement,
            local_execution,
        )?;
        let basis = if summary.structural_solid_recovery.iter().all(|element| {
            matches!(
                element.geometry,
                crate::assembly::SolidRecoveryGeometry::Tetrahedron4 { .. }
            )
        }) {
            "solid_tetrahedron4_constant_strain"
        } else {
            "solid_tetrahedron10_integration_point_projection"
        };
        return Ok(StructuralStrainRecovery {
            values: recovered.strain,
            stress: Some(recovered.stress),
            strain_energy_density: Some(recovered.strain_energy_density),
            integration_point_strain: recovered.integration_point_strain,
            integration_point_stress: recovered.integration_point_stress,
            element_count: summary.structural_solid_recovery.len().max(1),
            basis,
        });
    }

    let recovery_edges = structural_recovery_edges(summary);
    Ok(StructuralStrainRecovery {
        values: recover_edge_strain(displacement, &recovery_edges),
        stress: None,
        strain_energy_density: None,
        integration_point_strain: Vec::new(),
        integration_point_stress: Vec::new(),
        element_count: recovery_edges.len().max(1),
        basis: recovery_edges
            .first()
            .map(|edge| edge.basis.as_str())
            .unwrap_or(StructuralRecoveryBasis::OperatorConnectivity.as_str()),
    })
}

fn recover_edge_strain(
    displacement: &[f64],
    recovery_edges: &[StructuralRecoveryEdge],
) -> Vec<f64> {
    let element_count = recovery_edges.len().max(1);
    let mut strain = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
    for (element_index, edge) in recovery_edges.iter().enumerate() {
        let edge_strain = edge_strain_tensor(displacement, edge);
        let base = element_index * TENSOR_COMPONENT_COUNT;
        strain[base..base + TENSOR_COMPONENT_COUNT].copy_from_slice(&edge_strain);
    }
    strain
}

fn edge_strain_tensor(displacement: &[f64], edge: &StructuralRecoveryEdge) -> [f64; 6] {
    let length = if edge.edge_length_m.is_finite() && edge.edge_length_m > 0.0 {
        edge.edge_length_m
    } else {
        edge.hop.max(1) as f64
    };
    let from_node = edge.from_dof / VECTOR_COMPONENT_COUNT;
    let to_node = edge.to_dof / VECTOR_COMPONENT_COUNT;
    if from_node == to_node {
        let jump = displacement.get(edge.to_dof).copied().unwrap_or(0.0)
            - displacement.get(edge.from_dof).copied().unwrap_or(0.0);
        let mut tensor = [0.0; TENSOR_COMPONENT_COUNT];
        tensor[edge.component] = jump / length;
        return tensor;
    }

    let du = nodal_displacement(displacement, to_node);
    let u0 = nodal_displacement(displacement, from_node);
    let gradient = [
        (du[0] - u0[0]) / length,
        (du[1] - u0[1]) / length,
        (du[2] - u0[2]) / length,
    ];
    [
        gradient[0],
        gradient[1],
        gradient[2],
        0.5 * (gradient[0] + gradient[1]),
        0.5 * (gradient[1] + gradient[2]),
        0.5 * (gradient[0] + gradient[2]),
    ]
}

fn nodal_displacement(displacement: &[f64], node: usize) -> [f64; VECTOR_COMPONENT_COUNT] {
    let base = node * VECTOR_COMPONENT_COUNT;
    [
        displacement.get(base).copied().unwrap_or(0.0),
        displacement.get(base + 1).copied().unwrap_or(0.0),
        displacement.get(base + 2).copied().unwrap_or(0.0),
    ]
}

fn strain_component_coverage_ratio(
    displacement: &[f64],
    recovery_edges: &[StructuralRecoveryEdge],
) -> f64 {
    let expected_component_count = recovery_edges.len() * TENSOR_COMPONENT_COUNT;
    if expected_component_count == 0 {
        return 0.0;
    }
    let active_component_count = recovery_edges
        .iter()
        .map(|edge| {
            edge_strain_tensor(displacement, edge)
                .iter()
                .filter(|value| value.abs() > 0.0)
                .count()
        })
        .sum::<usize>();
    active_component_count as f64 / expected_component_count as f64
}

#[cfg(test)]
mod tests;
