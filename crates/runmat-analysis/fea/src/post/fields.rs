use runmat_analysis_core::AnalysisField;

use crate::contracts::{
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
    FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM,
    FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY, FEA_FIELD_STRUCTURAL_VON_MISES,
};
use crate::operator::{apply_k, apply_k_unconstrained};
use crate::{
    assembly::{AssemblySummary, PrepRecoveryEdgeSummary, StructuralMaterialSummary},
    solve::linear::LinearSolveResult,
};

const VECTOR_COMPONENT_COUNT: usize = 3;
const TENSOR_COMPONENT_COUNT: usize = 6;
type LocalTriangleCoordinates = [[f64; 2]; 3];
type LocalFrame = [[f64; 3]; 3];

pub fn recover_result_fields(
    summary: &AssemblySummary,
    solve_result: &LinearSolveResult,
) -> Vec<AnalysisField> {
    if !solve_result.converged {
        return empty_structural_fields();
    }

    let dof_count = summary.dof_count.max(3);
    let mut displacement_values = solve_result.solution.clone();
    if displacement_values.len() < dof_count {
        displacement_values.resize(dof_count, 0.0);
    }
    if displacement_values.is_empty() {
        displacement_values = vec![0.0; dof_count];
    }

    let node_count = dof_count.div_ceil(VECTOR_COMPONENT_COUNT).max(1);
    displacement_values.resize(node_count * VECTOR_COMPONENT_COUNT, 0.0);

    let strain_recovery = recover_structural_strain(summary, &displacement_values);
    let element_count = strain_recovery.element_count;
    let strain_values = strain_recovery.values;
    let stress_values = recover_stress(&strain_values, summary.structural_material);
    let von_mises_values = recover_von_mises(&stress_values);
    let internal_force = apply_k_unconstrained(&summary.operator, &solve_result.solution);
    let reaction_values = recover_reaction_force(summary, &internal_force);
    let strain_energy = recover_total_strain_energy(&solve_result.solution, &internal_force);
    let residual_metrics = recover_residual_metrics(summary, &solve_result.solution);

    vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            vec![node_count, VECTOR_COMPONENT_COUNT],
            displacement_values,
        ),
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
            FEA_FIELD_STRUCTURAL_REACTION_FORCE,
            reaction_shape(summary),
            reaction_values,
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
    ]
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuralFieldRecoveryMetrics {
    pub active_stiffness_edge_count: usize,
    pub prep_recovery_edge_count: usize,
    pub constrained_edge_count: usize,
    pub recovery_element_count: usize,
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
) -> StructuralFieldRecoveryMetrics {
    let recovery_edges = structural_recovery_edges(summary);
    let strain_recovery = recover_structural_strain(summary, displacement);
    let active_stiffness_edge_count = recovery_edges.len();
    let prep_recovery_edge_count = recovery_edges
        .iter()
        .filter(|edge| edge.basis == StructuralRecoveryBasis::PrepElementConnectivity)
        .count();
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
    let element_geometry_node_count = summary
        .prep_coordinates
        .map(|coordinates| coordinates.element_geometry_node_count)
        .unwrap_or(0);
    let element_geometry_edge_count = summary
        .prep_coordinates
        .map(|coordinates| coordinates.element_geometry_edge_count)
        .unwrap_or(0);
    let element_geometry_coverage_ratio = summary
        .prep_coordinates
        .map(|coordinates| coordinates.element_geometry_coverage_ratio)
        .unwrap_or(0.0);

    StructuralFieldRecoveryMetrics {
        active_stiffness_edge_count,
        prep_recovery_edge_count,
        constrained_edge_count,
        recovery_element_count: strain_recovery.element_count,
        max_edge_displacement_jump,
        max_edge_strain_norm,
        mean_edge_stiffness_ratio,
        mean_edge_length_m,
        strain_component_coverage_ratio,
        element_geometry_node_count,
        element_geometry_edge_count,
        element_geometry_coverage_ratio,
        basis: strain_recovery.basis,
    }
}

struct StructuralStrainRecovery {
    values: Vec<f64>,
    element_count: usize,
    basis: &'static str,
}

fn empty_structural_fields() -> Vec<AnalysisField> {
    vec![
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_DISPLACEMENT, vec![0, 3], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![0], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRAIN, vec![0, 6], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRESS, vec![0, 6], Vec::new()),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_REACTION_FORCE, vec![0, 3], Vec::new()),
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
    PrepConstantStrainBMatrix,
    PrepElementConnectivity,
    OperatorConnectivity,
}

impl StructuralRecoveryBasis {
    const fn as_str(self) -> &'static str {
        match self {
            Self::PrepConstantStrainBMatrix => "prep_constant_strain_b_matrix",
            Self::PrepElementConnectivity => "prep_element_connectivity",
            Self::OperatorConnectivity => "operator_connectivity",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StructuralBMatrixElement {
    nodes: [usize; 3],
    coordinates_m: [[f64; 3]; 3],
}

fn structural_recovery_edges(summary: &AssemblySummary) -> Vec<StructuralRecoveryEdge> {
    let prep_edges = prep_structural_recovery_edges(summary);
    if !prep_edges.is_empty() {
        return prep_edges;
    }

    summary
        .operator
        .stiffness_upper
        .iter()
        .enumerate()
        .filter_map(|(from_dof, stiffness)| {
            let to_dof = from_dof + 1;
            if to_dof >= summary.dof_count
                || *stiffness <= 0.0
                || summary
                    .operator
                    .constrained
                    .get(from_dof)
                    .copied()
                    .unwrap_or(false)
                || summary
                    .operator
                    .constrained
                    .get(to_dof)
                    .copied()
                    .unwrap_or(false)
            {
                return None;
            }
            let left_diag = summary
                .operator
                .stiffness_diag
                .get(from_dof)
                .copied()
                .unwrap_or(0.0);
            let right_diag = summary
                .operator
                .stiffness_diag
                .get(to_dof)
                .copied()
                .unwrap_or(0.0);
            let diag_scale = (0.5 * (left_diag + right_diag)).abs().max(1.0);
            Some(StructuralRecoveryEdge {
                from_dof,
                to_dof,
                component: from_dof % VECTOR_COMPONENT_COUNT,
                hop: 1,
                edge_length_m: 1.0,
                stiffness_ratio: (*stiffness / diag_scale).abs(),
                basis: StructuralRecoveryBasis::OperatorConnectivity,
            })
        })
        .collect()
}

fn prep_structural_recovery_edges(summary: &AssemblySummary) -> Vec<StructuralRecoveryEdge> {
    summary
        .prep_recovery_edges
        .iter()
        .filter_map(|edge| prep_recovery_edge(summary, *edge))
        .collect()
}

fn prep_recovery_edge(
    summary: &AssemblySummary,
    edge: PrepRecoveryEdgeSummary,
) -> Option<StructuralRecoveryEdge> {
    let from_dof = edge.from_dof.min(edge.to_dof);
    let to_dof = edge.from_dof.max(edge.to_dof);
    if from_dof == to_dof
        || to_dof >= summary.dof_count
        || summary
            .operator
            .constrained
            .get(from_dof)
            .copied()
            .unwrap_or(false)
        || summary
            .operator
            .constrained
            .get(to_dof)
            .copied()
            .unwrap_or(false)
    {
        return None;
    }

    let left_diag = summary
        .operator
        .stiffness_diag
        .get(from_dof)
        .copied()
        .unwrap_or(0.0);
    let right_diag = summary
        .operator
        .stiffness_diag
        .get(to_dof)
        .copied()
        .unwrap_or(0.0);
    let diag_scale = (0.5 * (left_diag + right_diag)).abs().max(1.0);
    let hop = to_dof.abs_diff(from_dof).max(1);
    let family_scale = match edge.element_family_index {
        0 => 0.95,
        1 => 1.0,
        2 => 1.05,
        3 => 1.1,
        _ => 0.9,
    };
    Some(StructuralRecoveryEdge {
        from_dof,
        to_dof,
        component: from_dof % VECTOR_COMPONENT_COUNT,
        hop,
        edge_length_m: finite_positive_or(edge.edge_length_m, hop as f64),
        stiffness_ratio: family_scale / (hop as f64 * diag_scale.sqrt().max(1.0)),
        basis: StructuralRecoveryBasis::PrepElementConnectivity,
    })
}

fn finite_positive_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn constrained_recovery_edge_count(summary: &AssemblySummary) -> usize {
    let prep_constrained = summary
        .prep_recovery_edges
        .iter()
        .filter(|edge| {
            let from_dof = edge.from_dof.min(edge.to_dof);
            let to_dof = edge.from_dof.max(edge.to_dof);
            to_dof < summary.dof_count
                && (summary
                    .operator
                    .constrained
                    .get(from_dof)
                    .copied()
                    .unwrap_or(false)
                    || summary
                        .operator
                        .constrained
                        .get(to_dof)
                        .copied()
                        .unwrap_or(false))
        })
        .count();
    if prep_constrained > 0 || !summary.prep_recovery_edges.is_empty() {
        return prep_constrained;
    }

    summary
        .operator
        .stiffness_upper
        .iter()
        .enumerate()
        .filter(|(from_dof, stiffness)| {
            let to_dof = from_dof + 1;
            **stiffness > 0.0
                && to_dof < summary.dof_count
                && (summary
                    .operator
                    .constrained
                    .get(*from_dof)
                    .copied()
                    .unwrap_or(false)
                    || summary
                        .operator
                        .constrained
                        .get(to_dof)
                        .copied()
                        .unwrap_or(false))
        })
        .count()
}

fn recover_structural_strain(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> StructuralStrainRecovery {
    let b_matrix_elements = prep_b_matrix_recovery_elements(summary, displacement);
    if !b_matrix_elements.is_empty() {
        return StructuralStrainRecovery {
            values: recover_b_matrix_strain(displacement, &b_matrix_elements),
            element_count: b_matrix_elements.len().max(1),
            basis: StructuralRecoveryBasis::PrepConstantStrainBMatrix.as_str(),
        };
    }

    let recovery_edges = structural_recovery_edges(summary);
    StructuralStrainRecovery {
        values: recover_edge_strain(displacement, &recovery_edges),
        element_count: recovery_edges.len().max(1),
        basis: recovery_edges
            .first()
            .map(|edge| edge.basis.as_str())
            .unwrap_or(StructuralRecoveryBasis::OperatorConnectivity.as_str()),
    }
}

fn prep_b_matrix_recovery_elements(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Vec<StructuralBMatrixElement> {
    let Some(prep_coordinates) = summary.prep_coordinates.as_ref() else {
        return Vec::new();
    };
    if prep_coordinates.element_geometry_coverage_ratio <= 0.0
        || prep_coordinates.element_geometry_node_count < 3
        || prep_coordinates.reference_element_area_m2 <= 0.0
        || !prep_coordinates.reference_element_area_m2.is_finite()
        || !reference_coordinates_are_valid(prep_coordinates.reference_element_coordinates_m)
    {
        return Vec::new();
    }

    let node_count = displacement.len().div_ceil(VECTOR_COMPONENT_COUNT);
    let mut nodes = summary
        .prep_recovery_edges
        .iter()
        .flat_map(|edge| {
            [
                edge.from_dof / VECTOR_COMPONENT_COUNT,
                edge.to_dof / VECTOR_COMPONENT_COUNT,
            ]
        })
        .filter(|node| *node < node_count && node_has_unconstrained_dof(summary, *node))
        .collect::<Vec<_>>();
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.len() < 3 {
        nodes = (0..node_count)
            .filter(|node| node_has_unconstrained_dof(summary, *node))
            .take(3)
            .collect();
    }

    nodes
        .chunks_exact(3)
        .map(|chunk| StructuralBMatrixElement {
            nodes: [chunk[0], chunk[1], chunk[2]],
            coordinates_m: prep_coordinates.reference_element_coordinates_m,
        })
        .collect()
}

fn reference_coordinates_are_valid(coordinates: [[f64; 3]; 3]) -> bool {
    coordinates.iter().flatten().all(|value| value.is_finite())
        && triangle_area_3d_m2(coordinates) > 0.0
}

fn node_has_unconstrained_dof(summary: &AssemblySummary, node: usize) -> bool {
    let base = node * VECTOR_COMPONENT_COUNT;
    (0..VECTOR_COMPONENT_COUNT).any(|component| {
        !summary
            .operator
            .constrained
            .get(base + component)
            .copied()
            .unwrap_or(false)
    })
}

fn recover_b_matrix_strain(
    displacement: &[f64],
    elements: &[StructuralBMatrixElement],
) -> Vec<f64> {
    let mut strain = vec![0.0; elements.len().max(1) * TENSOR_COMPONENT_COUNT];
    for (element_index, element) in elements.iter().enumerate() {
        let edge_strain = b_matrix_strain_tensor(displacement, element);
        let base = element_index * TENSOR_COMPONENT_COUNT;
        strain[base..base + TENSOR_COMPONENT_COUNT].copy_from_slice(&edge_strain);
    }
    strain
}

fn b_matrix_strain_tensor(displacement: &[f64], element: &StructuralBMatrixElement) -> [f64; 6] {
    let Some((local_coordinates, local_basis)) = local_triangle_coordinates(element.coordinates_m)
    else {
        return [0.0; TENSOR_COMPONENT_COUNT];
    };
    let denominator = triangle_signed_area2(local_coordinates);
    if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
        return [0.0; TENSOR_COMPONENT_COUNT];
    }

    let mut local_displacement = [[0.0_f64; 3]; 3];
    for (i, node) in element.nodes.iter().copied().enumerate() {
        let displacement_vector = nodal_displacement(displacement, node);
        local_displacement[i] = [
            dot3(displacement_vector, local_basis[0]),
            dot3(displacement_vector, local_basis[1]),
            dot3(displacement_vector, local_basis[2]),
        ];
    }

    let b = [
        local_coordinates[1][1] - local_coordinates[2][1],
        local_coordinates[2][1] - local_coordinates[0][1],
        local_coordinates[0][1] - local_coordinates[1][1],
    ];
    let c = [
        local_coordinates[2][0] - local_coordinates[1][0],
        local_coordinates[0][0] - local_coordinates[2][0],
        local_coordinates[1][0] - local_coordinates[0][0],
    ];

    let mut du_dx = 0.0_f64;
    let mut du_dy = 0.0_f64;
    let mut dv_dx = 0.0_f64;
    let mut dv_dy = 0.0_f64;
    let mut dw_dx = 0.0_f64;
    let mut dw_dy = 0.0_f64;
    for i in 0..3 {
        du_dx += b[i] * local_displacement[i][0];
        du_dy += c[i] * local_displacement[i][0];
        dv_dx += b[i] * local_displacement[i][1];
        dv_dy += c[i] * local_displacement[i][1];
        dw_dx += b[i] * local_displacement[i][2];
        dw_dy += c[i] * local_displacement[i][2];
    }

    [
        du_dx / denominator,
        dv_dy / denominator,
        0.0,
        (du_dy + dv_dx) / denominator,
        dw_dy / denominator,
        dw_dx / denominator,
    ]
}

fn local_triangle_coordinates(
    coordinates: [[f64; 3]; 3],
) -> Option<(LocalTriangleCoordinates, LocalFrame)> {
    let origin = coordinates[0];
    let edge01 = sub3(coordinates[1], origin);
    let edge02 = sub3(coordinates[2], origin);
    let e1 = normalize3(edge01)?;
    let normal = normalize3(cross3(edge01, edge02))?;
    let e2 = normalize3(cross3(normal, e1))?;
    let local = coordinates.map(|point| {
        let relative = sub3(point, origin);
        [dot3(relative, e1), dot3(relative, e2)]
    });
    Some((local, [e1, e2, normal]))
}

fn triangle_signed_area2(coordinates: [[f64; 2]; 3]) -> f64 {
    coordinates[0][0] * (coordinates[1][1] - coordinates[2][1])
        + coordinates[1][0] * (coordinates[2][1] - coordinates[0][1])
        + coordinates[2][0] * (coordinates[0][1] - coordinates[1][1])
}

fn triangle_area_3d_m2(coordinates: [[f64; 3]; 3]) -> f64 {
    let edge01 = sub3(coordinates[1], coordinates[0]);
    let edge02 = sub3(coordinates[2], coordinates[0]);
    0.5 * norm3(cross3(edge01, edge02))
}

fn sub3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm3(value: [f64; 3]) -> f64 {
    dot3(value, value).sqrt()
}

fn normalize3(value: [f64; 3]) -> Option<[f64; 3]> {
    let norm = norm3(value);
    (norm.is_finite() && norm > f64::EPSILON).then_some([
        value[0] / norm,
        value[1] / norm,
        value[2] / norm,
    ])
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
    let length = finite_positive_or(edge.edge_length_m, edge.hop.max(1) as f64);
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

fn recover_stress(strain: &[f64], material: StructuralMaterialSummary) -> Vec<f64> {
    let lambda = material.lame_lambda_pa.max(0.0);
    let mu = material.shear_modulus_pa.max(0.0);
    let mut stress = vec![0.0; strain.len()];
    for (element_index, strain_tensor) in strain.chunks_exact(TENSOR_COMPONENT_COUNT).enumerate() {
        let trace = strain_tensor[0] + strain_tensor[1] + strain_tensor[2];
        let base = element_index * TENSOR_COMPONENT_COUNT;
        stress[base] = lambda * trace + 2.0 * mu * strain_tensor[0];
        stress[base + 1] = lambda * trace + 2.0 * mu * strain_tensor[1];
        stress[base + 2] = lambda * trace + 2.0 * mu * strain_tensor[2];
        stress[base + 3] = mu * strain_tensor[3];
        stress[base + 4] = mu * strain_tensor[4];
        stress[base + 5] = mu * strain_tensor[5];
    }
    stress
}

fn recover_von_mises(stress: &[f64]) -> Vec<f64> {
    stress
        .chunks_exact(TENSOR_COMPONENT_COUNT)
        .map(|tensor| {
            let sxx = tensor[0];
            let syy = tensor[1];
            let szz = tensor[2];
            let txy = tensor[3];
            let tyz = tensor[4];
            let txz = tensor[5];
            (0.5 * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2))
                + 3.0 * (txy.powi(2) + tyz.powi(2) + txz.powi(2)))
            .sqrt()
        })
        .collect()
}

fn recover_reaction_force(summary: &AssemblySummary, internal_force: &[f64]) -> Vec<f64> {
    let mut reactions = vec![0.0; reaction_shape(summary).iter().product()];
    let mut constrained_ordinal = 0usize;
    for (dof, is_constrained) in summary.operator.constrained.iter().enumerate() {
        if !*is_constrained {
            continue;
        }
        let row = constrained_ordinal / VECTOR_COMPONENT_COUNT;
        let component = constrained_ordinal % VECTOR_COMPONENT_COUNT;
        let rhs = summary.operator.rhs.get(dof).copied().unwrap_or(0.0);
        let internal = internal_force.get(dof).copied().unwrap_or(0.0);
        reactions[row * VECTOR_COMPONENT_COUNT + component] = internal - rhs;
        constrained_ordinal += 1;
    }
    reactions
}

fn reaction_shape(summary: &AssemblySummary) -> Vec<usize> {
    let rows = summary
        .constrained_dof_count
        .div_ceil(VECTOR_COMPONENT_COUNT);
    vec![rows, VECTOR_COMPONENT_COUNT]
}

fn recover_total_strain_energy(displacement: &[f64], internal_force: &[f64]) -> f64 {
    0.5 * displacement
        .iter()
        .zip(internal_force.iter())
        .map(|(u, force)| u * force)
        .sum::<f64>()
}

struct StructuralResidualMetrics {
    normalized_residual_norm: f64,
    equation_scale: f64,
}

fn recover_residual_metrics(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> StructuralResidualMetrics {
    let mut solution = displacement.to_vec();
    solution.resize(summary.dof_count, 0.0);
    let applied = apply_k(&summary.operator, &solution);
    let residual_norm = applied
        .iter()
        .zip(summary.operator.rhs.iter())
        .map(|(lhs, rhs)| {
            let residual = lhs - rhs;
            residual * residual
        })
        .sum::<f64>()
        .sqrt();
    let equation_scale = summary
        .operator
        .rhs
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    StructuralResidualMetrics {
        normalized_residual_norm: residual_norm / equation_scale,
        equation_scale,
    }
}
