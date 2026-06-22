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

    let recovery_edges = structural_recovery_edges(summary);
    let element_count = recovery_edges.len().max(1);
    let strain_values = recover_strain(&displacement_values, &recovery_edges);
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
    pub mean_edge_stiffness_ratio: f64,
    pub mean_edge_length_m: f64,
    pub basis: &'static str,
}

pub fn structural_field_recovery_metrics(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> StructuralFieldRecoveryMetrics {
    let recovery_edges = structural_recovery_edges(summary);
    let active_stiffness_edge_count = recovery_edges.len();
    let prep_recovery_edge_count = recovery_edges
        .iter()
        .filter(|edge| edge.basis == StructuralRecoveryBasis::PrepElementConnectivity)
        .count();
    let constrained_edge_count = constrained_recovery_edge_count(summary);
    let mut max_edge_displacement_jump = 0.0_f64;
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

    StructuralFieldRecoveryMetrics {
        active_stiffness_edge_count,
        prep_recovery_edge_count,
        constrained_edge_count,
        recovery_element_count: active_stiffness_edge_count.max(1),
        max_edge_displacement_jump,
        mean_edge_stiffness_ratio,
        mean_edge_length_m,
        basis: recovery_edges
            .first()
            .map(|edge| edge.basis.as_str())
            .unwrap_or(StructuralRecoveryBasis::OperatorConnectivity.as_str()),
    }
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
    PrepElementConnectivity,
    OperatorConnectivity,
}

impl StructuralRecoveryBasis {
    const fn as_str(self) -> &'static str {
        match self {
            Self::PrepElementConnectivity => "prep_element_connectivity",
            Self::OperatorConnectivity => "operator_connectivity",
        }
    }
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

fn recover_strain(displacement: &[f64], recovery_edges: &[StructuralRecoveryEdge]) -> Vec<f64> {
    let element_count = recovery_edges.len().max(1);
    let mut strain = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
    for (element_index, edge) in recovery_edges.iter().enumerate() {
        let jump = displacement.get(edge.to_dof).copied().unwrap_or(0.0)
            - displacement.get(edge.from_dof).copied().unwrap_or(0.0);
        strain[element_index * TENSOR_COMPONENT_COUNT + edge.component] =
            jump / finite_positive_or(edge.edge_length_m, edge.hop.max(1) as f64);
    }
    strain
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
