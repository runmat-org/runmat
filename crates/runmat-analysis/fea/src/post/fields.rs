use runmat_analysis_core::AnalysisField;

use crate::contracts::{
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY, FEA_FIELD_STRUCTURAL_VON_MISES,
};
use crate::operator::apply_k_unconstrained;
use crate::{assembly::AssemblySummary, solve::linear::LinearSolveResult};

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

    let element_count = node_count.saturating_sub(1).max(1);
    let stiffness_scale = effective_stress_scale(summary);
    let strain_values = recover_strain(&displacement_values, element_count);
    let stress_values = recover_stress(&strain_values, stiffness_scale);
    let von_mises_values = recover_von_mises(&stress_values);
    let internal_force = apply_k_unconstrained(&summary.operator, &solve_result.solution);
    let reaction_values = recover_reaction_force(summary, &internal_force);
    let strain_energy = recover_total_strain_energy(&solve_result.solution, &internal_force);

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
    ]
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
    ]
}

fn effective_stress_scale(summary: &AssemblySummary) -> f64 {
    let active_count = summary
        .operator
        .stiffness_diag
        .iter()
        .filter(|value| value.is_finite() && **value > 0.0)
        .count()
        .max(1);
    let mean_stiffness = summary
        .operator
        .stiffness_diag
        .iter()
        .filter(|value| value.is_finite() && **value > 0.0)
        .sum::<f64>()
        / active_count as f64;
    (mean_stiffness * 1.0e3).max(1.0)
}

fn recover_strain(displacement: &[f64], element_count: usize) -> Vec<f64> {
    let mut strain = vec![0.0; element_count * TENSOR_COMPONENT_COUNT];
    for element_index in 0..element_count {
        let base = element_index * VECTOR_COMPONENT_COUNT;
        let next = (element_index + 1) * VECTOR_COMPONENT_COUNT;
        for component in 0..VECTOR_COMPONENT_COUNT {
            let value = displacement.get(next + component).copied().unwrap_or(0.0)
                - displacement.get(base + component).copied().unwrap_or(0.0);
            strain[element_index * TENSOR_COMPONENT_COUNT + component] = value;
        }
    }
    strain
}

fn recover_stress(strain: &[f64], stress_scale: f64) -> Vec<f64> {
    strain.iter().map(|value| value * stress_scale).collect()
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
