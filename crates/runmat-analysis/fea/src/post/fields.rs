use runmat_analysis_core::AnalysisField;

use crate::contracts::{FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_VON_MISES};
use crate::{assembly::AssemblySummary, solve::linear::LinearSolveResult};

pub fn recover_result_fields(
    summary: &AssemblySummary,
    solve_result: &LinearSolveResult,
) -> Vec<AnalysisField> {
    if !solve_result.converged {
        return vec![
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_DISPLACEMENT, vec![0], Vec::new()),
            AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![0], Vec::new()),
        ];
    }

    let dof_count = summary.dof_count.max(3);
    let mut displacement_values = solve_result.solution.clone();
    if displacement_values.len() < dof_count {
        displacement_values.resize(dof_count, 0.0);
    }
    if displacement_values.is_empty() {
        displacement_values = vec![0.0; dof_count];
    }

    let max_abs_displacement = displacement_values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let von_mises = (max_abs_displacement * 1.0e11).max(0.0);

    vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            vec![dof_count],
            displacement_values,
        ),
        AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_VON_MISES, vec![1], vec![von_mises]),
    ]
}
