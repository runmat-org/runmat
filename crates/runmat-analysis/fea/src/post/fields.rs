use serde::{Deserialize, Serialize};

use runmat_analysis_core::AnalysisField;

use crate::{assembly::AssemblySummary, solve::linear::LinearSolveResult};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostFieldResult {
    pub displacement_field: AnalysisField,
    pub von_mises_field: AnalysisField,
}

pub fn recover_result_fields(
    summary: &AssemblySummary,
    solve_result: &LinearSolveResult,
) -> PostFieldResult {
    if !solve_result.converged {
        return PostFieldResult {
            displacement_field: AnalysisField::host_f64("displacement", vec![0], Vec::new()),
            von_mises_field: AnalysisField::host_f64("von_mises", vec![0], Vec::new()),
        };
    }

    let dof_count = summary.dof_count.max(3);
    let mut displacement_field = solve_result.solution.clone();
    if displacement_field.len() < dof_count {
        displacement_field.resize(dof_count, 0.0);
    }
    if displacement_field.is_empty() {
        displacement_field = vec![0.0; dof_count];
    }

    let max_abs_displacement = displacement_field
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let von_mises = (max_abs_displacement * 1.0e11).max(0.0);

    PostFieldResult {
        displacement_field: AnalysisField::host_f64(
            "displacement",
            vec![dof_count],
            displacement_field,
        ),
        von_mises_field: AnalysisField::host_f64("von_mises", vec![1], vec![von_mises]),
    }
}
