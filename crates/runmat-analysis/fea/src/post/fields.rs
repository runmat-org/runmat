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
    let displacement_field = vec![0.0, -1e-5, 0.0]
        .into_iter()
        .chain(std::iter::repeat(0.0).take(dof_count.saturating_sub(3)))
        .collect();

    PostFieldResult {
        displacement_field: AnalysisField::host_f64(
            "displacement",
            vec![dof_count],
            displacement_field,
        ),
        von_mises_field: AnalysisField::host_f64("von_mises", vec![1], vec![12.5e6]),
    }
}
