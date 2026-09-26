//! Physical thermal snapshot shapes for nodal scalars and element vectors.

use runmat_analysis_core::AnalysisField;

pub(super) fn scalar_snapshots(
    values: Vec<Vec<f64>>,
    field_id: fn(usize) -> String,
) -> Vec<AnalysisField> {
    values
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            let len = values.len();
            AnalysisField::host_f64(field_id(index), vec![len], values)
        })
        .collect()
}

pub(super) fn vector_snapshots(
    values: Vec<Vec<f64>>,
    field_id: fn(usize) -> String,
) -> Vec<AnalysisField> {
    values
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            let count = values.len() / 3;
            AnalysisField::host_f64(field_id(index), vec![count, 3], values)
        })
        .collect()
}
