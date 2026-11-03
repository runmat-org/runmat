//! Physical thermal snapshots: each step's values stored under the field id and layout of the
//! published snapshot definition (`physics::thermal::history_fields`), the one authority.

use runmat_analysis_core::AnalysisField;

use crate::{contracts::FeaRunError, physics::thermal::history_fields};

/// Snapshots of history `snapshot` (see `history_fields::snapshot_entities`), one per step.
pub(super) fn snapshots(
    values: Vec<Vec<f64>>,
    snapshot: usize,
) -> Result<Vec<AnalysisField>, FeaRunError> {
    values
        .into_iter()
        .enumerate()
        .map(|(step, values)| {
            let definition = history_fields::snapshot_definition(snapshot, step)
                .map_err(|error| FeaRunError::Solve(error.to_string()))?;
            let layout = history_fields::storage_layout(&definition, step);
            let components = layout
                .component_count()
                .ok_or_else(|| FeaRunError::Solve("thermal snapshot value shape".into()))?;
            let shape = if components == 1 {
                vec![values.len()]
            } else {
                vec![values.len() / components, components]
            };
            Ok(AnalysisField::host_f64(definition.field_id().to_string(), layout, shape, values))
        })
        .collect()
}
