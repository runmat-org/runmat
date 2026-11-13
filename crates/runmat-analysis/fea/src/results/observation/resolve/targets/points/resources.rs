//! Returned point inventories and identity scratch beside the shared interpolation owner.
use super::*;

pub(super) fn interpolated_points_resident_bytes(
    projection: &AnalysisProjection,
    point_count: usize,
) -> Result<u64, ObservationError> {
    use crate::topology::storage::{add, slots};
    let per_point = projection.volume_point_location_resident_bytes()?;
    let count = projection
        .elements
        .iter()
        .map(|e| e.node_indices.len())
        .max()
        .unwrap_or(0);
    let identity =
        crate::results::observation::contract::resolved_point_identity_resident_bytes(count)?;
    let output = per_point
        .checked_mul(point_count as u64)
        .ok_or(ObservationError::Limit("point target inventories"))?;
    add(
        add(
            slots::<ResolvedObservationPoint>(point_count)
                .map_err(|e| ObservationError::Invalid(e.to_string()))?,
            output,
        )
        .map_err(|e| ObservationError::Invalid(e.to_string()))?,
        identity,
    )
    .map_err(|e| ObservationError::Invalid(e.to_string()))
}

#[cfg(test)]
mod tests;
