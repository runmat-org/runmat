//! Requested construction heap beside the original projection-to-observation constructor.
//! The authoritative projection is borrowed; its storage is excluded.
use super::*;
use crate::topology::storage::{add, multiply, slots};

pub fn solid_observation_topology_resident_bytes(
    projection: &AnalysisProjection,
) -> Result<u64, ObservationError> {
    let points = multiply(projection.elements.len(), TETRAHEDRON_FOUR_POINT_RULE.len())
        .map_err(storage_error)?;
    let mut bytes = 0;
    for value in [
        slots::<ObservationNode>(projection.nodes.len()),
        slots::<ObservationElement>(projection.elements.len()),
        slots::<ObservationElement>(points),
        slots::<Vec<usize>>(projection.elements.len()),
        slots::<usize>(points),
        SortedValues::<u64, Vec<usize>>::grouped_construction_resident_bytes(
            projection.nodes.len(),
        ),
    ] {
        bytes = add(bytes, value.map_err(storage_error)?).map_err(storage_error)?;
    }
    for element in &projection.elements {
        let labels = multiply(
            element.region_id.source_topology_id.len(),
            TETRAHEDRON_FOUR_POINT_RULE.len(),
        )
        .map_err(storage_error)?;
        bytes = add(
            bytes,
            u64::try_from(labels)
                .map_err(|_| ObservationError::Limit("observation region bytes"))?,
        )
        .map_err(storage_error)?;
    }
    // Canonical basis and point-coordinate accumulation use fixed stack arrays.
    Ok(bytes)
}

impl<'a> ObservationTopology<'a> {
    pub(in crate::results::observation) fn from_projection_admitted(
        projection: &'a AnalysisProjection,
        maximum: u64,
    ) -> Result<(Self, u64), ObservationError> {
        let bytes = solid_observation_topology_resident_bytes(projection)?;
        let remaining = maximum.checked_sub(bytes).ok_or(ObservationError::Limit(
            "solid observation topology residency",
        ))?;
        Ok((Self::from_projection(projection)?, remaining))
    }
}

#[cfg(test)]
mod tests;
