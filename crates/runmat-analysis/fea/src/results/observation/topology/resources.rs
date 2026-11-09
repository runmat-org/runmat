//! Requested construction heap beside the original projection-to-observation constructor.
//! The authoritative projection is borrowed; its storage is excluded.
use super::*;
use crate::topology::storage::{add, multiply, slots};

pub fn solid_observation_topology_resident_bytes(
    projection: &AnalysisProjection,
) -> Result<u64, ObservationError> {
    topology_resident_bytes(
        projection.nodes.len(),
        projection
            .elements
            .iter()
            .map(|element| element.region_id.source_topology_id.len()),
    )
}

/// The same law read from the raw mesh before any projection exists: a projection carries
/// at most the mesh's nodes and exactly its volume elements with their region labels.
pub fn solid_observation_topology_resident_bytes_for_mesh(
    mesh: &SolverMeshArtifact,
) -> Result<u64, ObservationError> {
    topology_resident_bytes(
        mesh.topology.nodes.len(),
        mesh.topology
            .volume_elements
            .iter()
            .map(|element| element.region_id.source_topology_id.len()),
    )
}

fn topology_resident_bytes(
    node_count: usize,
    element_labels: impl ExactSizeIterator<Item = usize>,
) -> Result<u64, ObservationError> {
    let element_count = element_labels.len();
    let points =
        multiply(element_count, TETRAHEDRON_FOUR_POINT_RULE.len()).map_err(storage_error)?;
    let mut bytes = 0;
    for value in [
        slots::<ObservationNode>(node_count),
        slots::<ObservationElement>(element_count),
        slots::<ObservationElement>(points),
        slots::<Vec<usize>>(element_count),
        slots::<usize>(points),
        SortedValues::<u64, Vec<usize>>::grouped_construction_resident_bytes(node_count),
    ] {
        bytes = add(bytes, value.map_err(storage_error)?).map_err(storage_error)?;
    }
    for label in element_labels {
        let labels = multiply(label, TETRAHEDRON_FOUR_POINT_RULE.len()).map_err(storage_error)?;
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
