//! Exact sampled storage authority; no structural topology or field interpolation is invented.
use super::*;

impl ObservationTopology {
    pub fn from_sampled(
        sampling: &SampledVolumeFieldTopology,
        mesh: &SolverMeshArtifact,
    ) -> Result<Self, ObservationError> {
        sampling.validate_for_mesh(mesh).map_err(invalid)?;
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let element_indices = projection
            .elements
            .iter()
            .enumerate()
            .map(|(index, element)| (element.stable_identity, index))
            .collect::<BTreeMap<_, _>>();
        let node_coordinates = projection
            .nodes
            .iter()
            .map(|node| (node.stable_identity, node.coordinates_m))
            .collect::<BTreeMap<_, _>>();
        let mut integration_points_by_parent = vec![Vec::new(); projection.elements.len()];
        let volume_integration_points = sampling
            .integration_samples()
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let parent = *element_indices.get(&sample.element_identity).ok_or(
                    ObservationError::Identity(
                        "sampled observation parent is absent from canonical projection",
                    ),
                )?;
                integration_points_by_parent[parent].push(index);
                Ok(ObservationElement {
                    identity: sample.identity,
                    region_id: projection.elements[parent]
                        .region_id
                        .source_topology_id
                        .clone(),
                    centroid_m: sample.position_m,
                })
            })
            .collect::<Result<Vec<_>, ObservationError>>()?;
        let oriented_edges = sampling
            .oriented_edges()
            .iter()
            .map(|edge| {
                let coordinates = edge
                    .node_identities
                    .map(|node| node_coordinates.get(&node).copied());
                let [Some(left), Some(right)] = coordinates else {
                    return Err(ObservationError::Identity(
                        "sampled observation edge endpoint is absent from canonical projection",
                    ));
                };
                Ok(ObservationElement {
                    identity: edge.identity,
                    region_id: String::new(),
                    // Enumeration geometry only: circulation is not a midpoint field value.
                    centroid_m: std::array::from_fn(|axis| (left[axis] + right[axis]) * 0.5),
                })
            })
            .collect::<Result<Vec<_>, ObservationError>>()?;
        Ok(Self {
            identity: None,
            topology_digest: sampling
                .identity()
                .map_err(invalid)?
                .discretization_digest(),
            nodes: projection
                .nodes
                .iter()
                .map(|node| ObservationNode {
                    identity: node.stable_identity,
                    coordinates_m: node.coordinates_m,
                })
                .collect(),
            volume_elements: projection
                .elements
                .iter()
                .map(|element| ObservationElement {
                    identity: element.stable_identity,
                    region_id: element.region_id.source_topology_id.clone(),
                    centroid_m: element.centroid_m,
                })
                .collect(),
            volume_integration_points,
            oriented_edges,
            beam_elements: Vec::new(),
            shell_elements: Vec::new(),
            node_selectors: projection
                .nodes
                .iter()
                .enumerate()
                .map(|(index, node)| (node.node_id, vec![index]))
                .collect(),
            solid_projection: Some(projection),
            sampling: Some(sampling.clone()),
            integration_points_by_parent,
        })
    }
}

fn invalid(error: impl std::fmt::Display) -> ObservationError {
    ObservationError::Invalid(error.to_string())
}
