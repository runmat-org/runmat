use super::*;
use crate::topology::{
    quadrature::{visit_volume_rule, TETRAHEDRON_FOUR_POINT_RULE},
    AnalysisProjection, TetrahedralEdgeSpace,
};

impl SampledVolumeFieldTopology {
    pub fn derive(
        mesh: &SolverMeshArtifact,
        rule: VolumeFieldSamplingRule,
    ) -> Result<Self, RecoveredFieldError> {
        let projection =
            AnalysisProjection::derive(mesh).map_err(|error| invalid(error.to_string()))?;
        let edge_space = TetrahedralEdgeSpace::derive(&projection)
            .map_err(|error| invalid(error.to_string()))?;
        let oriented_edges = edge_space
            .edges()
            .iter()
            .map(|edge| {
                let node_identities = edge.node_identities;
                Ok(SampledOrientedEdge {
                    identity: stable_identity("oriented-edge", &node_identities)?,
                    node_identities,
                })
            })
            .collect::<Result<Vec<_>, RecoveredFieldError>>()?;
        let mut integration_samples = Vec::new();
        for element in &projection.elements {
            let mut points = Vec::new();
            match rule {
                VolumeFieldSamplingRule::TetrahedronFourPoint => {
                    let coordinates = element
                        .node_indices
                        .iter()
                        .map(|&index| projection.nodes[index].coordinates_m.canonical_value())
                        .collect::<Vec<_>>();
                    visit_volume_rule(
                        element.order,
                        &coordinates,
                        TETRAHEDRON_FOUR_POINT_RULE.map(|barycentric| (barycentric, 1.0 / 24.0)),
                        |point| points.push(point),
                    )
                    .ok_or_else(|| invalid("invalid four-point volume sampling geometry"))?;
                }
                VolumeFieldSamplingRule::TetrahedronKeastFifteen => {
                    projection
                        .visit_element_quadrature(element, |point| points.push(point))
                        .map_err(|error| invalid(error.to_string()))?;
                }
            }
            for (ordinal, point) in points.into_iter().enumerate() {
                integration_samples.push(VolumeFieldIntegrationSample {
                    identity: stable_identity(
                        "integration-sample",
                        &(rule, element.stable_identity, ordinal),
                    )?,
                    element_identity: element.stable_identity,
                    ordinal,
                    position_m: point.position_m,
                    volume_weight_m3: point.volume_weight_m3,
                });
            }
        }
        let data = SamplingData {
            schema_version: 1,
            mesh_digest: mesh.canonical_digest,
            projection_digest: projection.projection_digest,
            rule,
            node_identities: projection
                .nodes
                .iter()
                .map(|node| StableDigest::from_bytes(node.stable_identity.0))
                .collect(),
            element_identities: projection
                .elements
                .iter()
                .map(|element| element.stable_identity)
                .collect(),
            element_corner_identities: projection
                .elements
                .iter()
                .map(|element| {
                    std::array::from_fn(|corner| {
                        StableDigest::from_bytes(
                            projection.nodes[element.node_indices[corner]]
                                .stable_identity
                                .0,
                        )
                    })
                })
                .collect(),
            oriented_edges,
            integration_samples,
        };
        let value = Self {
            sampling_digest: Digest::sha256(encode("sampling-layout", &data)?),
            data,
        };
        value.validate()?;
        Ok(value)
    }
}
