use super::*;
use crate::topology::{
    quadrature::{visit_volume_rule, TETRAHEDRON_FOUR_POINT_RULE},
    AnalysisProjection,
};

impl SampledVolumeFieldTopology {
    pub fn derive(
        mesh: &SolverMeshArtifact,
        rule: VolumeFieldSamplingRule,
    ) -> Result<Self, RecoveredFieldError> {
        let projection = AnalysisProjection::derive(mesh).map_err(|error| match error {
            crate::topology::AnalysisProjectionError::Accounting(error) => {
                RecoveredFieldError::Accounting(error)
            }
            other => invalid(other.to_string()),
        })?;
        Self::derive_from_projection(mesh, &projection, rule)
    }

    /// The layout of an already derived canonical projection of `mesh` (the caller's
    /// projection must be `AnalysisProjection::derive(mesh)`; its digest is recorded).
    pub(crate) fn derive_from_projection(
        mesh: &SolverMeshArtifact,
        projection: &AnalysisProjection,
        rule: VolumeFieldSamplingRule,
    ) -> Result<Self, RecoveredFieldError> {
        if projection.mesh_digest != mesh.canonical_digest {
            return Err(invalid("sampling projection belongs to another mesh"));
        }
        let element_corner_nodes = projection
            .elements
            .iter()
            .map(|element| {
                let mut corners = [0u32; 4];
                for (corner, slot) in corners.iter_mut().enumerate() {
                    let index = *element
                        .node_indices
                        .get(corner)
                        .ok_or_else(|| invalid("sampled element has fewer than four corners"))?;
                    *slot = u32::try_from(index)
                        .map_err(|_| invalid("sampled node index exceeds the layout range"))?;
                }
                Ok(corners)
            })
            .collect::<Result<Vec<_>, RecoveredFieldError>>()?;
        let data = SamplingData {
            schema_version: SCHEMA_VERSION,
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
            element_corner_nodes,
        };
        let value = Self::admit(data, None)?;
        // Every sample of the rule must exist with finite positive geometry.
        value.visit_integration_geometry(projection, |_| {})?;
        Ok(value)
    }
}

/// Sample geometry of every element in canonical order: the named rule over each projected
/// element. Shared by layout derivation and every consumer that needs positions or weights.
pub(super) fn visit_geometry(
    projection: &AnalysisProjection,
    rule: VolumeFieldSamplingRule,
    mut visit: impl FnMut(VolumeFieldSampleGeometry),
) -> Result<(), RecoveredFieldError> {
    let points_per_element = rule.points_per_element();
    let mut points = Vec::with_capacity(points_per_element);
    for (element_index, element) in projection.elements.iter().enumerate() {
        points.clear();
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
                    |point| points.push((point.position_m, point.volume_weight_m3)),
                )
                .map_err(RecoveredFieldError::Accounting)?
                .ok_or_else(|| invalid("invalid four-point volume sampling geometry"))?;
            }
            VolumeFieldSamplingRule::TetrahedronKeastFifteen => {
                projection
                    .visit_element_quadrature(element, |point| {
                        points.push((point.position_m, point.volume_weight_m3))
                    })
                    .map_err(|error| match error {
                        crate::topology::AnalysisProjectionError::Accounting(error) => {
                            RecoveredFieldError::Accounting(error)
                        }
                        other => invalid(other.to_string()),
                    })?;
            }
        }
        if points.len() != points_per_element {
            return Err(invalid("sampled element rule point count differs"));
        }
        for (ordinal, &(position_m, volume_weight_m3)) in points.iter().enumerate() {
            if !volume_weight_m3.is_finite()
                || volume_weight_m3 <= 0.0
                || position_m.iter().any(|coordinate| !coordinate.is_finite())
            {
                return Err(invalid("integration sample physical geometry is invalid"));
            }
            visit(VolumeFieldSampleGeometry {
                index: element_index * points_per_element + ordinal,
                element: element_index,
                ordinal,
                position_m,
                volume_weight_m3,
            });
        }
    }
    Ok(())
}
