use runmat_meshing_core::{MeshingCancellationSignal, SolverMeshArtifact};

use super::{
    boundary::project_boundary_face,
    contract::{
        AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy, ProjectedElement,
        ProjectedNode,
    },
    element::tetrahedron_geometry,
    identity::projection_digest,
    quadrature::visit_volume_quadrature,
    storage::{check_cancelled, reserve, SortedValues},
};

impl AnalysisProjection {
    pub fn derive(mesh: &SolverMeshArtifact) -> Result<Self, AnalysisProjectionError> {
        Self::derive_with_policy(mesh, AnalysisProjectionPolicy::default())
    }

    pub fn derive_with_policy(
        mesh: &SolverMeshArtifact,
        policy: AnalysisProjectionPolicy,
    ) -> Result<Self, AnalysisProjectionError> {
        check_cancelled()?;
        Self::derivation_resident_bytes_estimate(mesh)?;
        check_cancelled()?;
        mesh.validate_canonical_with_cancellation(&ProjectionCancellation)
            .map_err(|error| {
                if crate::is_fea_cancelled() {
                    AnalysisProjectionError::Cancelled
                } else {
                    AnalysisProjectionError::InvalidMesh(error.to_string())
                }
            })?;
        check_cancelled()?;
        let mut nodes = reserve(mesh.topology.nodes.len())?;
        for node in &mesh.topology.nodes {
            check_cancelled()?;
            nodes.push(ProjectedNode {
                node_id: node.node_id,
                stable_identity: node.stable_identity,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(node.coordinates_m.canonical_value()),
            });
        }
        let node_by_id = SortedValues::derive(
            nodes
                .iter()
                .enumerate()
                .map(|(index, node)| (node.node_id, index)),
            nodes.len(),
        )?;
        let element_by_id = SortedValues::derive(
            mesh.topology
                .volume_elements
                .iter()
                .enumerate()
                .map(|(index, element)| (element.element_id, index)),
            mesh.topology.volume_elements.len(),
        )?;
        let mut elements = reserve(mesh.topology.volume_elements.len())?;
        for element in &mesh.topology.volume_elements {
            check_cancelled()?;
            let mut node_indices = reserve(element.node_ids.len())?;
            for node_id in &element.node_ids {
                node_indices.push(*node_by_id.get(node_id).ok_or(
                    AnalysisProjectionError::UnknownNode {
                        entity_kind: "volume element",
                        entity_id: element.element_id,
                        node_id: *node_id,
                    },
                )?);
            }
            let corners =
                node_indices
                    .get(..4)
                    .ok_or(AnalysisProjectionError::InvalidElementGeometry {
                        element_id: element.element_id,
                    })?;
            let coordinates = [
                nodes[corners[0]].coordinates_m.canonical_value(),
                nodes[corners[1]].coordinates_m.canonical_value(),
                nodes[corners[2]].coordinates_m.canonical_value(),
                nodes[corners[3]].coordinates_m.canonical_value(),
            ];
            tetrahedron_geometry(coordinates).ok_or(
                AnalysisProjectionError::InvalidElementGeometry {
                    element_id: element.element_id,
                },
            )?;
            let mut element_coordinates = [[0.0; 3]; 10];
            for (slot, index) in element_coordinates.iter_mut().zip(&node_indices) {
                *slot = nodes[*index].coordinates_m.canonical_value();
            }
            let mut signed_volume_m3 = 0.0;
            let mut first_moment_m4 = [0.0; 3];
            visit_volume_quadrature(
                element.order,
                &element_coordinates[..node_indices.len()],
                |point| {
                    signed_volume_m3 += point.volume_weight_m3;
                    for (first_moment, position) in first_moment_m4.iter_mut().zip(point.position_m)
                    {
                        *first_moment += position * point.volume_weight_m3;
                    }
                },
            )
            .map_err(AnalysisProjectionError::Accounting)?
            .ok_or(AnalysisProjectionError::InvalidElementGeometry {
                element_id: element.element_id,
            })?;
            let centroid_m = first_moment_m4.map(|value| value / signed_volume_m3);
            let characteristic_length_m = (6.0 * signed_volume_m3).cbrt();
            elements.push(ProjectedElement {
                element_id: element.element_id,
                stable_identity: element.stable_identity,
                order: element.order,
                node_indices,
                region_id: element.region_id.clone(),
                provenance: element.provenance.clone(),
                signed_volume_m3,
                centroid_m,
                characteristic_length_m,
            });
        }
        let mut boundary_faces = reserve(mesh.topology.boundary_faces.len())?;
        for face in &mesh.topology.boundary_faces {
            check_cancelled()?;
            let mut node_indices = reserve(face.node_ids.len())?;
            for node_id in &face.node_ids {
                node_indices.push(*node_by_id.get(node_id).ok_or(
                    AnalysisProjectionError::UnknownNode {
                        entity_kind: "boundary face",
                        entity_id: face.face_id,
                        node_id: *node_id,
                    },
                )?);
            }
            let mut adjacent_element_indices = reserve(face.adjacent_volume_element_ids.len())?;
            for element_id in &face.adjacent_volume_element_ids {
                adjacent_element_indices.push(*element_by_id.get(element_id).ok_or(
                    AnalysisProjectionError::UnknownAdjacentElement {
                        face_id: face.face_id,
                        element_id: *element_id,
                    },
                )?);
            }
            boundary_faces.push(project_boundary_face(
                face,
                node_indices,
                adjacent_element_indices,
                &nodes,
                &elements,
            )?);
        }
        let bounds_m = std::array::from_fn(|axis| {
            nodes
                .iter()
                .fold([f64::INFINITY, f64::NEG_INFINITY], |[min, max], node| {
                    [
                        min.min(node.coordinates_m.canonical_value()[axis]),
                        max.max(node.coordinates_m.canonical_value()[axis]),
                    ]
                })
        });
        let characteristic_length_m = bounds_m
            .iter()
            .map(|[min, max]| (max - min).powi(2))
            .sum::<f64>()
            .sqrt();
        let node_neighbors = super::neighbors::derive(&elements, nodes.len())?;
        check_cancelled()?;
        Ok(Self {
            schema_version: 1,
            mesh_digest: mesh.canonical_digest,
            projection_digest: projection_digest(mesh.canonical_digest, policy)?,
            policy,
            nodes,
            elements,
            boundary_faces,
            node_neighbors,
            bounds_m,
            characteristic_length_m,
            node_by_id,
            element_by_id,
        })
    }
}

pub(crate) struct ProjectionCancellation;
impl MeshingCancellationSignal for ProjectionCancellation {
    fn is_cancelled(&self) -> bool {
        crate::is_fea_cancelled()
    }
}
