use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use super::{
    boundary::project_boundary_face,
    contract::{
        AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy, ProjectedElement,
        ProjectedNode,
    },
    element::tetrahedron_geometry,
    identity::projection_digest,
    quadrature::visit_volume_quadrature,
};

impl AnalysisProjection {
    pub fn derive(mesh: &SolverMeshArtifact) -> Result<Self, AnalysisProjectionError> {
        Self::derive_with_policy(mesh, AnalysisProjectionPolicy::default())
    }

    pub fn derive_with_policy(
        mesh: &SolverMeshArtifact,
        policy: AnalysisProjectionPolicy,
    ) -> Result<Self, AnalysisProjectionError> {
        mesh.validate_canonical()
            .map_err(|error| AnalysisProjectionError::InvalidMesh(error.to_string()))?;
        let nodes = mesh
            .topology
            .nodes
            .iter()
            .map(|node| ProjectedNode {
                node_id: node.node_id,
                stable_identity: node.stable_identity,
                coordinates_m: node.coordinates_m,
            })
            .collect::<Vec<_>>();
        let node_by_id = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.node_id, index))
            .collect::<BTreeMap<_, _>>();
        let mut element_by_id = BTreeMap::new();
        let mut elements = Vec::with_capacity(mesh.topology.volume_elements.len());
        let mut neighbor_sets = vec![BTreeSet::new(); nodes.len()];
        for element in &mesh.topology.volume_elements {
            let node_indices = element
                .node_ids
                .iter()
                .map(|node_id| {
                    node_by_id
                        .get(node_id)
                        .copied()
                        .ok_or(AnalysisProjectionError::UnknownNode {
                            entity_kind: "volume element",
                            entity_id: element.element_id,
                            node_id: *node_id,
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let corners =
                node_indices
                    .get(..4)
                    .ok_or(AnalysisProjectionError::InvalidElementGeometry {
                        element_id: element.element_id,
                    })?;
            let coordinates = [
                nodes[corners[0]].coordinates_m,
                nodes[corners[1]].coordinates_m,
                nodes[corners[2]].coordinates_m,
                nodes[corners[3]].coordinates_m,
            ];
            tetrahedron_geometry(coordinates).ok_or(
                AnalysisProjectionError::InvalidElementGeometry {
                    element_id: element.element_id,
                },
            )?;
            let element_coordinates = node_indices
                .iter()
                .map(|index| nodes[*index].coordinates_m)
                .collect::<Vec<_>>();
            let mut signed_volume_m3 = 0.0;
            let mut first_moment_m4 = [0.0; 3];
            visit_volume_quadrature(element.order, &element_coordinates, |point| {
                signed_volume_m3 += point.volume_weight_m3;
                for (first_moment, position) in first_moment_m4.iter_mut().zip(point.position_m) {
                    *first_moment += position * point.volume_weight_m3;
                }
            })
            .ok_or(AnalysisProjectionError::InvalidElementGeometry {
                element_id: element.element_id,
            })?;
            let centroid_m = first_moment_m4.map(|value| value / signed_volume_m3);
            let characteristic_length_m = (6.0 * signed_volume_m3).cbrt();
            for left in 0..node_indices.len() {
                for right in (left + 1)..node_indices.len() {
                    neighbor_sets[node_indices[left]].insert(node_indices[right]);
                    neighbor_sets[node_indices[right]].insert(node_indices[left]);
                }
            }
            element_by_id.insert(element.element_id, elements.len());
            elements.push(ProjectedElement {
                element_id: element.element_id,
                stable_identity: element.stable_identity,
                order: element.order,
                node_indices,
                region_id: element.region_id.clone(),
                signed_volume_m3,
                centroid_m,
                characteristic_length_m,
            });
        }
        let boundary_faces = mesh
            .topology
            .boundary_faces
            .iter()
            .map(|face| {
                let node_indices = face
                    .node_ids
                    .iter()
                    .map(|node_id| {
                        node_by_id.get(node_id).copied().ok_or(
                            AnalysisProjectionError::UnknownNode {
                                entity_kind: "boundary face",
                                entity_id: face.face_id,
                                node_id: *node_id,
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let adjacent_element_indices = face
                    .adjacent_volume_element_ids
                    .iter()
                    .map(|element_id| {
                        element_by_id.get(element_id).copied().ok_or(
                            AnalysisProjectionError::UnknownAdjacentElement {
                                face_id: face.face_id,
                                element_id: *element_id,
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                project_boundary_face(
                    face,
                    node_indices,
                    adjacent_element_indices,
                    &nodes,
                    &elements,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bounds_m = std::array::from_fn(|axis| {
            nodes
                .iter()
                .fold([f64::INFINITY, f64::NEG_INFINITY], |[min, max], node| {
                    [
                        min.min(node.coordinates_m[axis]),
                        max.max(node.coordinates_m[axis]),
                    ]
                })
        });
        let characteristic_length_m = bounds_m
            .iter()
            .map(|[min, max]| (max - min).powi(2))
            .sum::<f64>()
            .sqrt();
        let node_neighbors = neighbor_sets
            .into_iter()
            .map(|neighbors| neighbors.into_iter().collect())
            .collect();
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
