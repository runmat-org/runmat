use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    solver_boundary_face_identity, BoundaryFaceRole, BoundaryTriangleOrder,
    MeshingCancellationSignal, SolverBoundaryFace,
};

use super::{
    checkpoint, classification::ClassificationIndex, classification::ProjectedFaceClass, error,
    require_capacity, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyOptions, ProjectionInput,
};

pub(super) fn build_faces<'a>(
    input: &'a ProjectionInput<'_>,
    node_indices: &super::NodeIndex,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(Vec<SolverBoundaryFace>, Vec<ProjectedFaceClass<'a>>), DelaunaySolverTopologyError> {
    let classifications = ClassificationIndex::new(
        input.exact_topology,
        input.exact_evaluation.map(|evaluation| evaluation.control),
    )?;
    let count = input.volume_mesh.provenance.facets.len();
    if count as u64 > options.maximum_boundary_faces {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::ResourceLimit,
            "boundary face inventory exceeds its hard limit",
        ));
    }
    let storage = super::storage::Storage::for_input(input);
    let mut faces = storage.vector(count)?;
    let mut classes = storage.vector(count)?;
    for (index, binding) in input.volume_mesh.provenance.facets.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        require_capacity(
            "boundary face inventory",
            faces.len(),
            options.maximum_boundary_faces,
        )?;
        let [first, second, third] = binding.node_identities;
        let lookup = |identity| {
            node_indices
                .get(&identity)
                .copied()
                .ok_or_else(|| invalid_mesh("facet provenance references a missing volume node"))
        };
        let mut key = [lookup(first)?, lookup(second)?, lookup(third)?];
        key.sort_unstable();
        let (adjacent, count) =
            face_adjacency(input.volume_mesh.topology, key, options, cancellation)?;
        let adjacent = &adjacent[..count];
        let class = classifications.classify(&binding.entity_ids, &binding.region_ids)?;
        let expected_adjacency =
            usize::from(class.role == BoundaryFaceRole::ConformalInterface) + 1;
        if adjacent.len() != expected_adjacency {
            return Err(invalid_mesh(
                "protected facet adjacency does not match its exterior/contact/interface role",
            ));
        }
        let capacity = match input.request.element_order {
            runmat_meshing_core::ElementOrder::Tet4 => 3,
            runmat_meshing_core::ElementOrder::Tet10 => 6,
        };
        let mut node_ids = storage.vector(capacity)?;
        node_ids.extend(
            oriented_face_nodes(input, key, adjacent, &class)?.map(|vertex| vertex as u64 + 1),
        );
        faces.push(SolverBoundaryFace {
            face_id: faces.len() as u64 + 1,
            stable_identity: solver_boundary_face_identity(binding.node_identities),
            order: BoundaryTriangleOrder::Tri3,
            node_ids,
            adjacent_volume_element_ids: storage
                .collect(adjacent.iter().map(|tetrahedron| *tetrahedron as u64 + 1))?,
            role: class.role,
            provenance: storage.entities(binding.entity_ids.iter())?,
        });
        classes.push(class);
    }
    Ok((faces, classes))
}

fn oriented_face_nodes(
    input: &ProjectionInput<'_>,
    key: [u32; 3],
    adjacent: &[u32],
    class: &ProjectedFaceClass,
) -> Result<[u32; 3], DelaunaySolverTopologyError> {
    let tetrahedron_index = adjacent
        .iter()
        .find(|index| {
            input.volume_mesh.topology.tetrahedra[**index as usize]
                .region_id
                .as_ref()
                == Some(class.outward_region_id)
        })
        .ok_or_else(|| invalid_mesh("boundary face has no tetrahedron in its outward region"))?;
    let tetrahedron = &input.volume_mesh.topology.tetrahedra[*tetrahedron_index as usize];
    for opposite in 0..4 {
        let mut oriented = [0; 3];
        let mut cursor = 0;
        for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
            if slot != opposite {
                oriented[cursor] = *vertex;
                cursor += 1;
            }
        }
        let mut canonical = oriented;
        canonical.sort_unstable();
        if canonical != key {
            continue;
        }
        let points = [
            input.volume_mesh.topology.nodes[oriented[0] as usize]
                .coordinates_m
                .canonical_value(),
            input.volume_mesh.topology.nodes[oriented[1] as usize]
                .coordinates_m
                .canonical_value(),
            input.volume_mesh.topology.nodes[oriented[2] as usize]
                .coordinates_m
                .canonical_value(),
            input.volume_mesh.topology.nodes[tetrahedron.vertex_indices[opposite] as usize]
                .coordinates_m
                .canonical_value(),
        ];
        match orient3d(points)
            .map_err(|failure| invalid_mesh(format!("face orientation failed: {failure:?}")))?
        {
            PredicateSign::Positive => oriented.swap(0, 1),
            PredicateSign::Negative => {}
            PredicateSign::Zero => {
                return Err(invalid_mesh("solver boundary face is exactly degenerate"));
            }
        }
        return Ok(oriented);
    }
    Err(invalid_mesh(
        "protected facet is not a face of its adjacent tetrahedron",
    ))
}

/// Reuse independently validated local incidence; no all-volume face tree.
fn face_adjacency(
    topology: &crate::cdt::DelaunayVolumeTopology,
    key: [u32; 3],
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<([u32; 2], usize), DelaunaySolverTopologyError> {
    let star = key
        .iter()
        .filter_map(|vertex| topology.incidence.vertex_stars.get(*vertex as usize))
        .min_by_key(|star| star.len())
        .ok_or_else(|| invalid_mesh("boundary facet has no vertex incidence"))?;
    let mut adjacent = [0; 2];
    let mut count = 0;
    for (work, index) in star.iter().enumerate() {
        checkpoint(work as u64, options, cancellation)?;
        let cell = topology
            .tetrahedra
            .get(*index as usize)
            .ok_or_else(|| invalid_mesh("invalid vertex incidence"))?;
        if key
            .iter()
            .all(|vertex| cell.vertex_indices.contains(vertex))
        {
            if count == adjacent.len() {
                return Err(invalid_mesh("protected facet has nonmanifold adjacency"));
            }
            adjacent[count] = *index;
            count += 1;
        }
    }
    if count == 0 {
        return Err(invalid_mesh(
            "protected facet is absent from volume topology",
        ));
    }
    adjacent[..count].sort_unstable();
    Ok((adjacent, count))
}

mod edges;
pub(super) use edges::build_edges;

fn invalid_mesh(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}

#[cfg(test)]
mod adjacency_tests {
    use super::*;
    #[test]
    fn local_incidence_matches_complete_scan_for_interior_and_boundary_faces() {
        for cells in [2, 4, 8] {
            let topology = crate::cdt::test_fixtures::structured_volume::grid(cells, false);
            for cell in topology.tetrahedra.iter().take(12) {
                for opposite in 0..4 {
                    let mut key = [0; 3];
                    let mut cursor = 0;
                    for (slot, &vertex) in cell.vertex_indices.iter().enumerate() {
                        if slot != opposite {
                            key[cursor] = vertex;
                            cursor += 1;
                        }
                    }
                    key.sort_unstable();
                    let expected = topology
                        .tetrahedra
                        .iter()
                        .enumerate()
                        .filter(|(_, cell)| {
                            key.iter()
                                .all(|vertex| cell.vertex_indices.contains(vertex))
                        })
                        .map(|(index, _)| index as u32)
                        .collect::<Vec<_>>();
                    let (actual, count) = face_adjacency(
                        &topology,
                        key,
                        DelaunaySolverTopologyOptions::default(),
                        &runmat_meshing_core::NeverCancelled,
                    )
                    .unwrap();
                    assert_eq!(&actual[..count], expected.as_slice());
                }
            }
        }
    }
}
