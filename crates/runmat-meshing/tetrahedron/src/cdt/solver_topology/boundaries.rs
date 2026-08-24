use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    solver_boundary_edge_identity, solver_boundary_face_identity, BoundaryEdgeOrder,
    BoundaryFaceRole, BoundaryTriangleOrder, MeshingCancellationSignal, SolverBoundaryEdge,
    SolverBoundaryFace, StableDigest,
};

use super::{
    checkpoint, classification::ClassificationIndex, classification::ProjectedFaceClass, error,
    require_capacity, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyInput, DelaunaySolverTopologyOptions,
};

pub(super) fn build_faces(
    input: &DelaunaySolverTopologyInput<'_>,
    node_indices: &BTreeMap<StableDigest, u32>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(Vec<SolverBoundaryFace>, Vec<ProjectedFaceClass>), DelaunaySolverTopologyError> {
    let uses = tetrahedron_face_uses(input);
    let classifications = ClassificationIndex::new(input.exact_topology);
    let mut faces = Vec::new();
    let mut classes = Vec::new();
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
        let adjacent = uses
            .get(&key)
            .ok_or_else(|| invalid_mesh("protected facet is absent from volume topology"))?;
        let class = classifications.classify(&binding.entity_ids, &binding.region_ids)?;
        let expected_adjacency =
            usize::from(class.role == BoundaryFaceRole::ConformalInterface) + 1;
        if adjacent.len() != expected_adjacency {
            return Err(invalid_mesh(
                "protected facet adjacency does not match its exterior/contact/interface role",
            ));
        }
        faces.push(SolverBoundaryFace {
            face_id: faces.len() as u64 + 1,
            stable_identity: solver_boundary_face_identity(binding.node_identities),
            order: BoundaryTriangleOrder::Tri3,
            node_ids: oriented_face_nodes(input, key, adjacent, &class)?
                .map(|vertex| vertex as u64 + 1)
                .into(),
            adjacent_volume_element_ids: adjacent
                .iter()
                .map(|tetrahedron| *tetrahedron as u64 + 1)
                .collect(),
            role: class.role,
            provenance: binding.entity_ids.clone(),
        });
        classes.push(class);
    }
    Ok((faces, classes))
}

fn oriented_face_nodes(
    input: &DelaunaySolverTopologyInput<'_>,
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
                == Some(&class.outward_region_id)
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
            input.volume_mesh.topology.nodes[oriented[0] as usize].coordinates_m,
            input.volume_mesh.topology.nodes[oriented[1] as usize].coordinates_m,
            input.volume_mesh.topology.nodes[oriented[2] as usize].coordinates_m,
            input.volume_mesh.topology.nodes[tetrahedron.vertex_indices[opposite] as usize]
                .coordinates_m,
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

fn tetrahedron_face_uses(input: &DelaunaySolverTopologyInput<'_>) -> BTreeMap<[u32; 3], Vec<u32>> {
    let mut uses = BTreeMap::<[u32; 3], Vec<u32>>::new();
    for (tetrahedron_index, tetrahedron) in input.volume_mesh.topology.tetrahedra.iter().enumerate()
    {
        for opposite in 0..4 {
            let mut face = [0; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            uses.entry(face).or_default().push(tetrahedron_index as u32);
        }
    }
    uses
}

pub(super) fn build_edges(
    input: &DelaunaySolverTopologyInput<'_>,
    faces: &[SolverBoundaryFace],
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<SolverBoundaryEdge>, DelaunaySolverTopologyError> {
    let identity_by_node = input
        .volume_mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (index as u64 + 1, node.identity))
        .collect::<BTreeMap<_, _>>();
    let segment_provenance = input
        .volume_mesh
        .provenance
        .segments
        .iter()
        .map(|segment| (segment.node_identities, segment.entity_ids.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let mut edges = BTreeMap::<[u64; 2], (BTreeSet<u64>, BTreeSet<PersistentEntityId>)>::new();
    for (index, face) in faces.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        for pair in [[0, 1], [0, 2], [1, 2]] {
            let mut nodes = [face.node_ids[pair[0]], face.node_ids[pair[1]]];
            nodes.sort_unstable();
            let entry = edges.entry(nodes).or_default();
            entry.0.insert(face.face_id);
            let mut identities = [identity_by_node[&nodes[0]], identity_by_node[&nodes[1]]];
            identities.sort_unstable();
            if let Some(entities) = segment_provenance.get(&identities) {
                entry.1.extend(entities.iter().cloned());
            } else {
                entry.1.extend(face.provenance.iter().cloned());
            }
        }
    }
    if edges.len() as u64 > options.maximum_boundary_edges {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::ResourceLimit,
            format!(
                "boundary edge inventory exceeds its hard limit of {}",
                options.maximum_boundary_edges
            ),
        ));
    }
    edges
        .into_iter()
        .enumerate()
        .map(|(index, (node_ids, (face_ids, provenance)))| {
            if provenance.is_empty()
                || provenance.len() > super::construction::MAX_PROVENANCE_PER_ENTITY
            {
                return Err(invalid_mesh(
                    "boundary edge provenance is empty or exceeds its bound",
                ));
            }
            Ok(SolverBoundaryEdge {
                edge_id: index as u64 + 1,
                stable_identity: solver_boundary_edge_identity(
                    node_ids.map(|node_id| identity_by_node[&node_id]),
                ),
                order: BoundaryEdgeOrder::Line2,
                node_ids: node_ids.into(),
                adjacent_boundary_face_ids: face_ids.into_iter().collect(),
                provenance: provenance.into_iter().collect(),
            })
        })
        .collect()
}

fn invalid_mesh(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
