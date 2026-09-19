use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    quality::predicate::tetrahedron_signed_volume, solver_volume_element_identity, ElementOrder,
    MeshNeighbor, MeshingCancellationSignal, SolverMeshNode, SolverMeshTopology,
    SolverVolumeElement, StableDigest,
};

use super::{
    boundaries::{build_edges, build_faces},
    checkpoint, error,
    inventories::{build_contacts, build_interfaces, build_regions, field_topologies},
    parameters::build_node_exact_parameters,
    DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind, DelaunaySolverTopologyInput,
    DelaunaySolverTopologyOptions,
};

pub(super) const MAX_PROVENANCE_PER_ENTITY: usize = 32;

pub(super) fn construct(
    input: &DelaunaySolverTopologyInput<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SolverMeshTopology, DelaunaySolverTopologyError> {
    let node_indices = node_indices(input, options, cancellation)?;
    validate_regions(input)?;
    let nodes = build_nodes(input, &node_indices, options, cancellation)?;
    let volume_elements = build_elements(input, options, cancellation)?;
    let neighbors = build_neighbors(input, options, cancellation)?;
    let (boundary_faces, classes) = build_faces(input, &node_indices, options, cancellation)?;
    let boundary_edges = build_edges(input, &boundary_faces, options, cancellation)?;
    let regions = build_regions(input)?;
    let conformal_interfaces = build_interfaces(input, &classes)?;
    let contacts = build_contacts(input, &classes)?;
    let field_topologies = field_topologies(
        nodes.len(),
        volume_elements.len(),
        boundary_faces.len(),
        boundary_edges.len(),
    );
    Ok(SolverMeshTopology {
        nodes,
        volume_elements,
        neighbors,
        boundary_faces,
        boundary_edges,
        regions,
        conformal_interfaces,
        contacts,
        field_topologies,
    })
}

fn node_indices(
    input: &DelaunaySolverTopologyInput<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<BTreeMap<StableDigest, u32>, DelaunaySolverTopologyError> {
    let mut indices = BTreeMap::new();
    for (index, node) in input.volume_mesh.topology.nodes.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        indices.insert(node.identity, index as u32);
    }
    if indices.len() != input.volume_mesh.topology.nodes.len() {
        return Err(invalid_mesh("volume nodes repeat a stable identity"));
    }
    Ok(indices)
}

fn validate_regions(
    input: &DelaunaySolverTopologyInput<'_>,
) -> Result<(), DelaunaySolverTopologyError> {
    let exact_regions = input
        .exact_topology
        .regions
        .iter()
        .map(|region| &region.id)
        .collect::<BTreeSet<_>>();
    let mesh_regions = input
        .volume_mesh
        .topology
        .incidence
        .regions
        .iter()
        .map(|region| &region.region_id)
        .collect::<BTreeSet<_>>();
    if exact_regions != mesh_regions {
        return Err(invalid_mesh(
            "solver projection requires one nonempty mesh region for every exact region",
        ));
    }
    Ok(())
}

fn build_nodes(
    input: &DelaunaySolverTopologyInput<'_>,
    indices: &BTreeMap<StableDigest, u32>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<SolverMeshNode>, DelaunaySolverTopologyError> {
    let mut provenance = vec![BTreeSet::new(); indices.len()];
    for (tetrahedron_index, tetrahedron) in input.volume_mesh.topology.tetrahedra.iter().enumerate()
    {
        checkpoint(tetrahedron_index as u64, options, cancellation)?;
        let region = tetrahedron
            .region_id
            .as_ref()
            .ok_or_else(|| invalid_mesh("volume tetrahedron has no assigned region"))?;
        for vertex in tetrahedron.vertex_indices {
            provenance[vertex as usize].insert(region.clone());
        }
    }
    for binding in &input.volume_mesh.provenance.nodes {
        extend_nodes(
            &mut provenance,
            indices,
            &[binding.node_identity],
            &binding.entity_ids,
        )?;
    }
    for binding in &input.volume_mesh.provenance.segments {
        extend_nodes(
            &mut provenance,
            indices,
            &binding.node_identities,
            &binding.entity_ids,
        )?;
    }
    for binding in &input.volume_mesh.provenance.facets {
        extend_nodes(
            &mut provenance,
            indices,
            &binding.node_identities,
            &binding.entity_ids,
        )?;
    }
    let exact_parameters = build_node_exact_parameters(input, indices, options, cancellation)?;
    input
        .volume_mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .zip(exact_parameters)
        .map(|((index, node), exact_parameters)| {
            let provenance = provenance[index].iter().cloned().collect::<Vec<_>>();
            if provenance.is_empty() || provenance.len() > MAX_PROVENANCE_PER_ENTITY {
                return Err(invalid_mesh(
                    "solver node provenance is empty or exceeds its bound",
                ));
            }
            Ok(SolverMeshNode {
                node_id: index as u64 + 1,
                stable_identity: node.identity,
                coordinates_m: node.coordinates_m,
                provenance,
                exact_parameters,
            })
        })
        .collect()
}

fn extend_nodes<const N: usize>(
    target: &mut [BTreeSet<PersistentEntityId>],
    indices: &BTreeMap<StableDigest, u32>,
    identities: &[StableDigest; N],
    entities: &[PersistentEntityId],
) -> Result<(), DelaunaySolverTopologyError> {
    for identity in identities {
        let index = indices
            .get(identity)
            .ok_or_else(|| invalid_mesh("provenance references a missing volume node"))?;
        target[*index as usize].extend(entities.iter().cloned());
    }
    Ok(())
}

fn build_elements(
    input: &DelaunaySolverTopologyInput<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<SolverVolumeElement>, DelaunaySolverTopologyError> {
    let solid_by_region = input
        .exact_topology
        .regions
        .iter()
        .map(|region| (region.id.clone(), region.solid_id.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::with_capacity(input.volume_mesh.topology.tetrahedra.len());
    for (index, tetrahedron) in input.volume_mesh.topology.tetrahedra.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        let region = tetrahedron
            .region_id
            .clone()
            .ok_or_else(|| invalid_mesh("volume tetrahedron has no assigned region"))?;
        let solid = solid_by_region
            .get(&region)
            .cloned()
            .ok_or_else(|| invalid_mesh("volume region has no exact solid lineage"))?;
        let mut provenance = vec![region.clone(), solid];
        provenance.sort();
        provenance.dedup();
        let (vertices, _) = solver_vertices(input, tetrahedron.vertex_indices)?;
        result.push(SolverVolumeElement {
            element_id: index as u64 + 1,
            stable_identity: solver_volume_element_identity(
                vertices.map(|vertex| input.volume_mesh.topology.nodes[vertex as usize].identity),
            ),
            order: ElementOrder::Tet4,
            node_ids: vertices.iter().map(|vertex| *vertex as u64 + 1).collect(),
            region_id: region.clone(),
            provenance,
        });
    }
    Ok(result)
}

fn build_neighbors(
    input: &DelaunaySolverTopologyInput<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<MeshNeighbor>, DelaunaySolverTopologyError> {
    let mut result = Vec::with_capacity(input.volume_mesh.topology.tetrahedra.len() * 4);
    for (index, tetrahedron) in input.volume_mesh.topology.tetrahedra.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        let (_, source_face_by_solver_face) = solver_vertices(input, tetrahedron.vertex_indices)?;
        result.extend(source_face_by_solver_face.into_iter().enumerate().map(
            |(face, source_face)| MeshNeighbor {
                element_id: index as u64 + 1,
                local_face_index: face as u8,
                adjacent_element_id:
                    tetrahedron.neighbors[source_face].map(|value| value as u64 + 1),
            },
        ));
    }
    Ok(result)
}

/// Converts the CDT's robust-predicate orientation into the positive Jacobian convention used
/// by solver elements. The same permutation maps solver-local faces back to CDT-local faces.
fn solver_vertices(
    input: &DelaunaySolverTopologyInput<'_>,
    mut vertices: [u32; 4],
) -> Result<([u32; 4], [usize; 4]), DelaunaySolverTopologyError> {
    let points =
        vertices.map(|vertex| input.volume_mesh.topology.nodes[vertex as usize].coordinates_m);
    let signed_volume = tetrahedron_signed_volume(points);
    if !signed_volume.is_finite() || signed_volume == 0.0 {
        return Err(invalid_mesh(
            "volume tetrahedron has zero or non-finite solver Jacobian",
        ));
    }
    let mut source_face_by_solver_face = [0, 1, 2, 3];
    if signed_volume < 0.0 {
        vertices.swap(0, 1);
        source_face_by_solver_face.swap(0, 1);
    }
    Ok((vertices, source_face_by_solver_face))
}

fn invalid_mesh(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
