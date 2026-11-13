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
    DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind, DelaunaySolverTopologyOptions,
    ProjectionInput,
};

mod regions;
use regions::RegionSources;

pub(super) const MAX_PROVENANCE_PER_ENTITY: usize = 32;

pub(super) fn construct(
    input: &ProjectionInput<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SolverMeshTopology, DelaunaySolverTopologyError> {
    let node_indices = super::NodeIndex::new(input, options, cancellation)?;
    let region_sources = RegionSources::prepare(input, options, cancellation)?;
    let nodes = build_nodes(input, &node_indices, options, cancellation)?;
    let (volume_elements, neighbors) =
        build_elements_and_neighbors(input, &region_sources, options, cancellation)?;
    let (boundary_faces, classes) = build_faces(input, &node_indices, options, cancellation)?;
    let boundary_edges = build_edges(input, &boundary_faces, options, cancellation)?;
    let regions = build_regions(input)?;
    let conformal_interfaces = build_interfaces(input, &classes)?;
    let contacts = build_contacts(input, &classes)?;
    let field_topologies = field_topologies(
        input,
        nodes.len(),
        volume_elements.len(),
        boundary_faces.len(),
        boundary_edges.len(),
    )?;
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

fn build_nodes(
    input: &ProjectionInput<'_>,
    indices: &super::NodeIndex,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<SolverMeshNode>, DelaunaySolverTopologyError> {
    let storage = super::storage::Storage::for_input(input);
    let scratch_scope = storage
        .0
        .map(runmat_geometry_core::GeometryAllocationScope::new);
    let scratch = super::storage::Storage(
        scratch_scope
            .as_ref()
            .map(|scope| scope as &dyn runmat_geometry_core::GeometryEvaluationControl),
    );
    let mut provenance = scratch.collect((0..indices.len()).map(|_| Vec::new()))?;
    for (tetrahedron_index, tetrahedron) in input.volume_mesh.topology.tetrahedra.iter().enumerate()
    {
        checkpoint(tetrahedron_index as u64, options, cancellation)?;
        let region = tetrahedron
            .region_id
            .as_ref()
            .ok_or_else(|| invalid_mesh("volume tetrahedron has no assigned region"))?;
        for vertex in tetrahedron.vertex_indices {
            insert_entity_admitted(&mut provenance[vertex as usize], region, scratch)?;
        }
    }
    for binding in &input.volume_mesh.provenance.nodes {
        extend_nodes(
            &mut provenance,
            indices,
            &[binding.node_identity],
            &binding.entity_ids,
            scratch,
        )?;
    }
    for binding in &input.volume_mesh.provenance.segments {
        extend_nodes(
            &mut provenance,
            indices,
            &binding.node_identities,
            &binding.entity_ids,
            scratch,
        )?;
    }
    for binding in &input.volume_mesh.provenance.facets {
        extend_nodes(
            &mut provenance,
            indices,
            &binding.node_identities,
            &binding.entity_ids,
            scratch,
        )?;
    }
    let exact_parameters = build_node_exact_parameters(input, indices, options, cancellation)?;
    let mut nodes = storage.vector(indices.len())?;
    for ((index, node), exact_parameters) in input
        .volume_mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .zip(exact_parameters)
    {
        let provenance = storage.entities(provenance[index].iter().copied())?;
        if provenance.is_empty() || provenance.len() > MAX_PROVENANCE_PER_ENTITY {
            return Err(invalid_mesh(
                "solver node provenance is empty or exceeds its bound",
            ));
        }
        nodes.push(SolverMeshNode {
            node_id: index as u64 + 1,
            stable_identity: node.identity,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(node.coordinates_m.canonical_value()),
            provenance,
            exact_parameters,
        });
    }
    drop(provenance);
    drop(scratch_scope);
    Ok(nodes)
}

fn extend_nodes<'a, const N: usize>(
    target: &mut [Vec<&'a PersistentEntityId>],
    indices: &super::NodeIndex,
    identities: &[StableDigest; N],
    entities: &'a [PersistentEntityId],
    storage: super::storage::Storage<'_>,
) -> Result<(), DelaunaySolverTopologyError> {
    for identity in identities {
        let index = indices
            .get(identity)
            .ok_or_else(|| invalid_mesh("provenance references a missing volume node"))?;
        for entity in entities {
            insert_entity_admitted(&mut target[*index as usize], entity, storage)?;
        }
    }
    Ok(())
}

fn build_elements_and_neighbors(
    input: &ProjectionInput<'_>,
    regions: &RegionSources<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(Vec<SolverVolumeElement>, Vec<MeshNeighbor>), DelaunaySolverTopologyError> {
    let count = input.volume_mesh.topology.tetrahedra.len();
    let neighbor_count = count.checked_mul(4).ok_or_else(|| {
        error::failure(
            DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded,
            "solver neighbor inventory overflow",
        )
    })?;
    let storage = super::storage::Storage::for_input(input);
    let mut elements = storage.vector(count)?;
    let mut neighbors = storage.vector(neighbor_count)?;
    for (index, tetrahedron) in input.volume_mesh.topology.tetrahedra.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        let region = tetrahedron
            .region_id
            .as_ref()
            .ok_or_else(|| invalid_mesh("volume tetrahedron has no assigned region"))?;
        let solid = regions
            .solid(region)
            .ok_or_else(|| invalid_mesh("volume region has no exact solid lineage"))?;
        let mut provenance = storage.entities([region, solid].into_iter())?;
        provenance.sort();
        provenance.dedup();
        let (vertices, source_faces) = solver_vertices(input, tetrahedron.vertex_indices)?;
        let capacity = match input.request.element_order {
            ElementOrder::Tet4 => 4,
            ElementOrder::Tet10 => 10,
        };
        let mut node_ids = storage.vector(capacity)?;
        node_ids.extend(vertices.map(|vertex| vertex as u64 + 1));
        elements.push(SolverVolumeElement {
            element_id: index as u64 + 1,
            stable_identity: solver_volume_element_identity(
                vertices.map(|vertex| input.volume_mesh.topology.nodes[vertex as usize].identity),
            ),
            order: ElementOrder::Tet4,
            node_ids,
            region_id: storage.entity(region)?,
            provenance,
        });
        neighbors.extend(
            source_faces
                .into_iter()
                .enumerate()
                .map(|(face, source_face)| MeshNeighbor {
                    element_id: index as u64 + 1,
                    local_face_index: face as u8,
                    adjacent_element_id: tetrahedron.neighbors[source_face]
                        .map(|value| value as u64 + 1),
                }),
        );
    }
    Ok((elements, neighbors))
}

/// Converts the CDT's robust-predicate orientation into the positive Jacobian convention used
/// by solver elements. The same permutation maps solver-local faces back to CDT-local faces.
fn solver_vertices(
    input: &ProjectionInput<'_>,
    mut vertices: [u32; 4],
) -> Result<([u32; 4], [usize; 4]), DelaunaySolverTopologyError> {
    let points = vertices.map(|vertex| {
        input.volume_mesh.topology.nodes[vertex as usize]
            .coordinates_m
            .canonical_value()
    });
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

pub(super) fn insert_entity<'a>(
    target: &mut Vec<&'a PersistentEntityId>,
    entity: &'a PersistentEntityId,
) -> Result<(), DelaunaySolverTopologyError> {
    if let Err(index) = target.binary_search(&entity) {
        if target.len() == MAX_PROVENANCE_PER_ENTITY {
            return Err(invalid_mesh("solver entity provenance exceeds its bound"));
        }
        target.insert(index, entity);
    }
    Ok(())
}

fn insert_entity_admitted<'a>(
    target: &mut Vec<&'a PersistentEntityId>,
    entity: &'a PersistentEntityId,
    storage: super::storage::Storage<'_>,
) -> Result<(), DelaunaySolverTopologyError> {
    if target.binary_search(&entity).is_err() && target.len() == target.capacity() {
        let capacity = target
            .capacity()
            .saturating_mul(2)
            .clamp(4, MAX_PROVENANCE_PER_ENTITY);
        storage.reserve(target, capacity)?;
    }
    insert_entity(target, entity)
}
