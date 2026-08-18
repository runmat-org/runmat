use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    ElementOrder, MeshNeighbor, MeshingCancellationSignal, SolverMeshNode, SolverMeshTopology,
    SolverVolumeElement, StableDigest,
};

use super::{
    boundaries::{build_edges, build_faces},
    checkpoint, error,
    inventories::{build_contacts, build_interfaces, build_regions, field_topologies},
    DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind, DelaunaySolverTopologyInput,
    DelaunaySolverTopologyOptions,
};

pub(super) const MAX_PROVENANCE_PER_ENTITY: usize = 32;

pub(super) fn construct(
    input: &DelaunaySolverTopologyInput<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SolverMeshTopology, DelaunaySolverTopologyError> {
    let node_indices = node_indices(input)?;
    let materials = validate_materials(input)?;
    let nodes = build_nodes(input, &node_indices, options, cancellation)?;
    let volume_elements = build_elements(input, &materials)?;
    let neighbors = build_neighbors(input);
    let (boundary_faces, classes) = build_faces(input, &node_indices, options, cancellation)?;
    let boundary_edges = build_edges(input, &boundary_faces, options, cancellation)?;
    let regions = build_regions(input, &materials)?;
    let material_interfaces = build_interfaces(input, &classes)?;
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
        material_interfaces,
        contacts,
        field_topologies,
    })
}

fn node_indices(
    input: &DelaunaySolverTopologyInput<'_>,
) -> Result<BTreeMap<StableDigest, u32>, DelaunaySolverTopologyError> {
    let indices = input
        .volume_mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    if indices.len() != input.volume_mesh.topology.nodes.len() {
        return Err(invalid_mesh("volume nodes repeat a stable identity"));
    }
    Ok(indices)
}

fn validate_materials<'a>(
    input: &'a DelaunaySolverTopologyInput<'_>,
) -> Result<BTreeMap<&'a PersistentEntityId, &'a str>, DelaunaySolverTopologyError> {
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
        return Err(invalid_materials(
            "solver projection requires one nonempty mesh region for every exact region",
        ));
    }
    let mut materials = BTreeMap::new();
    let mut previous = None;
    for assignment in input.region_materials {
        if previous.is_some_and(|prior| prior >= &assignment.region_id)
            || assignment.region_id.kind != PersistentEntityKind::Region
            || assignment.region_id.validate().is_err()
            || assignment.material_id.is_empty()
            || assignment.material_id.len() > 256
            || assignment.material_id.chars().any(char::is_control)
            || materials
                .insert(&assignment.region_id, assignment.material_id.as_str())
                .is_some()
        {
            return Err(invalid_materials(
                "region materials must be valid, nonempty, unique, and canonically ordered",
            ));
        }
        previous = Some(&assignment.region_id);
    }
    if materials.keys().copied().collect::<BTreeSet<_>>() != exact_regions {
        return Err(invalid_materials(
            "region materials must classify every exact region exactly once",
        ));
    }
    Ok(materials)
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
    input
        .volume_mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let provenance = provenance[index].iter().cloned().collect::<Vec<_>>();
            if provenance.is_empty() || provenance.len() > MAX_PROVENANCE_PER_ENTITY {
                return Err(invalid_mesh(
                    "solver node provenance is empty or exceeds its bound",
                ));
            }
            Ok(SolverMeshNode {
                node_id: index as u64 + 1,
                coordinates_m: node.coordinates_m,
                provenance,
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
    materials: &BTreeMap<&PersistentEntityId, &str>,
) -> Result<Vec<SolverVolumeElement>, DelaunaySolverTopologyError> {
    input
        .volume_mesh
        .topology
        .tetrahedra
        .iter()
        .enumerate()
        .map(|(index, tetrahedron)| {
            let region = tetrahedron
                .region_id
                .clone()
                .ok_or_else(|| invalid_mesh("volume tetrahedron has no assigned region"))?;
            let material = materials
                .get(&region)
                .ok_or_else(|| invalid_materials("volume region has no material assignment"))?;
            Ok(SolverVolumeElement {
                element_id: index as u64 + 1,
                order: ElementOrder::Tet4,
                node_ids: tetrahedron
                    .vertex_indices
                    .iter()
                    .map(|vertex| *vertex as u64 + 1)
                    .collect(),
                region_id: region.clone(),
                material_id: (*material).to_owned(),
                provenance: vec![region],
            })
        })
        .collect()
}

fn build_neighbors(input: &DelaunaySolverTopologyInput<'_>) -> Vec<MeshNeighbor> {
    input
        .volume_mesh
        .topology
        .tetrahedra
        .iter()
        .enumerate()
        .flat_map(|(index, tetrahedron)| {
            tetrahedron
                .neighbors
                .iter()
                .enumerate()
                .map(move |(face, adjacent)| MeshNeighbor {
                    element_id: index as u64 + 1,
                    local_face_index: face as u8,
                    adjacent_element_id: adjacent.map(|value| value as u64 + 1),
                })
        })
        .collect()
}

fn invalid_mesh(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}

fn invalid_materials(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMaterials, reason)
}
