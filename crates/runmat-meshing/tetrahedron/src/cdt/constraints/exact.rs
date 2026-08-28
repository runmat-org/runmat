use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId, TopologicalOrientation};
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use runmat_meshing_curve::shared_curve_vertex_node_id;
use runmat_meshing_surface::{ExactSurfaceMesh, EXACT_SURFACE_MESH_SCHEMA_VERSION};

use super::{
    checkpoint, error, resource, sorted_segment, validate_delaunay_constraints, validate_options,
    DelaunayConstraintError, DelaunayConstraintErrorKind, DelaunayConstraintFacet,
    DelaunayConstraintFacetSide, DelaunayConstraintNode, DelaunayConstraintOptions,
    DelaunayConstraintSegment, DelaunayConstraints,
};

#[path = "exact/boundaries.rs"]
mod boundaries;
use boundaries::boundary_edges;

pub fn build_delaunay_constraints(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    options: DelaunayConstraintOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayConstraints, DelaunayConstraintError> {
    validate_options(options)?;
    topology
        .validate_solid_shell_boundaries()
        .map_err(|failure| invalid_geometry(failure.to_string()))?;
    validate_surface(topology, surface, options, cancellation)?;

    let face_sides = face_sides(topology)?;
    let contact_ids = contact_ids(topology);
    let node_by_id = surface
        .nodes
        .iter()
        .map(|node| (node.node_id, node))
        .collect::<BTreeMap<_, _>>();
    let mut source_vertex_by_node = BTreeMap::new();
    for vertex in &topology.vertices {
        let node_id = shared_curve_vertex_node_id(&vertex.id);
        if source_vertex_by_node
            .insert(node_id, vertex.id.clone())
            .is_some()
        {
            return Err(error(
                DelaunayConstraintErrorKind::IdentityCollision,
                "distinct persistent vertices produced one exact surface node identity",
            ));
        }
    }
    let boundary_edges = boundary_edges(topology, surface)?;

    let solid_triangles = surface
        .triangles
        .iter()
        .filter(|triangle| face_sides.contains_key(&triangle.source_face_id))
        .collect::<Vec<_>>();
    if solid_triangles.is_empty() || solid_triangles.len() as u64 > options.maximum_facets {
        return Err(resource(
            "exact solid surface facet inventory is empty or exceeds its hard limit",
        ));
    }
    let referenced_nodes = solid_triangles
        .iter()
        .flat_map(|triangle| triangle.node_ids)
        .collect::<BTreeSet<_>>();
    if referenced_nodes.len() as u64 > options.maximum_nodes {
        return Err(resource(
            "exact solid surface node inventory exceeds its hard limit",
        ));
    }
    let nodes = referenced_nodes
        .iter()
        .enumerate()
        .map(|(index, identity)| {
            checkpoint(index, options, cancellation)?;
            let source = node_by_id.get(identity).ok_or_else(|| {
                invalid_boundary("solid surface triangle references an absent exact surface node")
            })?;
            Ok(DelaunayConstraintNode {
                identity: *identity,
                source_vertex_id: source_vertex_by_node.get(identity).cloned(),
                coordinates_m: source.point_m,
            })
        })
        .collect::<Result<Vec<_>, DelaunayConstraintError>>()?;
    let node_index = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();

    let mut segment_keys = BTreeSet::new();
    let mut facets = Vec::with_capacity(solid_triangles.len());
    for (index, triangle) in solid_triangles.into_iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        let vertex_indices = triangle.node_ids.map(|identity| node_index[&identity]);
        for edge in 0..3 {
            segment_keys.insert(sorted_segment([
                vertex_indices[edge],
                vertex_indices[(edge + 1) % 3],
            ]));
        }
        let (positive_side, negative_side) = face_sides
            .get(&triangle.source_face_id)
            .cloned()
            .ok_or_else(|| invalid_boundary("solid face side classification is incomplete"))?;
        facets.push(DelaunayConstraintFacet {
            facet_id: triangle.triangle_id,
            chart_id: triangle.chart_id,
            vertex_indices,
            source_face_id: triangle.source_face_id.clone(),
            positive_side,
            negative_side,
            contact_ids: contact_ids
                .get(&triangle.source_face_id)
                .cloned()
                .unwrap_or_default(),
        });
    }
    if segment_keys.len() as u64 > options.maximum_segments {
        return Err(resource(
            "exact solid surface segment inventory exceeds its hard limit",
        ));
    }
    facets.sort_by_key(|facet| {
        let mut key = facet.vertex_indices;
        key.sort_unstable();
        (key, facet.facet_id)
    });
    let segments = segment_keys
        .into_iter()
        .map(|vertex_indices| {
            let identities = vertex_indices.map(|vertex| nodes[vertex as usize].identity);
            let exact = boundary_edges.get(&sorted_id_pair(identities));
            DelaunayConstraintSegment {
                vertex_indices,
                source_edge_id: exact.map(|segment| segment.edge_id.clone()),
                source_edge_parameters: exact.map(|segment| segment.parameters),
            }
        })
        .collect();
    let constraints = DelaunayConstraints {
        nodes,
        segments,
        facets,
    };
    validate_delaunay_constraints(&constraints, options, cancellation)?;
    Ok(constraints)
}

fn validate_surface(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    options: DelaunayConstraintOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayConstraintError> {
    if surface.schema_version != EXACT_SURFACE_MESH_SCHEMA_VERSION
        || surface.face_ids
            != topology
                .faces
                .iter()
                .map(|face| face.id.clone())
                .collect::<Vec<_>>()
    {
        return Err(invalid_boundary(
            "exact surface schema and face inventory must match exact topology",
        ));
    }
    if surface.nodes.len() as u64 > options.maximum_nodes
        || surface.triangles.len() as u64 > options.maximum_facets
        || surface.boundary_segments.len() as u64 > options.maximum_segments
    {
        return Err(resource(
            "exact surface source inventory exceeds the CDT constraint hard limits",
        ));
    }
    let mut node_ids = BTreeSet::new();
    for (index, node) in surface.nodes.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        if node.node_id == StableDigest::ZERO
            || node.point_m.iter().any(|value| !value.is_finite())
            || index > 0 && surface.nodes[index - 1].node_id >= node.node_id
            || !node_ids.insert(node.node_id)
        {
            return Err(invalid_boundary(
                "exact surface nodes must be finite, nonzero, canonical, and unique",
            ));
        }
    }
    let faces = topology
        .faces
        .iter()
        .map(|face| &face.id)
        .collect::<BTreeSet<_>>();
    let mut triangle_ids = BTreeSet::new();
    for (index, triangle) in surface.triangles.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        if triangle.triangle_id == StableDigest::ZERO
            || !triangle_ids.insert(triangle.triangle_id)
            || !faces.contains(&triangle.source_face_id)
            || triangle.node_ids.windows(2).any(|pair| pair[0] == pair[1])
            || triangle.node_ids[0] > triangle.node_ids[1]
            || triangle.node_ids[0] > triangle.node_ids[2]
            || triangle
                .node_ids
                .iter()
                .any(|identity| !node_ids.contains(identity))
        {
            return Err(invalid_boundary(
                "exact surface triangles must be unique, canonically rotated, and resolve exact face and node identities",
            ));
        }
    }
    Ok(())
}

fn face_sides(
    topology: &ExactBRepTopology,
) -> Result<
    BTreeMap<PersistentEntityId, (DelaunayConstraintFacetSide, DelaunayConstraintFacetSide)>,
    DelaunayConstraintError,
> {
    let mut region_by_solid = BTreeMap::new();
    let mut region_ids = BTreeSet::new();
    for region in &topology.regions {
        if !region_ids.insert(region.id.clone())
            || region_by_solid
                .insert(&region.solid_id, &region.id)
                .is_some()
        {
            return Err(invalid_geometry(
                "persistent regions and their solid ownership must be unique",
            ));
        }
    }
    let mut shell_owner = BTreeMap::new();
    for solid in &topology.solids {
        let region = region_by_solid.get(&solid.id).ok_or_else(|| {
            invalid_geometry("exact solid is missing its canonical persistent region")
        })?;
        if shell_owner
            .insert(
                &solid.outer_shell_id,
                ((*region).clone(), DelaunayConstraintFacetSide::Exterior),
            )
            .is_some()
        {
            return Err(invalid_geometry(
                "one exact shell has more than one solid owner",
            ));
        }
        for shell in &solid.void_shell_ids {
            if shell_owner
                .insert(
                    shell,
                    ((*region).clone(), DelaunayConstraintFacetSide::Void),
                )
                .is_some()
            {
                return Err(invalid_geometry(
                    "one exact shell has more than one solid owner",
                ));
            }
        }
    }
    let mut interfaces = BTreeSet::new();
    for interface in &topology.interfaces {
        if !interfaces.insert(&interface.face_id)
            || !region_ids.contains(&interface.side_a_region_id)
            || !region_ids.contains(&interface.side_b_region_id)
        {
            return Err(invalid_geometry(
                "exact interfaces require unique faces and two existing persistent regions",
            ));
        }
    }
    let mut result = BTreeMap::new();
    for shell in &topology.shells {
        let Some((region, absent)) = shell_owner.get(&shell.id) else {
            continue;
        };
        for face_use in &shell.face_uses {
            if interfaces.contains(&face_use.entity_id) {
                continue;
            }
            // Exact surface triangles already carry `ExactFace::orientation` (the surface join
            // reverses their winding when necessary). Shell and face-use orientation therefore
            // classify that oriented triangle directly; composing the face orientation again
            // would invert every reversed face twice.
            let orientation = compose(shell.orientation, face_use.orientation);
            let sides = region_and_absent(region.clone(), absent.clone(), orientation);
            if result.insert(face_use.entity_id.clone(), sides).is_some() {
                return Err(invalid_geometry(
                    "ordinary solid face has more than one solid shell use",
                ));
            }
        }
    }
    for interface in &topology.interfaces {
        let first = region_side(
            interface.side_a_region_id.clone(),
            interface.side_a_orientation,
        );
        let second = region_side(
            interface.side_b_region_id.clone(),
            interface.side_b_orientation,
        );
        let sides = merge_region_sides(first, second)?;
        if result.insert(interface.face_id.clone(), sides).is_some() {
            return Err(invalid_geometry(
                "one exact face has more than one interface classification",
            ));
        }
    }
    Ok(result)
}

fn region_and_absent(
    region: PersistentEntityId,
    absent: DelaunayConstraintFacetSide,
    orientation: TopologicalOrientation,
) -> (DelaunayConstraintFacetSide, DelaunayConstraintFacetSide) {
    match orientation {
        TopologicalOrientation::Forward => (absent, DelaunayConstraintFacetSide::Region(region)),
        TopologicalOrientation::Reversed => (DelaunayConstraintFacetSide::Region(region), absent),
    }
}

fn region_side(
    region: PersistentEntityId,
    orientation: TopologicalOrientation,
) -> (bool, DelaunayConstraintFacetSide) {
    (
        orientation == TopologicalOrientation::Reversed,
        DelaunayConstraintFacetSide::Region(region),
    )
}

fn merge_region_sides(
    first: (bool, DelaunayConstraintFacetSide),
    second: (bool, DelaunayConstraintFacetSide),
) -> Result<(DelaunayConstraintFacetSide, DelaunayConstraintFacetSide), DelaunayConstraintError> {
    match (first, second) {
        ((true, positive), (false, negative)) | ((false, negative), (true, positive)) => {
            Ok((positive, negative))
        }
        _ => Err(invalid_geometry(
            "conformal interface region uses do not classify opposite facet sides",
        )),
    }
}

fn contact_ids(
    topology: &ExactBRepTopology,
) -> BTreeMap<PersistentEntityId, Vec<PersistentEntityId>> {
    let mut result = BTreeMap::<PersistentEntityId, Vec<PersistentEntityId>>::new();
    for contact in &topology.contacts {
        for face in contact
            .side_a_face_ids
            .iter()
            .chain(&contact.side_b_face_ids)
        {
            result
                .entry(face.clone())
                .or_default()
                .push(contact.id.clone());
        }
    }
    result
}

pub(super) fn sorted_id_pair(mut identities: [StableDigest; 2]) -> [StableDigest; 2] {
    identities.sort_unstable();
    identities
}

fn compose(left: TopologicalOrientation, right: TopologicalOrientation) -> TopologicalOrientation {
    if left == right {
        TopologicalOrientation::Forward
    } else {
        TopologicalOrientation::Reversed
    }
}

fn invalid_geometry(reason: impl Into<String>) -> DelaunayConstraintError {
    error(DelaunayConstraintErrorKind::InvalidGeometry, reason)
}

fn invalid_boundary(reason: impl Into<String>) -> DelaunayConstraintError {
    error(DelaunayConstraintErrorKind::InvalidBoundary, reason)
}
