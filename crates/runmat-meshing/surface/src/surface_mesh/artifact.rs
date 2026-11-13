mod node_uses;
mod triangles;

use super::control::checkpoint;
use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId};
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    join::{
        conformity::{shell_evidence_with_control, validate_edge_conformity_with_control},
        validate_options,
    },
    ExactSurfaceJoinOptions, ExactSurfaceMesh, ExactSurfaceMeshError, ExactSurfaceMeshErrorKind,
    EXACT_SURFACE_MESH_SCHEMA_VERSION,
};

/// Validates the self-contained final publication contract. Face-partition replay remains the
/// stronger producer-side check; consumers validate only the canonical data they actually need.
pub fn validate_published_exact_surface_mesh(
    mesh: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
) -> Result<(), ExactSurfaceMeshError> {
    validate_with_control(mesh, topology, options, None)
}

pub(super) fn validate_with_control(
    mesh: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    validate_admitted(mesh, topology, options, control, None)
}

pub(super) fn validate_admitted(
    mesh: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
    allocations: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<(), ExactSurfaceMeshError> {
    checkpoint(control)?;
    validate_options(options)?;
    topology
        .validate_solid_shell_boundaries()
        .map_err(|error| invalid(error.to_string()))?;
    if mesh.schema_version != EXACT_SURFACE_MESH_SCHEMA_VERSION
        || !mesh
            .face_ids
            .iter()
            .eq(topology.faces.iter().map(|face| &face.id))
    {
        return Err(invalid(
            "published surface schema or exact face inventory is invalid",
        ));
    }
    if mesh.nodes.is_empty()
        || mesh.triangles.is_empty()
        || mesh.nodes.len() as u64 > options.maximum_nodes
        || mesh.triangles.len() as u64 > options.maximum_triangles
        || mesh.boundary_segments.len() as u64 > options.maximum_boundary_segments
    {
        return Err(limit("published surface inventory exceeds its hard limits"));
    }

    if mesh
        .nodes
        .windows(2)
        .any(|pair| pair[0].node_id >= pair[1].node_id)
    {
        return Err(invalid(
            "published surface nodes are not in canonical identity order",
        ));
    }
    let coedge = |id: &PersistentEntityId| {
        topology
            .coedges
            .binary_search_by(|coedge| coedge.id.cmp(id))
            .ok()
            .map(|index| &topology.coedges[index])
    };
    for node in &mesh.nodes {
        checkpoint(control)?;
        if node.node_id == StableDigest::ZERO
            || node.point_m.iter().any(|value| !value.is_finite())
            || node.uses.is_empty()
        {
            return Err(invalid("published surface node is invalid"));
        }
    }
    node_uses::validate(mesh, topology, control, allocations)?;
    triangles::validate(mesh, control, allocations)?;
    let _boundary_storage = super::join::nodes::reserve::<&crate::ExactFaceMeshBoundarySegment>(
        mesh.boundary_segments.len(),
        allocations,
    )?;
    let mut boundary_segments = Vec::with_capacity(mesh.boundary_segments.len());
    for segment in &mesh.boundary_segments {
        checkpoint(control)?;
        if coedge(&segment.source_coedge_id)
            .is_none_or(|coedge| coedge.edge_id != segment.source_edge_id)
            || segment.node_ids[0] == segment.node_ids[1]
            || segment.node_ids.iter().any(|node| {
                mesh.nodes
                    .binary_search_by_key(node, |node| node.node_id)
                    .is_err()
            })
            || segment
                .edge_parameters
                .iter()
                .any(|parameter| !parameter.is_finite())
        {
            return Err(invalid("published surface boundary segment is invalid"));
        }
        boundary_segments.push(segment);
    }
    runmat_cpu_execution::sort_in_place_with_checkpoint(
        &mut boundary_segments,
        |left, right| boundary_key(left).cmp(&boundary_key(right)),
        || checkpoint(control),
    )?;
    if boundary_segments
        .windows(2)
        .any(|pair| boundary_key(pair[0]) == boundary_key(pair[1]))
    {
        return Err(invalid("published surface boundary segment is invalid"));
    }
    validate_edge_conformity_with_control(topology, &mesh.boundary_segments, control)?;
    if mesh.shells != shell_evidence_with_control(topology, control)? {
        return Err(invalid(
            "published surface shell evidence differs from exact topology",
        ));
    }
    Ok(())
}

fn invalid(reason: impl Into<String>) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidInput, reason)
}

fn limit(reason: &str) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::ResourceLimit, reason)
}

type BoundaryKey<'a> = (
    &'a PersistentEntityId,
    &'a PersistentEntityId,
    [StableDigest; 2],
    [u64; 2],
);
fn boundary_key(segment: &crate::ExactFaceMeshBoundarySegment) -> BoundaryKey<'_> {
    (
        &segment.source_coedge_id,
        &segment.source_edge_id,
        segment.node_ids,
        segment.edge_parameters.map(f64::to_bits),
    )
}
