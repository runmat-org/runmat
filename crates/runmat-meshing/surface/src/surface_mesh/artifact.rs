use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::StableDigest;

use super::{
    join::{shell_evidence, validate_edge_conformity, validate_options},
    ExactSurfaceJoinOptions, ExactSurfaceMesh, ExactSurfaceMeshError, ExactSurfaceMeshErrorKind,
    EXACT_SURFACE_MESH_SCHEMA_VERSION,
};
use crate::face_mesh::{canonical_triangle, exact_face_triangle_id, valid_triangle_measures};

/// Validates the self-contained final publication contract. Face-partition replay remains the
/// stronger producer-side check; consumers validate only the canonical data they actually need.
pub fn validate_published_exact_surface_mesh(
    mesh: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
) -> Result<(), ExactSurfaceMeshError> {
    validate_options(options)?;
    topology
        .validate_solid_shell_boundaries()
        .map_err(|error| invalid(error.to_string()))?;
    if mesh.schema_version != EXACT_SURFACE_MESH_SCHEMA_VERSION
        || mesh.face_ids
            != topology
                .faces
                .iter()
                .map(|face| face.id.clone())
                .collect::<Vec<_>>()
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

    let faces = topology
        .faces
        .iter()
        .map(|face| &face.id)
        .collect::<BTreeSet<_>>();
    let coedges = topology
        .coedges
        .iter()
        .map(|coedge| (&coedge.id, coedge))
        .collect::<BTreeMap<_, _>>();
    let mut nodes = BTreeMap::new();
    for node in &mesh.nodes {
        if node.node_id == StableDigest::ZERO
            || node.point_m.iter().any(|value| !value.is_finite())
            || node.uses.is_empty()
            || nodes.insert(node.node_id, node).is_some()
        {
            return Err(invalid("published surface node is invalid"));
        }
        let mut uses = BTreeSet::new();
        for use_record in &node.uses {
            if !faces.contains(&use_record.source_face_id)
                || use_record.chart_id == StableDigest::ZERO
                || use_record
                    .uv
                    .iter()
                    .chain(&use_record.evaluator_uv)
                    .any(|value| !value.is_finite())
                || !uses.insert((
                    &use_record.source_face_id,
                    use_record.chart_id,
                    use_record.uv.map(f64::to_bits),
                    use_record.evaluator_uv.map(f64::to_bits),
                ))
            {
                return Err(invalid("published surface node use is invalid"));
            }
            let mut parameters = BTreeSet::new();
            for parameter in &use_record.exact_edge_parameters {
                if parameter.source_coedge_id.kind != PersistentEntityKind::Coedge
                    || parameter.source_edge_id.kind != PersistentEntityKind::Edge
                    || coedges
                        .get(&parameter.source_coedge_id)
                        .is_none_or(|coedge| {
                            coedge.face_id != use_record.source_face_id
                                || coedge.edge_id != parameter.source_edge_id
                        })
                    || !parameter.parameter.is_finite()
                    || !parameters.insert((
                        &parameter.source_coedge_id,
                        &parameter.source_edge_id,
                        parameter.parameter.to_bits(),
                    ))
                {
                    return Err(invalid("published surface exact-edge parameter is invalid"));
                }
            }
        }
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

    validate_triangles(mesh, &faces, &nodes)?;
    let mut boundary_segments = BTreeSet::new();
    for segment in &mesh.boundary_segments {
        if coedges
            .get(&segment.source_coedge_id)
            .is_none_or(|coedge| coedge.edge_id != segment.source_edge_id)
            || segment.node_ids[0] == segment.node_ids[1]
            || segment
                .node_ids
                .iter()
                .any(|node| !nodes.contains_key(node))
            || segment
                .edge_parameters
                .iter()
                .any(|parameter| !parameter.is_finite())
            || !boundary_segments.insert((
                &segment.source_coedge_id,
                &segment.source_edge_id,
                segment.node_ids,
                segment.edge_parameters.map(f64::to_bits),
            ))
        {
            return Err(invalid("published surface boundary segment is invalid"));
        }
    }
    validate_edge_conformity(topology, &mesh.boundary_segments)?;
    if mesh.shells != shell_evidence(topology)? {
        return Err(invalid(
            "published surface shell evidence differs from exact topology",
        ));
    }
    Ok(())
}

fn validate_triangles(
    mesh: &ExactSurfaceMesh,
    faces: &BTreeSet<&PersistentEntityId>,
    nodes: &BTreeMap<StableDigest, &crate::ExactFaceMeshNode>,
) -> Result<(), ExactSurfaceMeshError> {
    let mut identities = BTreeSet::new();
    let mut facets = BTreeSet::new();
    let mut covered_faces = BTreeSet::new();
    let mut maximum_chordal = 0.0_f64;
    let mut maximum_normal = 0.0_f64;
    for triangle in &mesh.triangles {
        let (canonical, _) = canonical_triangle(triangle.node_ids);
        let mut facet = triangle.node_ids;
        facet.sort_unstable();
        if !faces.contains(&triangle.source_face_id)
            || triangle.chart_id == StableDigest::ZERO
            || triangle.triangle_id != exact_face_triangle_id(triangle.chart_id, triangle.node_ids)
            || canonical != triangle.node_ids
            || !identities.insert(triangle.triangle_id)
            || !facets.insert((&triangle.source_face_id, facet))
            || facet[0] == facet[1]
            || facet[1] == facet[2]
            || triangle.node_ids.iter().any(|identity| {
                nodes.get(identity).is_none_or(|node| {
                    !node.uses.iter().any(|use_record| {
                        use_record.source_face_id == triangle.source_face_id
                            && use_record.chart_id == triangle.chart_id
                    })
                })
            })
            || triangle.acceptance_sample_count == 0
            || !valid_triangle_measures(triangle)
        {
            return Err(invalid("published surface triangle is invalid"));
        }
        covered_faces.insert(&triangle.source_face_id);
        maximum_chordal = maximum_chordal.max(triangle.accepted_chordal_deviation_m);
        maximum_normal = maximum_normal.max(triangle.accepted_normal_deviation_rad);
    }
    if covered_faces != *faces
        || mesh.maximum_chordal_deviation_m != maximum_chordal
        || mesh.maximum_normal_deviation_rad != maximum_normal
    {
        return Err(invalid(
            "published surface face coverage or quality summary is invalid",
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
