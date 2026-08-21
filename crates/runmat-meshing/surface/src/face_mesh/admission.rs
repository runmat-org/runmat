use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityKind};
use runmat_meshing_core::StableDigest;

use super::{
    identity::{canonical_triangle, exact_face_triangle_id},
    ExactFaceJoinError, ExactFaceJoinErrorKind, ExactFaceMesh, EXACT_FACE_MESH_SCHEMA_VERSION,
};

pub(crate) fn validate_exact_face_mesh_contract(
    mesh: &ExactFaceMesh,
    topology: &ExactBRepTopology,
) -> Result<(), ExactFaceJoinError> {
    if mesh.schema_version != EXACT_FACE_MESH_SCHEMA_VERSION
        || !topology
            .faces
            .iter()
            .any(|face| face.id == mesh.source_face_id)
        || mesh.nodes.is_empty()
        || mesh.triangles.is_empty()
    {
        return Err(invalid(
            mesh,
            "face mesh schema, owner, or inventory is invalid",
        ));
    }
    let mut nodes = BTreeMap::new();
    let coedges = topology
        .coedges
        .iter()
        .map(|coedge| (&coedge.id, coedge))
        .collect::<BTreeMap<_, _>>();
    for node in &mesh.nodes {
        if node.node_id == StableDigest::ZERO
            || node.point_m.iter().any(|value| !value.is_finite())
            || node.uses.is_empty()
            || nodes.insert(node.node_id, node).is_some()
        {
            return Err(invalid(
                mesh,
                "face mesh node identity or geometry is invalid",
            ));
        }
        for use_record in &node.uses {
            if use_record.source_face_id != mesh.source_face_id
                || use_record.chart_id == StableDigest::ZERO
                || use_record
                    .uv
                    .iter()
                    .chain(&use_record.evaluator_uv)
                    .any(|value| !value.is_finite())
            {
                return Err(invalid(mesh, "face mesh node use is invalid"));
            }
            let mut edge_parameters = BTreeSet::new();
            for parameter in &use_record.exact_edge_parameters {
                if parameter.source_coedge_id.kind != PersistentEntityKind::Coedge
                    || parameter.source_edge_id.kind != PersistentEntityKind::Edge
                    || coedges
                        .get(&parameter.source_coedge_id)
                        .is_none_or(|coedge| {
                            coedge.face_id != mesh.source_face_id
                                || coedge.edge_id != parameter.source_edge_id
                        })
                    || !parameter.parameter.is_finite()
                    || !edge_parameters.insert((
                        &parameter.source_coedge_id,
                        &parameter.source_edge_id,
                        parameter.parameter.to_bits(),
                    ))
                {
                    return Err(invalid(mesh, "face mesh exact edge parameter is invalid"));
                }
            }
        }
    }
    if !mesh
        .nodes
        .windows(2)
        .all(|pair| pair[0].node_id < pair[1].node_id)
    {
        return Err(invalid(
            mesh,
            "face mesh nodes are not in canonical identity order",
        ));
    }

    let mut triangle_ids = BTreeSet::new();
    let mut triangle_facets = BTreeSet::new();
    let mut maximum_chordal = 0.0_f64;
    let mut maximum_normal = 0.0_f64;
    for triangle in &mesh.triangles {
        let (canonical, _) = canonical_triangle(triangle.node_ids);
        let mut facet = triangle.node_ids;
        facet.sort();
        if triangle.source_face_id != mesh.source_face_id
            || triangle.chart_id == StableDigest::ZERO
            || triangle.triangle_id != exact_face_triangle_id(triangle.chart_id, triangle.node_ids)
            || canonical != triangle.node_ids
            || !triangle_ids.insert(triangle.triangle_id)
            || !triangle_facets.insert(facet)
            || facet[0] == facet[1]
            || facet[1] == facet[2]
            || triangle.node_ids.iter().any(|node_id| {
                nodes.get(node_id).is_none_or(|node| {
                    !node.uses.iter().any(|use_record| {
                        use_record.source_face_id == mesh.source_face_id
                            && use_record.chart_id == triangle.chart_id
                    })
                })
            })
            || triangle.acceptance_sample_count == 0
            || !valid_triangle_measures(triangle)
        {
            return Err(invalid(mesh, "face mesh triangle contract is invalid"));
        }
        maximum_chordal = maximum_chordal.max(triangle.accepted_chordal_deviation_m);
        maximum_normal = maximum_normal.max(triangle.accepted_normal_deviation_rad);
    }
    let mut boundaries = BTreeSet::new();
    for segment in &mesh.boundary_segments {
        if segment.source_coedge_id.kind != PersistentEntityKind::Coedge
            || segment.source_edge_id.kind != PersistentEntityKind::Edge
            || coedges.get(&segment.source_coedge_id).is_none_or(|coedge| {
                coedge.face_id != mesh.source_face_id || coedge.edge_id != segment.source_edge_id
            })
            || segment.node_ids[0] == segment.node_ids[1]
            || segment
                .node_ids
                .iter()
                .any(|node_id| !nodes.contains_key(node_id))
            || segment
                .edge_parameters
                .iter()
                .any(|value| !value.is_finite())
            || !boundaries.insert((
                &segment.source_coedge_id,
                &segment.source_edge_id,
                segment.node_ids,
                segment.edge_parameters.map(f64::to_bits),
            ))
        {
            return Err(invalid(mesh, "face mesh boundary segment is invalid"));
        }
    }
    if mesh
        .joined_chart_cuts
        .windows(2)
        .any(|pair| pair[0].cut_id >= pair[1].cut_id)
        || mesh
            .joined_chart_cuts
            .iter()
            .any(|cut| cut.cut_id == StableDigest::ZERO || cut.piece_count == 0)
        || mesh.maximum_chordal_deviation_m != maximum_chordal
        || mesh.maximum_normal_deviation_rad != maximum_normal
    {
        return Err(invalid(
            mesh,
            "face mesh join or quality summary is invalid",
        ));
    }
    Ok(())
}

pub(crate) fn valid_triangle_measures(triangle: &super::ExactFaceMeshTriangle) -> bool {
    triangle
        .unit_normal
        .into_iter()
        .chain(triangle.metric_edge_lengths)
        .chain([
            triangle.physical_area_m2,
            triangle.minimum_metric_angle_rad,
            triangle.physical_aspect_ratio,
            triangle.chordal_deviation_m,
            triangle.normal_deviation_rad,
            triangle.accepted_chordal_deviation_m,
            triangle.accepted_normal_deviation_rad,
        ])
        .all(f64::is_finite)
        && triangle.physical_area_m2 > 0.0
        && triangle
            .metric_edge_lengths
            .iter()
            .all(|value| *value > 0.0)
        && triangle.minimum_metric_angle_rad > 0.0
        && triangle.physical_aspect_ratio > 0.0
        && triangle.chordal_deviation_m >= 0.0
        && triangle.normal_deviation_rad >= 0.0
        && triangle.accepted_chordal_deviation_m >= 0.0
        && triangle.accepted_normal_deviation_rad >= 0.0
}

fn invalid(mesh: &ExactFaceMesh, reason: &str) -> ExactFaceJoinError {
    ExactFaceJoinError::new(
        ExactFaceJoinErrorKind::InvalidInput,
        &mesh.source_face_id,
        reason,
    )
}
