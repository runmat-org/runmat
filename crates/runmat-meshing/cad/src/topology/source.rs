use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::GeometryAsset;

use super::SourceTopologyModel;

mod generic;
mod merge;
mod semantics;
mod types;

use generic::generic_coplanar_face_ids;
use merge::merge_stable_cad_faces;
use semantics::{
    cad_topology_source, evaluator_faces_by_imported_id, face_regions_by_source_face,
    imported_face_id_for_regions, imported_face_ids_by_region, semantic_face_regions,
};
pub use types::{
    CadEdge, CadEntityId, CadEntityKind, CadFace, CadShell, CadTopologyError, CadTopologyModel,
    CadTopologyReport, CadTopologySource, CadVertex, CadVolume,
};

pub fn build_cad_topology(
    geometry: &GeometryAsset,
    topology: &SourceTopologyModel,
) -> Result<CadTopologyModel, CadTopologyError> {
    if topology.vertices.is_empty() || topology.faces.is_empty() {
        return Err(CadTopologyError::EmptyTopology);
    }

    let source = cad_topology_source(geometry);
    let semantic_face_regions = semantic_face_regions(geometry);
    let imported_face_ids_by_region = imported_face_ids_by_region(geometry);
    let evaluator_faces = evaluator_faces_by_imported_id(geometry);
    let face_region_by_source_face = face_regions_by_source_face(geometry);
    let generic_face_ids_by_source_face = if source == CadTopologySource::GenericCadMesh {
        generic_coplanar_face_ids(topology, &face_region_by_source_face)
    } else {
        BTreeMap::new()
    };

    let vertices = topology
        .vertices
        .iter()
        .map(|vertex| CadVertex {
            entity_id: CadEntityId {
                kind: CadEntityKind::Vertex,
                id: format!("cad_vertex_{}", vertex.vertex_id),
            },
            source_vertex_id: vertex.vertex_id,
            coordinates_m: vertex.coordinates_m,
        })
        .collect::<Vec<_>>();

    let face_seeds = topology
        .faces
        .iter()
        .map(|face| {
            let mapped_region_ids = face_region_by_source_face
                .get(&face.source_triangle_id)
                .cloned();
            let identity_region_ids = mapped_region_ids.clone().unwrap_or_default();
            let region_ids = mapped_region_ids.unwrap_or_else(|| face.region_ids.clone());
            let imported_face_id =
                imported_face_id_for_regions(&imported_face_ids_by_region, &identity_region_ids);
            let evaluator_face = imported_face_id.and_then(|face_id| evaluator_faces.get(&face_id));
            let entity_id = identity_region_ids
                .iter()
                .find(|region_id| semantic_face_regions.contains(*region_id))
                .cloned()
                .or_else(|| generic_face_ids_by_source_face.get(&face.face_id).cloned())
                .unwrap_or_else(|| format!("cad_face_{}", face.face_id));
            CadFace {
                entity_id: CadEntityId {
                    kind: CadEntityKind::Face,
                    id: entity_id,
                },
                imported_face_id,
                evaluator_id: evaluator_face.map(|face| face.evaluator_id.clone()),
                evaluator_supports_point_evaluation: evaluator_face
                    .is_some_and(|face| face.supports_point_evaluation),
                evaluator_supports_projection: evaluator_face
                    .is_some_and(|face| face.supports_projection),
                evaluator_supports_normal: evaluator_face.is_some_and(|face| face.supports_normal),
                evaluator_supports_derivatives: evaluator_face
                    .is_some_and(|face| face.supports_derivatives),
                evaluator_supports_curvature: evaluator_face
                    .is_some_and(|face| face.supports_curvature),
                evaluator_reference_point_m: evaluator_face.and_then(|face| face.reference_point_m),
                evaluator_unit_normal: evaluator_face.and_then(|face| face.reference_unit_normal),
                evaluator_samples: evaluator_face
                    .map(|face| face.evaluation_samples.clone())
                    .unwrap_or_default(),
                source_face_ids: vec![face.face_id],
                source_edge_ids: face.edge_ids.to_vec(),
                loop_edge_ids: face
                    .edge_ids
                    .iter()
                    .map(|edge_id| format!("cad_edge_{edge_id}"))
                    .collect(),
                region_ids,
                area_m2: face.area_m2,
                unit_normal: face.unit_normal,
            }
        })
        .collect::<Vec<_>>();
    let faces = merge_stable_cad_faces(face_seeds);

    let face_ids_by_source_face = faces
        .iter()
        .flat_map(|face| {
            face.source_face_ids
                .iter()
                .map(|source_face_id| (*source_face_id, face.entity_id.id.clone()))
                .collect::<Vec<_>>()
        })
        .collect::<BTreeMap<_, _>>();
    let edges = topology
        .edges
        .iter()
        .map(|edge| CadEdge {
            entity_id: CadEntityId {
                kind: CadEntityKind::Edge,
                id: format!("cad_edge_{}", edge.edge_id),
            },
            source_edge_id: edge.edge_id,
            vertex_ids: [
                format!("cad_vertex_{}", edge.node_ids[0]),
                format!("cad_vertex_{}", edge.node_ids[1]),
            ],
            adjacent_face_ids: edge
                .adjacent_face_ids
                .iter()
                .filter_map(|face_id| face_ids_by_source_face.get(face_id).cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            length_m: edge.length_m,
        })
        .collect::<Vec<_>>();

    let closed = topology
        .edges
        .iter()
        .all(|edge| edge.adjacent_face_ids.len() == 2);
    let shell = CadShell {
        entity_id: CadEntityId {
            kind: CadEntityKind::Shell,
            id: "cad_shell_0".to_string(),
        },
        face_ids: faces.iter().map(|face| face.entity_id.id.clone()).collect(),
        closed,
    };
    let volume = CadVolume {
        entity_id: CadEntityId {
            kind: CadEntityKind::Volume,
            id: "cad_volume_0".to_string(),
        },
        shell_ids: vec![shell.entity_id.id.clone()],
        region_ids: topology.region_ids.clone(),
    };

    let semantic_face_count = faces
        .iter()
        .filter(|face| semantic_face_regions.contains(&face.entity_id.id))
        .count();
    let imported_face_count = faces
        .iter()
        .filter(|face| face.imported_face_id.is_some())
        .count();
    let evaluator_face_count = faces
        .iter()
        .filter(|face| {
            face.imported_face_id
                .is_some_and(|face_id| evaluator_faces.contains_key(&face_id))
        })
        .count();
    let generic_face_count = faces.len().saturating_sub(semantic_face_count);
    let report = CadTopologyReport {
        source,
        vertex_count: vertices.len(),
        edge_count: edges.len(),
        face_count: faces.len(),
        shell_count: 1,
        volume_count: usize::from(closed),
        semantic_face_count,
        imported_face_count,
        evaluator_face_count,
        generic_face_count,
        closed_shell_count: usize::from(closed),
    };

    Ok(CadTopologyModel {
        source_geometry_id: topology.source_geometry_id.clone(),
        source_geometry_revision: topology.source_geometry_revision,
        source_geometry_sha256: topology.source_geometry_sha256.clone(),
        source,
        vertices,
        edges,
        faces,
        shells: vec![shell],
        volumes: closed.then_some(volume).into_iter().collect(),
        report,
    })
}

#[cfg(test)]
mod tests;
