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
    CadEdge, CadEntityId, CadEntityKind, CadFace, CadLoop, CadShell, CadTopologyError,
    CadTopologyModel, CadTopologyReport, CadTopologySource, CadVertex, CadVolume,
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
                loop_ids: Vec::new(),
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
    let mut faces = merge_stable_cad_faces(face_seeds);
    let loops = attach_outer_loops(&mut faces);

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
        loop_count: loops.len(),
        hole_loop_count: loops.iter().filter(|cad_loop| !cad_loop.is_outer).count(),
        closed_shell_count: usize::from(closed),
    };

    let model = CadTopologyModel {
        source_geometry_id: topology.source_geometry_id.clone(),
        source_geometry_revision: topology.source_geometry_revision,
        source_geometry_sha256: topology.source_geometry_sha256.clone(),
        source,
        vertices,
        edges,
        loops,
        faces,
        shells: vec![shell],
        volumes: closed.then_some(volume).into_iter().collect(),
        report,
    };
    validate_cad_topology_model(&model)?;
    Ok(model)
}

fn attach_outer_loops(faces: &mut [CadFace]) -> Vec<CadLoop> {
    let mut loops = Vec::<CadLoop>::with_capacity(faces.len());
    for face in faces {
        let loop_id = format!("cad_loop_{}_outer", face.entity_id.id);
        face.loop_ids = vec![loop_id.clone()];
        loops.push(CadLoop {
            entity_id: CadEntityId {
                kind: CadEntityKind::Loop,
                id: loop_id,
            },
            face_id: face.entity_id.id.clone(),
            edge_ids: face.loop_edge_ids.clone(),
            is_outer: true,
        });
    }
    loops
}

pub fn validate_cad_topology_model(model: &CadTopologyModel) -> Result<(), CadTopologyError> {
    let vertex_ids = collect_entity_ids(
        CadEntityKind::Vertex,
        model.vertices.iter().map(|vertex| &vertex.entity_id),
    )?;
    let edge_ids = collect_entity_ids(
        CadEntityKind::Edge,
        model.edges.iter().map(|edge| &edge.entity_id),
    )?;
    let loop_ids = collect_entity_ids(
        CadEntityKind::Loop,
        model.loops.iter().map(|cad_loop| &cad_loop.entity_id),
    )?;
    let face_ids = collect_entity_ids(
        CadEntityKind::Face,
        model.faces.iter().map(|face| &face.entity_id),
    )?;
    let shell_ids = collect_entity_ids(
        CadEntityKind::Shell,
        model.shells.iter().map(|shell| &shell.entity_id),
    )?;
    let _volume_ids = collect_entity_ids(
        CadEntityKind::Volume,
        model.volumes.iter().map(|volume| &volume.entity_id),
    )?;
    let loop_face_ids = model
        .loops
        .iter()
        .map(|cad_loop| (cad_loop.entity_id.id.as_str(), cad_loop.face_id.as_str()))
        .collect::<BTreeMap<_, _>>();
    let face_loop_ids = model
        .faces
        .iter()
        .map(|face| {
            (
                face.entity_id.id.as_str(),
                face.loop_ids
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    for edge in &model.edges {
        for vertex_id in &edge.vertex_ids {
            require_reference(
                CadEntityKind::Edge,
                &edge.entity_id.id,
                CadEntityKind::Vertex,
                vertex_id,
                &vertex_ids,
            )?;
        }
        for face_id in &edge.adjacent_face_ids {
            require_reference(
                CadEntityKind::Edge,
                &edge.entity_id.id,
                CadEntityKind::Face,
                face_id,
                &face_ids,
            )?;
        }
    }
    for cad_loop in &model.loops {
        require_reference(
            CadEntityKind::Loop,
            &cad_loop.entity_id.id,
            CadEntityKind::Face,
            &cad_loop.face_id,
            &face_ids,
        )?;
        require_face_lists_loop(&cad_loop.face_id, &cad_loop.entity_id.id, &face_loop_ids)?;
        for edge_id in &cad_loop.edge_ids {
            require_reference(
                CadEntityKind::Loop,
                &cad_loop.entity_id.id,
                CadEntityKind::Edge,
                edge_id,
                &edge_ids,
            )?;
        }
    }
    for face in &model.faces {
        for loop_id in &face.loop_ids {
            require_reference(
                CadEntityKind::Face,
                &face.entity_id.id,
                CadEntityKind::Loop,
                loop_id,
                &loop_ids,
            )?;
            require_loop_belongs_to_face(&face.entity_id.id, loop_id, &loop_face_ids)?;
        }
        for edge_id in &face.loop_edge_ids {
            require_reference(
                CadEntityKind::Face,
                &face.entity_id.id,
                CadEntityKind::Edge,
                edge_id,
                &edge_ids,
            )?;
        }
    }
    for shell in &model.shells {
        for face_id in &shell.face_ids {
            require_reference(
                CadEntityKind::Shell,
                &shell.entity_id.id,
                CadEntityKind::Face,
                face_id,
                &face_ids,
            )?;
        }
    }
    for volume in &model.volumes {
        for shell_id in &volume.shell_ids {
            require_reference(
                CadEntityKind::Volume,
                &volume.entity_id.id,
                CadEntityKind::Shell,
                shell_id,
                &shell_ids,
            )?;
        }
    }

    validate_report_count(
        "vertex_count",
        model.vertices.len(),
        model.report.vertex_count,
    )?;
    validate_report_count("edge_count", model.edges.len(), model.report.edge_count)?;
    validate_report_count("loop_count", model.loops.len(), model.report.loop_count)?;
    validate_report_count("face_count", model.faces.len(), model.report.face_count)?;
    validate_report_count("shell_count", model.shells.len(), model.report.shell_count)?;
    validate_report_count(
        "volume_count",
        model.volumes.len(),
        model.report.volume_count,
    )?;
    validate_report_count(
        "hole_loop_count",
        model
            .loops
            .iter()
            .filter(|cad_loop| !cad_loop.is_outer)
            .count(),
        model.report.hole_loop_count,
    )?;
    validate_report_count(
        "closed_shell_count",
        model.shells.iter().filter(|shell| shell.closed).count(),
        model.report.closed_shell_count,
    )?;
    Ok(())
}

fn require_face_lists_loop(
    face_id: &str,
    loop_id: &str,
    face_loop_ids: &BTreeMap<&str, BTreeSet<&str>>,
) -> Result<(), CadTopologyError> {
    if face_loop_ids
        .get(face_id)
        .is_some_and(|loop_ids| loop_ids.contains(loop_id))
    {
        Ok(())
    } else {
        Err(CadTopologyError::MissingFaceLoopReference {
            face_id: face_id.to_string(),
            loop_id: loop_id.to_string(),
        })
    }
}

fn require_loop_belongs_to_face(
    face_id: &str,
    loop_id: &str,
    loop_face_ids: &BTreeMap<&str, &str>,
) -> Result<(), CadTopologyError> {
    match loop_face_ids.get(loop_id) {
        Some(loop_face_id) if *loop_face_id == face_id => Ok(()),
        Some(loop_face_id) => Err(CadTopologyError::LoopFaceMismatch {
            loop_id: loop_id.to_string(),
            expected_face_id: face_id.to_string(),
            actual_face_id: (*loop_face_id).to_string(),
        }),
        None => Err(CadTopologyError::MissingEntityReference {
            owner_kind: CadEntityKind::Face,
            owner_id: face_id.to_string(),
            reference_kind: CadEntityKind::Loop,
            reference_id: loop_id.to_string(),
        }),
    }
}

fn collect_entity_ids<'a>(
    expected_kind: CadEntityKind,
    ids: impl Iterator<Item = &'a CadEntityId>,
) -> Result<BTreeSet<String>, CadTopologyError> {
    let mut seen = BTreeSet::<String>::new();
    for entity_id in ids {
        if entity_id.kind != expected_kind {
            return Err(CadTopologyError::EntityKindMismatch {
                expected: expected_kind,
                actual: entity_id.kind,
                id: entity_id.id.clone(),
            });
        }
        if !seen.insert(entity_id.id.clone()) {
            return Err(CadTopologyError::DuplicateEntityId {
                kind: expected_kind,
                id: entity_id.id.clone(),
            });
        }
    }
    Ok(seen)
}

fn require_reference(
    owner_kind: CadEntityKind,
    owner_id: &str,
    reference_kind: CadEntityKind,
    reference_id: &str,
    valid_ids: &BTreeSet<String>,
) -> Result<(), CadTopologyError> {
    if valid_ids.contains(reference_id) {
        Ok(())
    } else {
        Err(CadTopologyError::MissingEntityReference {
            owner_kind,
            owner_id: owner_id.to_string(),
            reference_kind,
            reference_id: reference_id.to_string(),
        })
    }
}

fn validate_report_count(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), CadTopologyError> {
    if expected == actual {
        Ok(())
    } else {
        Err(CadTopologyError::ReportCountMismatch {
            field,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests;
