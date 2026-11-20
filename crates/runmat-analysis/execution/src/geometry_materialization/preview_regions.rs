//! Study regions drawn over a materialized geometry preview.
//!
//! A resolved region names exact entities at any topology level. The preview draws faces,
//! so each region becomes the triangle ranges of the faces that contain its entities.
use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    EntityIdRange, ExactBRepTopology, GeometryAsset, PersistentEntityId, Region,
    RegionEntityMapping, ResolvedGeometryRegions,
};

pub(super) fn add_materialized_regions(
    asset: &mut GeometryAsset,
    resolved: &ResolvedGeometryRegions,
    topology: &ExactBRepTopology,
    source_face_ids: &BTreeMap<u64, PersistentEntityId>,
) {
    let Some(mesh_id) = asset
        .surface_meshes
        .first()
        .map(|mesh| mesh.mesh_id.clone())
    else {
        return;
    };
    for (region_id, region) in resolved {
        if asset
            .regions
            .iter()
            .any(|item| item.region_id == *region_id)
        {
            continue;
        }
        let face_indices = region
            .entity_ids
            .iter()
            .flat_map(|entity| incident_face_indices(entity, topology, source_face_ids))
            .collect::<BTreeSet<_>>();
        let ranges = merge_ranges(face_indices.into_iter().flat_map(|face_index| {
            let face_region_id = source_face_ids
                .get(&(face_index as u64))
                .and_then(|entity| entity.analysis_selection_id().ok());
            asset
                .region_entity_mappings
                .iter()
                .filter(move |mapping| {
                    Some(mapping.region_id.as_str()) == face_region_id.as_deref()
                })
                .flat_map(|mapping| mapping.ranges.iter().copied())
        }));
        if ranges.is_empty() {
            continue;
        }
        asset.regions.push(Region {
            region_id: region_id.clone(),
            name: region_id.clone(),
            tag: None,
            cad_ownership: None,
        });
        asset.region_entity_mappings.push(RegionEntityMapping::new(
            region_id.clone(),
            mesh_id.clone(),
            runmat_geometry_core::EntityKind::Face,
            ranges,
        ));
    }
}

fn incident_face_indices(
    entity: &PersistentEntityId,
    topology: &ExactBRepTopology,
    source_face_ids: &BTreeMap<u64, PersistentEntityId>,
) -> Vec<usize> {
    source_face_ids
        .iter()
        .filter_map(|(ordinal, face_id)| {
            let face = topology.faces.iter().find(|face| face.id == *face_id)?;
            face_contains_entity(face, entity, topology).then_some(*ordinal as usize)
        })
        .collect()
}

fn face_contains_entity(
    face: &runmat_geometry_core::ExactFace,
    entity: &PersistentEntityId,
    topology: &ExactBRepTopology,
) -> bool {
    if face.id == *entity
        || face.outer_wire_id == *entity
        || face.inner_wire_ids.iter().any(|id| id == entity)
    {
        return true;
    }
    let face_coedges = topology
        .coedges
        .iter()
        .filter(|item| item.face_id == face.id);
    for coedge in face_coedges {
        if coedge.id == *entity || coedge.edge_id == *entity {
            return true;
        }
        if topology.edges.iter().any(|edge| {
            edge.id == coedge.edge_id
                && (edge.start_vertex_id.as_ref() == Some(entity)
                    || edge.end_vertex_id.as_ref() == Some(entity))
        }) {
            return true;
        }
    }
    topology.shells.iter().any(|shell| {
        shell
            .face_uses
            .iter()
            .any(|face_use| face_use.entity_id == face.id)
            && (shell.id == *entity || shell_owned_by_entity(shell, entity, topology))
    })
}

fn shell_owned_by_entity(
    shell: &runmat_geometry_core::ExactShell,
    entity: &PersistentEntityId,
    topology: &ExactBRepTopology,
) -> bool {
    let owned_by_solid_body = topology.solids.iter().any(|solid| {
        let owns_shell = solid.outer_shell_id == shell.id
            || solid.void_shell_ids.iter().any(|id| id == &shell.id);
        owns_shell
            && (solid.id == *entity
                || topology
                    .regions
                    .iter()
                    .any(|region| region.solid_id == solid.id && region.id == *entity)
                || topology.lumps.iter().any(|lump| {
                    lump.solid_ids.iter().any(|id| id == &solid.id)
                        && (lump.id == *entity
                            || topology.bodies.iter().any(|body| {
                                body.lump_ids.iter().any(|id| id == &lump.id) && body.id == *entity
                            }))
                }))
    });
    owned_by_solid_body
        || topology
            .bodies
            .iter()
            .any(|body| body.id == *entity && body.sheet_shell_ids.iter().any(|id| id == &shell.id))
}

fn merge_ranges(ranges: impl Iterator<Item = EntityIdRange>) -> Vec<EntityIdRange> {
    let mut ranges = ranges.collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.start);
    let mut merged: Vec<EntityIdRange> = Vec::new();
    for range in ranges {
        if let Some(previous) = merged.last_mut() {
            let previous_end = previous.end_exclusive().unwrap_or(u64::MAX);
            let range_end = range.end_exclusive().unwrap_or(u64::MAX);
            if range.start <= previous_end {
                previous.count = previous_end.max(range_end).saturating_sub(previous.start);
                continue;
            }
        }
        merged.push(range);
    }
    merged
}

#[cfg(test)]
mod tests {
    use runmat_geometry_core::{
        CurveEvaluatorId, ExactBody, ExactCoedge, ExactEdge, ExactFace, ExactLump, ExactRegion,
        ExactShell, ExactSolid, ExactVertex, ExactWire, MassPropertiesEvaluatorId,
        OrientedEntityUse, PcurveEvaluatorId, PersistentEntityKind, SurfaceEvaluatorId,
        TopologicalOrientation, TrimClassifierId, EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
    };

    use super::*;

    fn id(kind: PersistentEntityKind, value: &str) -> PersistentEntityId {
        PersistentEntityId::new(kind, value.to_string(), vec!["root".to_string()])
    }

    fn topology(sheet_body: bool) -> ExactBRepTopology {
        let root = id(PersistentEntityKind::Assembly, "root");
        let body = id(PersistentEntityKind::Body, "body");
        let lump = id(PersistentEntityKind::Lump, "lump");
        let solid = id(PersistentEntityKind::Solid, "solid");
        let region = id(PersistentEntityKind::Region, "region");
        let shell = id(PersistentEntityKind::Shell, "shell");
        let face = id(PersistentEntityKind::Face, "face");
        let wire = id(PersistentEntityKind::Wire, "wire");
        let coedge = id(PersistentEntityKind::Coedge, "coedge");
        let edge = id(PersistentEntityKind::Edge, "edge");
        let vertex = id(PersistentEntityKind::Vertex, "vertex");
        ExactBRepTopology {
            schema_version: EXACT_BREP_TOPOLOGY_SCHEMA_VERSION,
            root_assembly_id: root,
            assemblies: Vec::new(),
            instances: Vec::new(),
            bodies: vec![ExactBody {
                id: body,
                mass_properties_evaluator_id: MassPropertiesEvaluatorId::new("mass:body")
                    .expect("valid evaluator ID"),
                lump_ids: (!sheet_body).then_some(lump.clone()).into_iter().collect(),
                is_sheet_body: sheet_body,
                sheet_shell_ids: sheet_body.then_some(shell.clone()).into_iter().collect(),
            }],
            lumps: (!sheet_body)
                .then_some(ExactLump {
                    id: lump,
                    solid_ids: vec![solid.clone()],
                })
                .into_iter()
                .collect(),
            solids: (!sheet_body)
                .then_some(ExactSolid {
                    id: solid.clone(),
                    outer_shell_id: shell.clone(),
                    void_shell_ids: Vec::new(),
                })
                .into_iter()
                .collect(),
            regions: (!sheet_body)
                .then_some(ExactRegion {
                    id: region,
                    solid_id: solid,
                })
                .into_iter()
                .collect(),
            shells: vec![ExactShell {
                id: shell,
                orientation: TopologicalOrientation::Forward,
                face_uses: vec![OrientedEntityUse {
                    entity_id: face.clone(),
                    orientation: TopologicalOrientation::Forward,
                }],
            }],
            faces: vec![ExactFace {
                id: face.clone(),
                orientation: TopologicalOrientation::Forward,
                surface_evaluator_id: SurfaceEvaluatorId::new("surface:face")
                    .expect("valid evaluator ID"),
                trim_classifier_id: TrimClassifierId::new("trim:face").expect("valid evaluator ID"),
                outer_wire_id: wire.clone(),
                inner_wire_ids: Vec::new(),
                periodic_u: false,
                periodic_v: false,
                has_singularity: false,
            }],
            wires: vec![ExactWire {
                id: wire,
                orientation: TopologicalOrientation::Forward,
                coedge_ids: vec![coedge.clone()],
            }],
            coedges: vec![ExactCoedge {
                id: coedge,
                face_id: face,
                edge_id: edge.clone(),
                orientation: TopologicalOrientation::Forward,
                pcurve_evaluator_id: PcurveEvaluatorId::new("pcurve:edge")
                    .expect("valid evaluator ID"),
                seam_image: None,
            }],
            edges: vec![ExactEdge {
                id: edge,
                curve_evaluator_id: CurveEvaluatorId::new("curve:edge")
                    .expect("valid evaluator ID"),
                start_vertex_id: Some(vertex.clone()),
                end_vertex_id: Some(vertex.clone()),
                tolerance_m: runmat_geometry_core::physical::Length::new(1.0e-8),
                is_closed: true,
                is_periodic: false,
                is_degenerate: false,
            }],
            vertices: vec![ExactVertex {
                id: vertex,
                point_m: runmat_geometry_core::physical::Position::new([0.0, 0.0, 0.0]),
                tolerance_m: runmat_geometry_core::physical::Length::new(1.0e-8),
            }],
            interfaces: Vec::new(),
            contacts: Vec::new(),
        }
    }

    fn source_face_ids() -> BTreeMap<u64, PersistentEntityId> {
        BTreeMap::from([(0, id(PersistentEntityKind::Face, "face"))])
    }

    #[test]
    fn projects_every_solid_topology_level_to_its_incident_face() {
        let topology = topology(false);
        let source_face_ids = source_face_ids();
        for (kind, source_topology_id) in [
            (PersistentEntityKind::Face, "face"),
            (PersistentEntityKind::Wire, "wire"),
            (PersistentEntityKind::Coedge, "coedge"),
            (PersistentEntityKind::Edge, "edge"),
            (PersistentEntityKind::Vertex, "vertex"),
            (PersistentEntityKind::Shell, "shell"),
            (PersistentEntityKind::Solid, "solid"),
            (PersistentEntityKind::Region, "region"),
            (PersistentEntityKind::Lump, "lump"),
            (PersistentEntityKind::Body, "body"),
        ] {
            assert_eq!(
                incident_face_indices(&id(kind, source_topology_id), &topology, &source_face_ids,),
                vec![0],
                "{source_topology_id} should project to its incident face"
            );
        }
    }

    #[test]
    fn projects_sheet_body_to_its_owned_shell_faces() {
        let topology = topology(true);
        let source_face_ids = source_face_ids();
        assert_eq!(
            incident_face_indices(
                &id(PersistentEntityKind::Body, "body"),
                &topology,
                &source_face_ids,
            ),
            vec![0]
        );
    }

    #[test]
    fn joins_display_ordinals_to_faces_by_persistent_identity() {
        let mut topology = topology(false);
        let original_face = topology.faces[0].id.clone();
        let mut other_face = topology.faces[0].clone();
        other_face.id = id(PersistentEntityKind::Face, "other-face");
        topology.faces.insert(0, other_face.clone());
        let source_face_ids = BTreeMap::from([(0, other_face.id), (1, original_face.clone())]);

        assert_eq!(
            incident_face_indices(&original_face, &topology, &source_face_ids),
            vec![1]
        );
    }

    #[test]
    fn ignores_entities_outside_the_face_ownership_chain() {
        let topology = topology(false);
        let source_face_ids = source_face_ids();
        assert!(incident_face_indices(
            &id(PersistentEntityKind::Face, "different-face"),
            &topology,
            &source_face_ids,
        )
        .is_empty());
    }

    #[test]
    fn merges_overlapping_and_adjacent_triangle_ranges() {
        assert_eq!(
            merge_ranges(
                [
                    EntityIdRange { start: 8, count: 2 },
                    EntityIdRange { start: 2, count: 3 },
                    EntityIdRange { start: 5, count: 3 },
                    EntityIdRange {
                        start: 20,
                        count: 1,
                    },
                ]
                .into_iter()
            ),
            vec![
                EntityIdRange { start: 2, count: 8 },
                EntityIdRange {
                    start: 20,
                    count: 1,
                },
            ]
        );
    }
}
