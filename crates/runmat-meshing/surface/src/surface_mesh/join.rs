use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId, TopologicalOrientation};
use runmat_meshing_core::StableDigest;

use super::{
    validate_exact_face_mesh_batch, ExactFaceMeshBatch, ExactSurfaceJoinOptions, ExactSurfaceMesh,
    ExactSurfaceMeshError, ExactSurfaceMeshErrorKind, ExactSurfaceShellEvidence,
    EXACT_SURFACE_MESH_SCHEMA_VERSION, MAX_EXACT_FACE_PARTITIONS,
};
use crate::{ExactFaceMeshBoundarySegment, ExactFaceMeshNode};

pub fn join_exact_face_mesh_batches(
    topology: &ExactBRepTopology,
    mut batches: Vec<ExactFaceMeshBatch>,
    options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    validate_options(options)?;
    if batches.is_empty() || batches.len() > MAX_EXACT_FACE_PARTITIONS {
        return Err(invalid(
            "surface join requires a bounded nonempty partition set",
        ));
    }
    batches.sort_by_key(|batch| batch.partition.partition_index);
    for (index, batch) in batches.iter().enumerate() {
        validate_exact_face_mesh_batch(batch, topology)?;
        if batch.partition.partition_index != index as u32
            || batch.partition.partition_count != batches.len() as u32
        {
            return Err(invalid(
                "surface join partitions do not form one complete canonical set",
            ));
        }
    }
    let faces = batches
        .into_iter()
        .flat_map(|batch| batch.faces)
        .collect::<Vec<_>>();
    if faces.len() != topology.faces.len()
        || faces
            .iter()
            .zip(&topology.faces)
            .any(|(mesh, face)| mesh.source_face_id != face.id)
    {
        return Err(invalid(
            "surface join does not exactly cover the canonical face inventory",
        ));
    }
    topology
        .validate_solid_shell_boundaries()
        .map_err(|error| invalid(error.to_string()))?;

    let mut nodes = BTreeMap::<StableDigest, ExactFaceMeshNode>::new();
    let mut triangle_ids = BTreeSet::new();
    let mut triangles = Vec::new();
    let mut boundary_segments = Vec::new();
    let mut maximum_chordal_deviation_m = 0.0_f64;
    let mut maximum_normal_deviation_rad = 0.0_f64;
    for face in &faces {
        maximum_chordal_deviation_m =
            maximum_chordal_deviation_m.max(face.maximum_chordal_deviation_m);
        maximum_normal_deviation_rad =
            maximum_normal_deviation_rad.max(face.maximum_normal_deviation_rad);
        for node in &face.nodes {
            if nodes.len() as u64 >= options.maximum_nodes && !nodes.contains_key(&node.node_id) {
                return Err(limit("surface join exceeded its node hard limit"));
            }
            match nodes.entry(node.node_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(node.clone());
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if distance(entry.get().point_m, node.point_m) > options.coordinate_tolerance_m
                    {
                        return Err(invalid("shared face nodes disagree in exact 3D position")
                            .with_face(&face.source_face_id));
                    }
                    for use_record in &node.uses {
                        if entry.get().uses.iter().any(|existing| {
                            existing.source_face_id == use_record.source_face_id
                                && existing.chart_id == use_record.chart_id
                                && existing.uv == use_record.uv
                        }) {
                            return Err(invalid("surface join contains a duplicate node use")
                                .with_face(&face.source_face_id));
                        }
                        entry.get_mut().uses.push(use_record.clone());
                    }
                }
            }
        }
        for triangle in &face.triangles {
            if triangles.len() as u64 >= options.maximum_triangles {
                return Err(limit("surface join exceeded its triangle hard limit"));
            }
            if !triangle_ids.insert(triangle.triangle_id) {
                return Err(
                    invalid("surface join contains a duplicate triangle identity")
                        .with_face(&face.source_face_id),
                );
            }
            triangles.push(triangle.clone());
        }
        if boundary_segments.len() as u64 + face.boundary_segments.len() as u64
            > options.maximum_boundary_segments
        {
            return Err(limit(
                "surface join exceeded its boundary-segment hard limit",
            ));
        }
        boundary_segments.extend(face.boundary_segments.iter().cloned());
    }
    validate_edge_conformity(topology, &boundary_segments)?;
    let shells = shell_evidence(topology)?;
    let result = ExactSurfaceMesh {
        schema_version: EXACT_SURFACE_MESH_SCHEMA_VERSION,
        face_ids: faces.into_iter().map(|face| face.source_face_id).collect(),
        nodes: nodes.into_values().collect(),
        triangles,
        boundary_segments,
        shells,
        maximum_chordal_deviation_m,
        maximum_normal_deviation_rad,
    };
    Ok(result)
}

pub fn validate_exact_surface_mesh(
    result: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
) -> Result<(), ExactSurfaceMeshError> {
    let expected = join_exact_face_mesh_batches(topology, batches.to_vec(), options)?;
    if result != &expected {
        return Err(invalid(
            "exact surface mesh differs from the canonical face-batch join",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
struct CanonicalEdgePiece {
    edge_parameters: [f64; 2],
    node_ids: [StableDigest; 2],
}

pub(super) fn validate_edge_conformity(
    topology: &ExactBRepTopology,
    segments: &[ExactFaceMeshBoundarySegment],
) -> Result<(), ExactSurfaceMeshError> {
    let coedges = topology
        .coedges
        .iter()
        .map(|coedge| (&coedge.id, coedge))
        .collect::<BTreeMap<_, _>>();
    let mut coverage =
        BTreeMap::<PersistentEntityId, BTreeMap<PersistentEntityId, Vec<CanonicalEdgePiece>>>::new(
        );
    for segment in segments {
        let coedge = coedges
            .get(&segment.source_coedge_id)
            .ok_or_else(|| invalid("surface boundary references an absent exact coedge"))?;
        let mut piece = CanonicalEdgePiece {
            edge_parameters: segment.edge_parameters,
            node_ids: segment.node_ids,
        };
        if piece.edge_parameters[0] > piece.edge_parameters[1] {
            piece.edge_parameters.swap(0, 1);
            piece.node_ids.swap(0, 1);
        }
        if piece.edge_parameters[0] >= piece.edge_parameters[1] {
            return Err(
                invalid("surface boundary has a non-increasing exact edge interval")
                    .with_face(&coedge.face_id),
            );
        }
        coverage
            .entry(coedge.edge_id.clone())
            .or_default()
            .entry(coedge.id.clone())
            .or_default()
            .push(piece);
    }
    for by_coedge in coverage.values_mut() {
        for pieces in by_coedge.values_mut() {
            pieces.sort_by(|left, right| {
                left.edge_parameters[0]
                    .total_cmp(&right.edge_parameters[0])
                    .then_with(|| left.edge_parameters[1].total_cmp(&right.edge_parameters[1]))
                    .then_with(|| left.node_ids.cmp(&right.node_ids))
            });
        }
    }
    for edge in &topology.edges {
        if edge.is_degenerate {
            continue;
        }
        let expected = topology
            .coedges
            .iter()
            .filter(|coedge| coedge.edge_id == edge.id)
            .collect::<Vec<_>>();
        let Some(actual) = coverage.get(&edge.id) else {
            return Err(invalid("surface join is missing an exact edge"));
        };
        if actual.len() != expected.len()
            || expected
                .iter()
                .any(|coedge| !actual.contains_key(&coedge.id))
        {
            return Err(invalid("surface join has incomplete exact coedge coverage"));
        }
        let mut uses = actual.values();
        let first = uses.next().expect("nondegenerate edge has a coedge");
        if uses.any(|candidate| candidate != first) {
            return Err(invalid(
                "adjacent faces disagree on shared exact edge segmentation",
            ));
        }
    }
    Ok(())
}

pub(super) fn shell_evidence(
    topology: &ExactBRepTopology,
) -> Result<Vec<ExactSurfaceShellEvidence>, ExactSurfaceMeshError> {
    let sheet_shells = topology
        .bodies
        .iter()
        .flat_map(|body| &body.sheet_shell_ids)
        .collect::<BTreeSet<_>>();
    let solid_shells = topology
        .solids
        .iter()
        .flat_map(|solid| std::iter::once(&solid.outer_shell_id).chain(&solid.void_shell_ids))
        .collect::<BTreeSet<_>>();
    let faces = topology
        .faces
        .iter()
        .map(|face| (&face.id, face))
        .collect::<BTreeMap<_, _>>();
    let wires = topology
        .wires
        .iter()
        .map(|wire| (&wire.id, wire))
        .collect::<BTreeMap<_, _>>();
    let coedges = topology
        .coedges
        .iter()
        .map(|coedge| (&coedge.id, coedge))
        .collect::<BTreeMap<_, _>>();
    let edges = topology
        .edges
        .iter()
        .map(|edge| (&edge.id, edge))
        .collect::<BTreeMap<_, _>>();
    topology
        .shells
        .iter()
        .map(|shell| {
            let is_sheet_shell = sheet_shells.contains(&shell.id);
            if is_sheet_shell == solid_shells.contains(&shell.id) {
                return Err(
                    invalid("shell ownership is neither uniquely solid nor sheet")
                        .with_shell(&shell.id),
                );
            }
            let mut uses = BTreeMap::<PersistentEntityId, Vec<TopologicalOrientation>>::new();
            for face_use in &shell.face_uses {
                let face = faces.get(&face_use.entity_id).ok_or_else(|| {
                    invalid("shell references an absent face").with_shell(&shell.id)
                })?;
                for wire_id in std::iter::once(&face.outer_wire_id).chain(&face.inner_wire_ids) {
                    let wire = wires.get(wire_id).ok_or_else(|| {
                        invalid("face references an absent wire").with_shell(&shell.id)
                    })?;
                    for coedge_id in &wire.coedge_ids {
                        let coedge = coedges.get(coedge_id).ok_or_else(|| {
                            invalid("wire references an absent coedge").with_shell(&shell.id)
                        })?;
                        let edge = edges.get(&coedge.edge_id).ok_or_else(|| {
                            invalid("coedge references an absent edge").with_shell(&shell.id)
                        })?;
                        if !edge.is_degenerate {
                            uses.entry(edge.id.clone()).or_default().push(compose(
                                shell.orientation,
                                face_use.orientation,
                                coedge.orientation,
                            ));
                        }
                    }
                }
            }
            let shared_edge_count = uses
                .values()
                .filter(|uses| uses.len() == 2 && uses[0] != uses[1])
                .count() as u64;
            let open_edge_count = uses.values().filter(|uses| uses.len() == 1).count() as u64;
            let nonmanifold_edge_count = uses
                .values()
                .filter(|uses| uses.len() > 2 || (uses.len() == 2 && uses[0] == uses[1]))
                .count() as u64;
            let is_watertight = open_edge_count == 0 && nonmanifold_edge_count == 0;
            if !is_sheet_shell && !is_watertight {
                return Err(
                    invalid("solid shell is not watertight and manifold").with_shell(&shell.id)
                );
            }
            Ok(ExactSurfaceShellEvidence {
                source_shell_id: shell.id.clone(),
                face_count: shell.face_uses.len() as u64,
                shared_edge_count,
                open_edge_count,
                nonmanifold_edge_count,
                is_sheet_shell,
                is_watertight,
            })
        })
        .collect()
}

fn compose(
    shell: TopologicalOrientation,
    face: TopologicalOrientation,
    coedge: TopologicalOrientation,
) -> TopologicalOrientation {
    if [shell, face, coedge]
        .into_iter()
        .filter(|orientation| *orientation == TopologicalOrientation::Reversed)
        .count()
        % 2
        == 1
    {
        TopologicalOrientation::Reversed
    } else {
        TopologicalOrientation::Forward
    }
}

pub(super) fn validate_options(
    options: ExactSurfaceJoinOptions,
) -> Result<(), ExactSurfaceMeshError> {
    if !options.coordinate_tolerance_m.is_finite()
        || options.coordinate_tolerance_m <= 0.0
        || options.maximum_nodes == 0
        || options.maximum_triangles == 0
        || options.maximum_boundary_segments == 0
    {
        return Err(ExactSurfaceMeshError::new(
            ExactSurfaceMeshErrorKind::InvalidOptions,
            "surface join tolerance and hard limits must be finite and positive",
        ));
    }
    Ok(())
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>()
        .sqrt()
}

fn invalid(reason: impl Into<String>) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidInput, reason)
}

fn limit(reason: &str) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::ResourceLimit, reason)
}
