//! Exact edge conformity and oriented shell evidence for canonical surface joins.
use super::{invalid, ExactSurfaceMeshError};
use crate::surface_mesh::control::checkpoint;
use crate::ExactFaceMeshBoundarySegment;
use crate::ExactSurfaceShellEvidence;
use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId, TopologicalOrientation};
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
struct CanonicalEdgePiece {
    edge_parameters: [f64; 2],
    node_ids: [StableDigest; 2],
}

pub(in crate::surface_mesh) fn validate_edge_conformity_with_control(
    topology: &ExactBRepTopology,
    segments: &[ExactFaceMeshBoundarySegment],
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    checkpoint(control)?;
    let coedges = topology
        .coedges
        .iter()
        .map(|coedge| (&coedge.id, coedge))
        .collect::<BTreeMap<_, _>>();
    let mut coverage =
        BTreeMap::<PersistentEntityId, BTreeMap<PersistentEntityId, Vec<CanonicalEdgePiece>>>::new(
        );
    for segment in segments {
        checkpoint(control)?;
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
            checkpoint(control)?;
            pieces.sort_by(|left, right| {
                left.edge_parameters[0]
                    .total_cmp(&right.edge_parameters[0])
                    .then_with(|| left.edge_parameters[1].total_cmp(&right.edge_parameters[1]))
                    .then_with(|| left.node_ids.cmp(&right.node_ids))
            });
        }
    }
    for edge in &topology.edges {
        checkpoint(control)?;
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

pub(in crate::surface_mesh) fn shell_evidence_with_control(
    topology: &ExactBRepTopology,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<Vec<ExactSurfaceShellEvidence>, ExactSurfaceMeshError> {
    checkpoint(control)?;
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
            checkpoint(control)?;
            let is_sheet_shell = sheet_shells.contains(&shell.id);
            if is_sheet_shell == solid_shells.contains(&shell.id) {
                return Err(
                    invalid("shell ownership is neither uniquely solid nor sheet")
                        .with_shell(&shell.id),
                );
            }
            let mut uses = BTreeMap::<PersistentEntityId, Vec<TopologicalOrientation>>::new();
            for face_use in &shell.face_uses {
                checkpoint(control)?;
                let face = faces.get(&face_use.entity_id).ok_or_else(|| {
                    invalid("shell references an absent face").with_shell(&shell.id)
                })?;
                for wire_id in std::iter::once(&face.outer_wire_id).chain(&face.inner_wire_ids) {
                    let wire = wires.get(wire_id).ok_or_else(|| {
                        invalid("face references an absent wire").with_shell(&shell.id)
                    })?;
                    for coedge_id in &wire.coedge_ids {
                        checkpoint(control)?;
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
