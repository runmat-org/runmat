use std::collections::BTreeMap;

use super::{ExactBRepTopology, GeometryContractError, PersistentEntityId, TopologicalOrientation};

impl ExactBRepTopology {
    /// Independently proves that every nondegenerate edge of every solid shell has exactly two
    /// oppositely oriented boundary uses. Sheet bodies are intentionally excluded because their
    /// boundary may be open.
    pub fn validate_solid_shell_boundaries(&self) -> Result<(), GeometryContractError> {
        let shells = self
            .shells
            .iter()
            .map(|shell| (&shell.id, shell))
            .collect::<BTreeMap<_, _>>();
        let faces = self
            .faces
            .iter()
            .map(|face| (&face.id, face))
            .collect::<BTreeMap<_, _>>();
        let wires = self
            .wires
            .iter()
            .map(|wire| (&wire.id, wire))
            .collect::<BTreeMap<_, _>>();
        let coedges = self
            .coedges
            .iter()
            .map(|coedge| (&coedge.id, coedge))
            .collect::<BTreeMap<_, _>>();
        let edges = self
            .edges
            .iter()
            .map(|edge| (&edge.id, edge))
            .collect::<BTreeMap<_, _>>();

        for solid in &self.solids {
            for shell_id in std::iter::once(&solid.outer_shell_id).chain(&solid.void_shell_ids) {
                let shell = shells
                    .get(shell_id)
                    .ok_or_else(|| incomplete("solid shell"))?;
                let mut edge_uses =
                    BTreeMap::<&PersistentEntityId, Vec<TopologicalOrientation>>::new();
                for face_use in &shell.face_uses {
                    let face = faces
                        .get(&face_use.entity_id)
                        .ok_or_else(|| incomplete("shell face"))?;
                    for wire_id in std::iter::once(&face.outer_wire_id).chain(&face.inner_wire_ids)
                    {
                        let wire = wires.get(wire_id).ok_or_else(|| incomplete("face wire"))?;
                        for coedge_id in &wire.coedge_ids {
                            let coedge = coedges
                                .get(coedge_id)
                                .ok_or_else(|| incomplete("wire coedge"))?;
                            let edge = edges
                                .get(&coedge.edge_id)
                                .ok_or_else(|| incomplete("coedge edge"))?;
                            if edge.is_degenerate {
                                continue;
                            }
                            edge_uses.entry(&edge.id).or_default().push(compose(
                                shell.orientation,
                                face_use.orientation,
                                coedge.orientation,
                            ));
                        }
                    }
                }
                for uses in edge_uses.values() {
                    if uses.len() != 2 || uses[0] == uses[1] {
                        return Err(GeometryContractError::invalid(
                            "solid shell boundary",
                            "each nondegenerate edge requires exactly two oppositely oriented uses within its shell",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

fn compose(
    shell: TopologicalOrientation,
    face: TopologicalOrientation,
    coedge: TopologicalOrientation,
) -> TopologicalOrientation {
    let reversed = [shell, face, coedge]
        .into_iter()
        .filter(|orientation| *orientation == TopologicalOrientation::Reversed)
        .count()
        % 2
        == 1;
    if reversed {
        TopologicalOrientation::Reversed
    } else {
        TopologicalOrientation::Forward
    }
}

fn incomplete(field: &str) -> GeometryContractError {
    GeometryContractError::invalid(field, "topology reference index is incomplete")
}
