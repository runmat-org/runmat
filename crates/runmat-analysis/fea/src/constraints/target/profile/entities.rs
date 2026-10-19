//! Borrow original candidate provenance for codec profiling; never clone a selection.
use super::*;
use serde::{ser::SerializeSeq, Serialize};

pub(super) struct EntityInventory<'a> {
    pub(super) mesh: &'a SolverMeshArtifact,
    pub(super) target: &'a RegionTarget,
    pub(super) selector: &'a Selector,
    pub(super) count: usize,
}
impl Serialize for EntityInventory<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.count))?;
        let mut emitted = 0usize;
        match self.selector {
            Selector::Face => {
                for face in &self.mesh.topology.boundary_faces {
                    if crate::is_fea_cancelled() {
                        return Err(serde::ser::Error::custom(
                            "constraint selection profiling cancelled",
                        ));
                    }
                    for entity in face
                        .provenance
                        .iter()
                        .filter(|e| self.target.matches_persistent_entity(e))
                    {
                        sequence.serialize_element(entity)?;
                        emitted = emitted.checked_add(1).ok_or_else(|| {
                            serde::ser::Error::custom("selection inventory overflow")
                        })?;
                    }
                }
            }
            Selector::ExactPoint => {
                for entity in self.target.resolved_entities() {
                    if crate::is_fea_cancelled() {
                        return Err(serde::ser::Error::custom(
                            "constraint selection profiling cancelled",
                        ));
                    }
                    sequence.serialize_element(entity)?;
                    emitted = emitted
                        .checked_add(1)
                        .ok_or_else(|| serde::ser::Error::custom("selection inventory overflow"))?;
                }
            }
            Selector::Point(_) => {}
        }
        if emitted != self.count {
            return Err(serde::ser::Error::custom(
                "selection inventory count changed",
            ));
        }
        sequence.end()
    }
}
