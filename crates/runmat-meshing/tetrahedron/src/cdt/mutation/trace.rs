//! Capture only checked local edits along the active trial branch. Nested
//! rollback truncates rejected steps. These records are data, not transferable
//! admission: retained search seeds must replay through current checked APIs.
use super::{storage::MutableTetrahedron, TopologyMutation};
use crate::cdt::DelaunayVolumeNode;
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;
use std::collections::BTreeSet;
mod history;
pub(in crate::cdt) use history::TopologyEdits;

#[derive(Clone, Debug, PartialEq)]
pub(in crate::cdt) enum TopologyEdit {
    AppendNode(DelaunayVolumeNode),
    ReplaceNode {
        source_identity: StableDigest,
        replacement: DelaunayVolumeNode,
        affected: Vec<[StableDigest; 4]>,
    },
    RelocateCavity {
        replacements: Vec<(StableDigest, DelaunayVolumeNode)>,
        removed: Vec<[StableDigest; 4]>,
        created: Vec<([StableDigest; 4], Option<PersistentEntityId>)>,
    },
    ReplaceCavity {
        removed: Vec<[StableDigest; 4]>,
        created: Vec<([StableDigest; 4], Option<PersistentEntityId>)>,
    },
}

impl TopologyMutation {
    pub(in crate::cdt) fn enable_edit_trace(&mut self) {
        assert!(
            self.trial_journals.is_empty(),
            "trace policy cannot change inside a trial"
        );
        self.trace_enabled = true;
    }
    pub(in crate::cdt) fn clear_edit_trace(&mut self) {
        assert!(
            self.trial_journals.is_empty(),
            "cannot rebase a speculative edit trace"
        );
        self.edit_trace.clear();
    }
    pub(in crate::cdt) fn active_edits(&self) -> &TopologyEdits {
        assert!(
            self.trace_enabled && !self.trial_journals.is_empty(),
            "capture requires an active traced trial"
        );
        &self.edit_trace
    }
    pub(in crate::cdt) fn capture_edits(&self) -> TopologyEdits {
        let edits = self.active_edits();
        report_retained_history(edits);
        edits.clone()
    }
    pub(in crate::cdt::mutation) fn record_cavity_trace(
        &mut self,
        removed: &BTreeSet<u32>,
        created: &[(u32, MutableTetrahedron)],
    ) {
        if !self.trace_enabled || self.trial_journals.is_empty() {
            return;
        }
        let identities =
            |vertices: [u32; 4]| vertices.map(|vertex| self.nodes[vertex as usize].identity);
        let removed = removed
            .iter()
            .map(|index| {
                identities(
                    self.tetrahedron(*index)
                        .expect("prepared removed cell")
                        .vertex_indices,
                )
            })
            .collect();
        let created = created
            .iter()
            .map(|(_, cell)| (identities(cell.vertex_indices), cell.region_id.clone()))
            .collect();
        self.edit_trace
            .push(TopologyEdit::ReplaceCavity { removed, created });
    }
}

// At most one observation per doubling in a process; no per-edit event inventory.
fn report_retained_history(edits: &TopologyEdits) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT_REPORT: AtomicUsize = AtomicUsize::new(512);
    if !tracing::enabled!(tracing::Level::INFO) {
        return;
    }
    let threshold = NEXT_REPORT.load(Ordering::Relaxed);
    if edits.len() < threshold {
        return;
    }
    let next = edits.len().saturating_mul(2);
    if NEXT_REPORT
        .compare_exchange(threshold, next, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        tracing::info!(
            retained_edit_count = edits.len(),
            shared_edit_blocks = edits.block_count(),
            "meshing retained chronological edit history"
        );
    }
}
