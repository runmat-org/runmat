//! Heap bytes the working quality holds, for the run's memory ceiling: the cell map with each
//! entry's metric vectors, both rankings, open trial journals and the comparison baseline.
use super::{ranking::Rank, CellKey, Entry, Ranking, WorkingVolumeQuality};
use crate::cdt::DelaunayTetrahedronQuality;
use runmat_meshing_core::resident_bytes::{arc_bytes, btree_bytes, vec_bytes};
use std::collections::BTreeMap;
use std::sync::Arc;

impl Entry {
    fn payload_bytes(&self) -> u64 {
        quality_payload_bytes(&self.quality)
    }
}

/// Heap bytes of one tetrahedron quality record's vectors.
pub(in crate::cdt) fn quality_payload_bytes(quality: &DelaunayTetrahedronQuality) -> u64 {
    vec_bytes::<runmat_geometry_core::PersistentEntityId>(
        quality.incident_metric_entity_ids.capacity(),
    )
    .saturating_add(vec_bytes::<runmat_meshing_size::metric::MetricSourceKind>(
        quality.active_metric_sources.capacity(),
    ))
}

impl Ranking {
    fn resident_bytes(&self) -> u64 {
        btree_bytes::<Rank, CellKey>(self.priorities.len())
            .saturating_add(btree_bytes::<u64, i64>(self.delta.len()))
    }
}

fn journal_bytes(journal: &BTreeMap<CellKey, Option<Arc<Entry>>>) -> u64 {
    journal.values().flatten().fold(
        btree_bytes::<CellKey, Option<Arc<Entry>>>(journal.len()),
        |sum, entry| {
            sum.saturating_add(arc_bytes::<Entry>())
                .saturating_add(entry.payload_bytes())
        },
    )
}

impl WorkingVolumeQuality {
    /// An upper bound on the heap bytes this quality owner holds now; the walk visits every cell
    /// once. Journaled entries are counted as owned even when a cell still shares them.
    pub(in crate::cdt) fn resident_bytes(&self) -> u64 {
        let cells = self.cells.values().fold(
            btree_bytes::<CellKey, Entry>(self.cells.len()),
            |sum, entry| sum.saturating_add(entry.payload_bytes()),
        );
        let journals = self.journals.iter().fold(
            vec_bytes::<BTreeMap<CellKey, Option<Arc<Entry>>>>(self.journals.capacity()),
            |sum, journal| sum.saturating_add(journal_bytes(journal)),
        );
        [
            cells,
            self.shape.resident_bytes(),
            self.refinement.resident_bytes(),
            journals,
            journal_bytes(&self.baseline_originals),
        ]
        .into_iter()
        .fold(0, u64::saturating_add)
    }
}
