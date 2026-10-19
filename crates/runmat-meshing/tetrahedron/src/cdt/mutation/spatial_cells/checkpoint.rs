//! A trial shares the persistent root and records only pending keys it changes.
use super::*;
#[derive(Clone, Debug)]
pub(in crate::cdt::mutation) struct CellSpatialCheckpoint {
    root: BoundsSpatialIndex<CellKey>,
    depth: usize,
}
impl CellSpatialCache {
    pub(in crate::cdt::mutation) fn checkpoint(&mut self) -> CellSpatialCheckpoint {
        CellSpatialCheckpoint {
            root: self.index.clone(),
            depth: self.pending.begin(),
        }
    }
    pub(in crate::cdt::mutation) fn commit_checkpoint(
        &mut self,
        checkpoint: CellSpatialCheckpoint,
    ) {
        self.pending.commit(checkpoint.depth);
    }
    pub(in crate::cdt::mutation) fn rollback_checkpoint(
        &mut self,
        checkpoint: CellSpatialCheckpoint,
    ) {
        self.pending.rollback(checkpoint.depth);
        self.index = checkpoint.root;
    }
}
