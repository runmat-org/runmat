//! Fallible index preparation stays inside rollback. Committing and advancing
//! the accepted comparison revision then require no cancellable operation.
use super::*;

impl WorkingTrial<'_, '_> {
    pub(in crate::cdt) fn commit_prepared(
        self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<CommittedVolumeChange, VolumeStateError> {
        let mut poll = || {
            if cancellation.is_cancelled() {
                Err(error(
                    VolumeStateErrorKind::Cancelled,
                    "working commit preparation cancelled",
                ))
            } else {
                Ok(())
            }
        };
        self.topology.prepare_point_spatial_index(&mut poll)?;
        self.topology.prepare_cell_spatial_index(&mut poll)?;
        poll()?;
        let change = CommittedVolumeChange {
            binding: self.binding.clone(),
            revision: self.revision,
        };
        self.commit();
        Ok(change)
    }
}

impl WorkingState<'_> {
    pub(in crate::cdt) fn rebase_committed(&mut self, change: CommittedVolumeChange) {
        assert!(
            Arc::ptr_eq(&self.binding, &change.binding) && self.revision == change.revision,
            "committed change must belong to the same accepted comparison revision"
        );
        self.finish_rebase();
    }
    pub(super) fn finish_rebase(&mut self) {
        self.topology.clear_edit_trace();
        self.quality.reset_comparison();
        self.revision = self
            .revision
            .checked_add(1)
            .expect("bounded search revision");
    }
}
