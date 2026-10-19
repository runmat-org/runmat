//! Visibility edits coexist with inventories and provider work under one scope allowance.
use super::*;

pub(super) struct Edits<'a, 'store> {
    store: &'a access::FilesystemAccess<'store>,
    values: Vec<WorkspaceEdit>,
    reservation: access::Reservation<'a>,
}
impl<'a, 'store> Edits<'a, 'store> {
    pub(super) fn new(
        store: &'a access::FilesystemAccess<'store>,
    ) -> Result<Self, AnalysisPublicationError> {
        Ok(Self {
            store,
            values: Vec::new(),
            reservation: store
                .reserve_resident(store.retained_estimate(&Vec::<WorkspaceEdit>::new())?)?,
        })
    }
    pub(super) fn push(&mut self, edit: WorkspaceEdit) -> Result<(), AnalysisPublicationError> {
        // Marker bytes are produced by admitted preparation; path temporaries fit
        // the shared envelope. Charge the retained edit before growing its vector.
        self.reservation
            .grow(self.store.retained_increment_estimate(&[&edit])?)?;
        self.values.push(edit);
        Ok(())
    }
    pub(super) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub(super) fn into_parts(self) -> (Vec<WorkspaceEdit>, access::Reservation<'a>) {
        (self.values, self.reservation)
    }
}
