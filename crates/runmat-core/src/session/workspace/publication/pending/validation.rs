//! Physical table growth and pre-effect slot validation.

use runmat_runtime::context::host_allocation::HostAllocationError;

use super::PendingBindingParticipant;
use crate::session::workspace::publication::patch::{PendingBindingPatch, PreparedPatchUpdate};

impl PendingBindingParticipant {
    pub(in crate::session::workspace::publication) fn table_growth_for_action(
        &self,
        name: &str,
        introduces_binding: bool,
    ) -> Result<u64, HostAllocationError> {
        let insertions = self
            .patch()
            .patch_insertions_after(name, introduces_binding)?;
        self.lease
            .try_borrow()
            .map_err(|_| HostAllocationError::StaleAuthority)?
            .growth_retained_bytes(insertions)
    }

    pub(in crate::session::workspace::publication) fn table_growth_retained_bytes_for_insertions(
        &self,
        insertions: usize,
    ) -> Result<u64, HostAllocationError> {
        self.lease
            .try_borrow()
            .map_err(|_| HostAllocationError::StaleAuthority)?
            .growth_retained_bytes(insertions)
    }

    pub(in crate::session::workspace::publication) fn validate_update_slots(
        &self,
        update: &PreparedPatchUpdate,
    ) -> Result<(), HostAllocationError> {
        let table = self
            .lease
            .try_borrow()
            .map_err(|_| HostAllocationError::StaleAuthority)?;
        self.patch().validate_slots_after(update, &table)
    }

    pub(in crate::session::workspace::publication) fn validate_replacement_slots(
        &self,
        replacement: &PendingBindingPatch,
    ) -> Result<(), HostAllocationError> {
        let table = self
            .lease
            .try_borrow()
            .map_err(|_| HostAllocationError::StaleAuthority)?;
        replacement.validate_for_table(&table)
    }
}
