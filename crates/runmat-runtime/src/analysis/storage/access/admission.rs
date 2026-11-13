//! Lifetime reservations against one captured host storage-operation ceiling.
use super::*;
use std::cell::Cell;

pub(super) struct ScopeBudget {
    remaining: Cell<u64>,
}
impl ScopeBudget {
    pub(super) fn new(maximum: u64) -> Self {
        Self {
            remaining: Cell::new(maximum),
        }
    }
}

pub(in crate::analysis::storage) struct Reservation<'a> {
    budget: &'a ScopeBudget,
    bytes: u64,
}
impl Reservation<'_> {
    pub(in crate::analysis::storage) fn grow(
        &mut self,
        bytes: u64,
    ) -> Result<(), AnalysisPublicationError> {
        let remaining = self
            .budget
            .remaining
            .get()
            .checked_sub(bytes)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: format!(
                    "artifact operation requests {bytes} additional bytes with {} remaining",
                    self.budget.remaining.get()
                ),
            })?;
        let reserved = self.bytes.checked_add(bytes).ok_or_else(|| {
            AnalysisPublicationError::ResourceLimit {
                message: "artifact reservation overflow".into(),
            }
        })?;
        self.budget.remaining.set(remaining);
        self.bytes = reserved;
        Ok(())
    }
}
impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        // Successful reservations never exceed the captured ceiling, including
        // non-LIFO drops. Failed growth leaves both counters unchanged.
        self.budget
            .remaining
            .set(self.budget.remaining.get() + self.bytes);
    }
}
impl FilesystemAccess<'_> {
    pub(in crate::analysis::storage) fn remaining_bytes(&self) -> u64 {
        self.budget.remaining.get()
    }
    pub(in crate::analysis::storage) fn reserve_resident(
        &self,
        bytes: u64,
    ) -> Result<Reservation<'_>, AnalysisPublicationError> {
        let mut reservation = Reservation {
            budget: &self.budget,
            bytes: 0,
        };
        reservation.grow(bytes)?;
        Ok(reservation)
    }
    pub(in crate::analysis::storage) fn retained_estimate(
        &self,
        value: &impl Serialize,
    ) -> Result<u64, AnalysisPublicationError> {
        runmat_canonical_codec::profile_serialized(
            value,
            runmat_canonical_codec::CanonicalLimits::content_bounded(
                usize::try_from(self.remaining_bytes()).map_err(|_| {
                    AnalysisPublicationError::ResourceLimit {
                        message: "artifact budget exceeds address space".into(),
                    }
                })?,
                65_536,
                64,
            ),
        )
        .map(|profile| profile.materialization_resident_bytes_estimate)
        .map_err(Into::into)
    }
    /// A retained collection reserves its shared envelope once, then charges each
    /// additional entry through this same codec authority before growth.
    pub(in crate::analysis::storage) fn retained_increment_estimate(
        &self,
        value: &impl Serialize,
    ) -> Result<u64, AnalysisPublicationError> {
        runmat_canonical_codec::profile_serialized(
            value,
            runmat_canonical_codec::CanonicalLimits::content_bounded(
                usize::try_from(self.remaining_bytes()).map_err(|_| {
                    AnalysisPublicationError::ResourceLimit {
                        message: "artifact budget exceeds address space".into(),
                    }
                })?,
                65_536,
                64,
            ),
        )
        .map_err(AnalysisPublicationError::from)?
        .materialization_bytes_without_envelope()
        .ok_or_else(|| "retained inventory profile lacks its shared envelope".into())
    }
    pub(in crate::analysis::storage) fn retain_directory(
        &self,
        entries: &[DirEntry],
    ) -> Result<Reservation<'_>, AnalysisPublicationError> {
        let bytes = entries
            .iter()
            .try_fold(0u64, |total, entry| {
                total.checked_add(entry.resident_bytes()?)
            })
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "artifact directory residency overflow".into(),
            })?;
        self.reserve_resident(bytes)
    }
}
