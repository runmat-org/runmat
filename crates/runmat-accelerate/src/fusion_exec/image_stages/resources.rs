//! Checked upper bound for the exact selected hook sequence. This is a sum of
//! provider-requested incremental peaks, so it safely covers overlapping
//! scratch and published handles without pretending to know device capacity.
//! It is not a cost or executable placement receipt by itself.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ImageStagePhysicalUpperBound {
    pub requested_bytes: u64,
    pub dispatches: u32,
}

impl ImageStagePhysicalUpperBound {
    pub fn add(&mut self, requested_bytes: u64, dispatches: u32) -> Option<()> {
        self.requested_bytes = self.requested_bytes.checked_add(requested_bytes)?;
        self.dispatches = self.dispatches.checked_add(dispatches)?;
        Some(())
    }

    pub fn combine(&mut self, other: Self) -> Option<()> {
        self.add(other.requested_bytes, other.dispatches)
    }
}
