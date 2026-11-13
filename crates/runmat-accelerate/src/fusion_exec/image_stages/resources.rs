//! Checked upper bound for the exact selected hook sequence. This is a sum of
//! provider-requested incremental peaks, so it safely covers overlapping
//! scratch and published handles without pretending to know device capacity.
//! It is not a cost or executable placement receipt by itself.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ImageStagePhysicalUpperBound {
    pub requested_bytes: u64,
    pub dispatches: u32,
    /// None means at least one actual selected hook has no owner-issued
    /// submission geometry. Dispatch count is not a substitute for it.
    pub queue_submissions: Option<u32>,
}

impl Default for ImageStagePhysicalUpperBound {
    fn default() -> Self {
        Self {
            requested_bytes: 0,
            dispatches: 0,
            queue_submissions: Some(0),
        }
    }
}

impl ImageStagePhysicalUpperBound {
    pub fn add(
        &mut self,
        requested_bytes: u64,
        dispatches: u32,
        queue_submissions: Option<u32>,
    ) -> Option<()> {
        self.requested_bytes = self.requested_bytes.checked_add(requested_bytes)?;
        self.dispatches = self.dispatches.checked_add(dispatches)?;
        self.queue_submissions = self
            .queue_submissions
            .zip(queue_submissions)
            .and_then(|(left, right)| left.checked_add(right));
        Some(())
    }

    pub fn combine(&mut self, other: Self) -> Option<()> {
        self.add(
            other.requested_bytes,
            other.dispatches,
            other.queue_submissions,
        )
    }
}
