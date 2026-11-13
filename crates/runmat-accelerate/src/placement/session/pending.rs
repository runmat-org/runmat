//! Bounded owner-thread delivery. Device callbacks mutate receipts, never this
//! RefCell/session. Source polling happens outside every mutable borrow.
use runmat_execution::PlacementInvalidation;
use runmat_runtime::context::{RuntimePlacementFeedbackDelivery, RuntimePlacementFeedbackSource};
use std::cell::{Cell, RefCell};

const MAX_PENDING: usize = 64;

#[derive(Default)]
pub(super) struct PendingFeedback {
    sources: RefCell<Vec<Box<dyn RuntimePlacementFeedbackSource>>>,
    draining: Cell<bool>,
    discarded: Cell<u64>,
    epoch: Cell<u64>,
}

impl PendingFeedback {
    pub(super) fn has_sources(&self) -> bool {
        self.draining.get() || !self.sources.borrow().is_empty()
    }

    pub(super) fn push(
        &self,
        source: Box<dyn RuntimePlacementFeedbackSource>,
    ) -> Result<(), Box<dyn RuntimePlacementFeedbackSource>> {
        if self.draining.get() || self.sources.borrow().len() >= MAX_PENDING {
            self.discarded.set(self.discarded.get().saturating_add(1));
            log::debug!("placement observation discarded: pending delivery capacity unavailable");
            return Err(source);
        }
        self.sources.borrow_mut().push(source);
        Ok(())
    }

    pub(super) fn take_ready(&self) -> Vec<RuntimePlacementFeedbackDelivery> {
        if self.draining.replace(true) {
            return Vec::new();
        }
        struct Reset<'a>(&'a Cell<bool>);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(false);
            }
        }
        let _reset = Reset(&self.draining);
        let epoch = self.epoch.get();
        let sources = std::mem::take(&mut *self.sources.borrow_mut());
        let mut pending = Vec::new();
        let mut ready = Vec::new();
        for mut source in sources {
            match source.take_ready() {
                Some(feedback) => ready.push(feedback),
                None if source.terminal_without_feedback() => {}
                None => pending.push(source),
            }
        }
        if self.epoch.get() != epoch {
            self.discarded.set(
                self.discarded
                    .get()
                    .saturating_add((pending.len() + ready.len()) as u64),
            );
            return Vec::new();
        }
        *self.sources.borrow_mut() = pending;
        ready
    }

    pub(super) fn invalidate(&self, invalidation: &PlacementInvalidation) {
        self.epoch.set(self.epoch.get().wrapping_add(1));
        let sources = std::mem::take(&mut *self.sources.borrow_mut());
        let mut retained = Vec::new();
        for source in sources {
            let signature = source.signature();
            let invalid = invalidation.matches_signature(signature);
            if invalid {
                self.discarded.set(self.discarded.get().saturating_add(1));
            } else {
                retained.push(source);
            }
        }
        *self.sources.borrow_mut() = retained;
    }
}

impl Drop for PendingFeedback {
    fn drop(&mut self) {
        let outstanding = self.sources.get_mut().len();
        if outstanding != 0 || self.discarded.get() != 0 {
            log::debug!("placement observation delivery ended: {} still pending, {} discarded; no synchronization forced", outstanding, self.discarded.get());
        }
    }
}

#[cfg(test)]
mod tests;
