//! One owning registry for compatibility child state and exact object leases.

use super::PlotChildHandleState;
use runmat_value::GraphicsReceiverRef;
use std::collections::HashMap;

/// The state and optional exact receiver have one lifecycle. A numeric key
/// alone cannot authenticate either the chart or a callback Target.
#[derive(Debug)]
pub(super) struct PlotChildEntry {
    pub(super) state: PlotChildHandleState,
    pub(super) receiver: Option<GraphicsReceiverRef>,
}

impl PlotChildEntry {
    fn new(state: PlotChildHandleState) -> Self {
        Self {
            state,
            receiver: None,
        }
    }
}

/// Removed entries are parked until the outer plot registry guard has retired.
#[derive(Default)]
pub(super) struct PlotChildRegistry {
    pub(super) entries: HashMap<u64, PlotChildEntry>,
    pub(super) retired: Vec<PlotChildEntry>,
}

impl PlotChildRegistry {
    pub(super) fn insert(&mut self, id: u64, state: PlotChildHandleState) {
        if let Some(previous) = self.entries.insert(id, PlotChildEntry::new(state)) {
            self.retired.push(previous);
        }
    }

    pub(super) fn get(&self, id: &u64) -> Option<&PlotChildHandleState> {
        self.entries.get(id).map(|entry| &entry.state)
    }

    pub(super) fn get_mut(&mut self, id: &u64) -> Option<&mut PlotChildHandleState> {
        self.entries.get_mut(id).map(|entry| &mut entry.state)
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &PlotChildHandleState> {
        self.entries.values().map(|entry| &entry.state)
    }

    pub(super) fn values_mut(&mut self) -> impl Iterator<Item = &mut PlotChildHandleState> {
        self.entries.values_mut().map(|entry| &mut entry.state)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&u64, &PlotChildHandleState)> {
        self.entries.iter().map(|(id, entry)| (id, &entry.state))
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&u64, &mut PlotChildHandleState) -> bool) {
        self.retired.extend(
            self.entries
                .extract_if(|id, entry| !keep(id, &mut entry.state))
                .map(|(_, entry)| entry),
        );
    }

    #[cfg(test)]
    pub(super) fn clear(&mut self) {
        self.retired
            .extend(self.entries.drain().map(|(_, entry)| entry));
    }
}
