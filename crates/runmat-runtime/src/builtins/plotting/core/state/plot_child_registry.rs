//! One owning registry for compatibility child state and exact object leases.

use super::PlotChildHandleState;
use runmat_value::GraphicsReceiverRef;
use std::collections::HashMap;
use std::sync::Arc;

/// The state and optional exact receiver have one lifecycle. A numeric key
/// alone cannot authenticate either the chart or a callback Target.
#[derive(Debug)]
pub(super) struct PlotChildEntry {
    pub(super) state: Arc<PlotChildHandleState>,
    pub(super) receiver: Option<GraphicsReceiverRef>,
}

impl PlotChildEntry {
    fn new(state: PlotChildHandleState) -> Self {
        Self {
            state: Arc::new(state),
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
        self.entries.get(id).map(|entry| entry.state.as_ref())
    }

    pub(super) fn get_mut(&mut self, id: &u64) -> Option<&mut PlotChildHandleState> {
        self.entries
            .get_mut(id)
            .map(|entry| Arc::make_mut(&mut entry.state))
    }

    /// Publish a prepared complete state without copying the old payload only
    /// to overwrite it. The exact receiver stays on the live entry.
    pub(super) fn replace_if(
        &mut self,
        id: u64,
        predicate: impl FnOnce(&PlotChildHandleState) -> bool,
        replacement: PlotChildHandleState,
    ) -> bool {
        let Some(entry) = self
            .entries
            .get_mut(&id)
            .filter(|entry| predicate(entry.state.as_ref()))
        else {
            return false;
        };
        let previous = std::mem::replace(&mut entry.state, Arc::new(replacement));
        // Park only the displaced payload; no receiver is retired or minted.
        self.retired.push(PlotChildEntry {
            state: previous,
            receiver: None,
        });
        true
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &PlotChildHandleState> {
        self.entries.values().map(|entry| entry.state.as_ref())
    }

    /// Inspect borrowed state before making only the selected entry writable.
    pub(super) fn find_mut(
        &mut self,
        mut predicate: impl FnMut(&PlotChildHandleState) -> bool,
    ) -> Option<&mut PlotChildHandleState> {
        self.entries
            .values_mut()
            .find(|entry| predicate(entry.state.as_ref()))
            .map(|entry| Arc::make_mut(&mut entry.state))
    }

    pub(super) fn for_each_mut_where(
        &mut self,
        mut predicate: impl FnMut(&PlotChildHandleState) -> bool,
        mut update: impl FnMut(&mut PlotChildHandleState),
    ) {
        for entry in self.entries.values_mut() {
            if predicate(entry.state.as_ref()) {
                update(Arc::make_mut(&mut entry.state));
            }
        }
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&u64, &PlotChildHandleState)> {
        self.entries
            .iter()
            .map(|(id, entry)| (id, entry.state.as_ref()))
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&u64, &PlotChildHandleState) -> bool) {
        self.retired.extend(
            self.entries
                .extract_if(|id, entry| !keep(id, entry.state.as_ref()))
                .map(|(_, entry)| entry),
        );
    }

    /// Only selected entries may copy retained state. Other charts are inspected
    /// by reference, including when an immutable property snapshot is pending.
    pub(super) fn retain_mut_where(
        &mut self,
        mut predicate: impl FnMut(&PlotChildHandleState) -> bool,
        mut keep: impl FnMut(&u64, &mut PlotChildHandleState) -> bool,
    ) {
        self.retired.extend(
            self.entries
                .extract_if(|id, entry| {
                    predicate(entry.state.as_ref()) && !keep(id, Arc::make_mut(&mut entry.state))
                })
                .map(|(_, entry)| entry),
        );
    }

    #[cfg(test)]
    pub(super) fn clear(&mut self) {
        self.retired
            .extend(self.entries.drain().map(|(_, entry)| entry));
    }
}
