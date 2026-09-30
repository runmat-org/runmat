//! Active global aliases of one execution frame.
//!
//! The VM and native executor share this authority with their dynamic-workspace
//! adapter. A detached snapshot is used only when transferring a frame across
//! execution engines; it must not become a second mutable alias store.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameGlobalBinding {
    Named(String),
    LegacySlot,
}

#[derive(Clone, Debug, Default)]
pub struct FrameGlobalBindings(Rc<RefCell<HashMap<usize, FrameGlobalBinding>>>);

impl FrameGlobalBindings {
    pub fn from_snapshot(bindings: HashMap<usize, FrameGlobalBinding>) -> Self {
        Self(Rc::new(RefCell::new(bindings)))
    }

    pub fn snapshot(&self) -> HashMap<usize, FrameGlobalBinding> {
        self.0.borrow().clone()
    }

    pub fn declare_named(&self, slot: usize, name: String) {
        self.0
            .borrow_mut()
            .insert(slot, FrameGlobalBinding::Named(name));
    }

    pub fn declare_legacy_slot(&self, slot: usize) {
        self.0
            .borrow_mut()
            .insert(slot, FrameGlobalBinding::LegacySlot);
    }

    pub fn binding_for_slot(&self, slot: usize) -> Option<FrameGlobalBinding> {
        self.0.borrow().get(&slot).cloned()
    }

    pub fn name_for_slot(&self, slot: usize) -> Option<String> {
        match self.binding_for_slot(slot) {
            Some(FrameGlobalBinding::Named(name)) => Some(name),
            Some(FrameGlobalBinding::LegacySlot) | None => None,
        }
    }

    pub fn is_bound(&self, slot: usize, name: &str) -> bool {
        self.0
            .borrow()
            .get(&slot)
            .is_some_and(|bound| matches!(bound, FrameGlobalBinding::Named(bound) if bound == name))
    }

    pub fn unbind_slot(&self, slot: usize) -> Option<FrameGlobalBinding> {
        self.0.borrow_mut().remove(&slot)
    }

    pub fn unbind_name(&self, name: &str) {
        self.0.borrow_mut().retain(
            |_, binding| !matches!(binding, FrameGlobalBinding::Named(bound) if bound == name),
        );
    }

    pub fn unbind_all(&self) {
        self.0.borrow_mut().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::FrameGlobalBindings;

    #[test]
    fn clones_share_frame_authority_but_snapshots_are_detached() {
        let owner = FrameGlobalBindings::default();
        let adapter = owner.clone();
        owner.declare_named(2, "alpha".into());
        owner.declare_legacy_slot(3);
        assert!(adapter.is_bound(2, "alpha"));
        let snapshot = owner.snapshot();
        adapter.unbind_slot(2);
        assert_eq!(owner.name_for_slot(2), None);
        assert_eq!(
            snapshot.get(&2),
            Some(&super::FrameGlobalBinding::Named("alpha".into()))
        );
        assert_eq!(
            snapshot.get(&3),
            Some(&super::FrameGlobalBinding::LegacySlot)
        );
        let resumed = FrameGlobalBindings::from_snapshot(snapshot);
        resumed.declare_named(2, "beta".into());
        assert_eq!(resumed.name_for_slot(2).as_deref(), Some("beta"));
        assert_eq!(owner.name_for_slot(2), None);
        assert_eq!(
            resumed.binding_for_slot(3),
            Some(super::FrameGlobalBinding::LegacySlot)
        );
    }
}
