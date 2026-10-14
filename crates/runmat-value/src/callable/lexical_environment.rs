use runmat_gc_api::GcHandle;
use runmat_types::BindingId;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

/// Runtime-only, per-activation storage for shared nested bindings. The GC
/// handle points to a private cell row; the map gives semantic bindings their
/// slots without exposing either as a language value.
#[derive(Debug, Clone)]
pub struct LexicalEnvironment {
    node: Rc<LexicalEnvironmentNode>,
}

#[derive(Debug)]
struct LexicalEnvironmentNode {
    handle: GcHandle,
    slots: RefCell<BTreeMap<BindingId, LexicalSlot>>,
    parent: Option<LexicalEnvironment>,
}

#[derive(Debug, Clone, Copy)]
enum LexicalSlot {
    Uninitialized,
    Assigned(usize),
    Vacant(usize),
}

impl LexicalEnvironment {
    pub fn new(handle: GcHandle, parent: Option<LexicalEnvironment>) -> Self {
        Self {
            node: Rc::new(LexicalEnvironmentNode {
                handle,
                slots: RefCell::new(BTreeMap::new()),
                parent,
            }),
        }
    }

    pub fn handle(&self) -> GcHandle {
        self.node.handle
    }

    /// Host-local graph traversal only. The returned activation is never a
    /// portable identity; the admitted function layout supplies that proof.
    pub fn parent(&self) -> Option<Self> {
        self.node.parent.clone()
    }

    /// Local reservations, including uninitialized and retired slots. An
    /// imported layout must account for every reservation before snapshot.
    pub fn owned_binding_ids(&self) -> Vec<BindingId> {
        self.node.slots.borrow().keys().copied().collect()
    }

    pub fn resolve_slot(&self, binding: BindingId) -> Option<(GcHandle, usize)> {
        let mut current = Some(self);
        while let Some(environment) = current {
            if let Some(slot) = environment.node.slots.borrow().get(&binding).copied() {
                return match slot {
                    LexicalSlot::Assigned(slot) => Some((environment.node.handle, slot)),
                    LexicalSlot::Uninitialized | LexicalSlot::Vacant(_) => None,
                };
            }
            current = environment.node.parent.as_ref();
        }
        None
    }

    pub fn register_slot(&self, binding: BindingId, slot: usize) -> bool {
        let mut slots = self.node.slots.borrow_mut();
        if slots
            .get(&binding)
            .is_some_and(|slot| matches!(slot, LexicalSlot::Assigned(_)))
        {
            return false;
        }
        slots.insert(binding, LexicalSlot::Assigned(slot));
        true
    }

    pub fn local_slot(&self, binding: BindingId) -> Option<usize> {
        match self.node.slots.borrow().get(&binding).copied() {
            Some(LexicalSlot::Assigned(slot)) => Some(slot),
            _ => None,
        }
    }

    pub fn reusable_local_slot(&self, binding: BindingId) -> Option<usize> {
        match self.node.slots.borrow().get(&binding).copied() {
            Some(LexicalSlot::Vacant(slot)) => Some(slot),
            _ => None,
        }
    }

    pub fn reserve_owned_binding(&self, binding: BindingId) -> bool {
        let mut slots = self.node.slots.borrow_mut();
        if slots.contains_key(&binding) {
            return false;
        }
        slots.insert(binding, LexicalSlot::Uninitialized);
        true
    }

    pub fn owns_binding(&self, binding: BindingId) -> bool {
        self.node.slots.borrow().contains_key(&binding)
    }

    pub fn unregister_local_slot(&self, binding: BindingId) -> bool {
        let mut slots = self.node.slots.borrow_mut();
        match slots.get_mut(&binding) {
            Some(slot) => match *slot {
                LexicalSlot::Assigned(index) => {
                    *slot = LexicalSlot::Vacant(index);
                    true
                }
                LexicalSlot::Uninitialized | LexicalSlot::Vacant(_) => false,
            },
            None => false,
        }
    }

    pub fn unregister_slot(&self, binding: BindingId) -> bool {
        let mut current = Some(self);
        while let Some(environment) = current {
            if let Some(slot) = environment.node.slots.borrow_mut().get_mut(&binding) {
                if let LexicalSlot::Assigned(index) = *slot {
                    *slot = LexicalSlot::Vacant(index);
                    return true;
                }
                return false;
            }
            current = environment.node.parent.as_ref();
        }
        false
    }

    pub fn owner_for_binding(&self, binding: BindingId) -> Option<Self> {
        let mut current = Some(self);
        while let Some(environment) = current {
            if environment.node.slots.borrow().contains_key(&binding) {
                return Some(environment.clone());
            }
            current = environment.node.parent.as_ref();
        }
        None
    }

    pub fn for_each_handle(&self, mut visit: impl FnMut(GcHandle)) {
        let mut current = Some(self);
        while let Some(environment) = current {
            visit(environment.node.handle);
            current = environment.node.parent.as_ref();
        }
    }
}

impl PartialEq for LexicalEnvironment {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.node, &other.node)
    }
}
