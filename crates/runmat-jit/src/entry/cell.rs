use std::{cell::RefCell, rc::Rc};

use runmat_types::ProgramFunctionId;

use crate::{invalidation::DependencySnapshot, GenericExecutor};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EntryKey(pub String);

#[derive(Clone)]
pub struct PublishedEntry {
    pub executor: Rc<GenericExecutor>,
    pub entrypoint: ProgramFunctionId,
    pub dependencies: DependencySnapshot,
    pub publication: u64,
}

#[derive(Default)]
struct EntryState {
    publication: u64,
    target: Option<PublishedEntry>,
}

/// Stable indirection retained while compiled targets are replaced or retired.
#[derive(Clone)]
pub struct EntryCell {
    key: EntryKey,
    state: Rc<RefCell<EntryState>>,
}

impl EntryCell {
    pub(crate) fn new(key: EntryKey) -> Self {
        Self {
            key,
            state: Rc::new(RefCell::new(EntryState::default())),
        }
    }

    pub fn key(&self) -> &EntryKey {
        &self.key
    }

    pub fn resolve(&self, current: &DependencySnapshot) -> Option<PublishedEntry> {
        self.state
            .borrow()
            .target
            .as_ref()
            .filter(|target| target.dependencies.is_satisfied_by(current))
            .cloned()
    }

    pub(crate) fn publish(
        &self,
        executor: Rc<GenericExecutor>,
        entrypoint: ProgramFunctionId,
        dependencies: DependencySnapshot,
    ) -> Result<PublishedEntry, &'static str> {
        if dependencies.is_empty() {
            return Err("published native entry must declare dependencies");
        }
        let mut state = self.state.borrow_mut();
        state.publication = state
            .publication
            .checked_add(1)
            .ok_or("entry publication generation exhausted")?;
        let target = PublishedEntry {
            executor,
            entrypoint,
            dependencies,
            publication: state.publication,
        };
        state.target = Some(target.clone());
        Ok(target)
    }

    pub(crate) fn invalidate_if_changed(&self, current: &DependencySnapshot) -> bool {
        let mut state = self.state.borrow_mut();
        let invalid = state
            .target
            .as_ref()
            .is_some_and(|target| !target.dependencies.is_satisfied_by(current));
        if invalid {
            state.target = None;
        }
        invalid
    }

    pub fn is_published(&self) -> bool {
        self.state.borrow().target.is_some()
    }
}
