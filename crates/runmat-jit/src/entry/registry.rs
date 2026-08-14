use std::{collections::BTreeMap, rc::Rc};

use runmat_types::ProgramFunctionId;

use crate::{invalidation::DependencySnapshot, GenericExecutor};

use super::{EntryCell, EntryKey, PublishedEntry};

/// JIT-owned stable entry publication and retirement registry.
#[derive(Default)]
pub struct EntryRegistry {
    cells: BTreeMap<EntryKey, EntryCell>,
}

impl EntryRegistry {
    pub fn cell(&mut self, key: EntryKey) -> EntryCell {
        self.cells
            .entry(key.clone())
            .or_insert_with(|| EntryCell::new(key))
            .clone()
    }

    pub fn resolve(&self, key: &EntryKey, current: &DependencySnapshot) -> Option<PublishedEntry> {
        self.cells.get(key)?.resolve(current)
    }

    pub fn publish(
        &mut self,
        key: EntryKey,
        executor: Rc<GenericExecutor>,
        entrypoint: ProgramFunctionId,
        dependencies: DependencySnapshot,
    ) -> Result<PublishedEntry, &'static str> {
        self.cell(key).publish(executor, entrypoint, dependencies)
    }

    pub fn invalidate(&mut self, current: &DependencySnapshot) -> usize {
        self.cells
            .values()
            .filter(|cell| cell.invalidate_if_changed(current))
            .count()
    }

    pub fn retained_cell_count(&self) -> usize {
        self.cells.len()
    }

    pub fn published_entry_count(&self) -> usize {
        self.cells
            .values()
            .filter(|cell| cell.is_published())
            .count()
    }
}
