use std::{collections::BTreeMap, rc::Rc};

use runmat_execution::Digest;
use runmat_types::ProgramFunctionId;

use crate::{invalidation::DependencySnapshot, tiering::TierAvailability, GenericExecutor};

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

    pub fn resolve_specialized(
        &self,
        key: &EntryKey,
        profile: Digest,
        current: &DependencySnapshot,
    ) -> Option<PublishedEntry> {
        self.cells.get(key)?.resolve_specialized(profile, current)
    }

    pub fn publish_specialized(
        &mut self,
        key: EntryKey,
        profile: Digest,
        executor: Rc<GenericExecutor>,
        entrypoint: ProgramFunctionId,
        dependencies: DependencySnapshot,
    ) -> Result<PublishedEntry, &'static str> {
        self.cell(key)
            .publish_specialized(profile, executor, entrypoint, dependencies)
    }

    pub fn availability(
        &self,
        key: &EntryKey,
        current: &DependencySnapshot,
        pending_compilations: usize,
    ) -> TierAvailability {
        let Some(cell) = self.cells.get(key) else {
            return TierAvailability {
                pending_compilations,
                ..TierAvailability::default()
            };
        };
        TierAvailability {
            generic_ready: cell.resolve(current).is_some(),
            specialized_profiles: cell.specialized_profiles(current),
            pending_compilations,
            retained_versions: cell.retained_versions(current),
            retained_code_bytes: self.retained_code_bytes(current),
        }
    }

    /// Retire the oldest specialized versions until both session limits hold.
    /// Generic continuations are never evicted by this operation.
    pub fn enforce_limits(
        &mut self,
        current: &DependencySnapshot,
        max_versions_per_entry: usize,
        max_code_bytes: u64,
    ) -> Result<usize, &'static str> {
        if max_versions_per_entry == 0 || max_code_bytes == 0 {
            return Err("native entry limits must be non-zero");
        }
        let mut retired = 0;
        for cell in self.cells.values() {
            while cell.retained_versions(current) > max_versions_per_entry {
                if cell.retire_oldest_specialized().is_none() {
                    break;
                }
                retired += 1;
            }
        }
        while self.retained_code_bytes(current) > max_code_bytes {
            let oldest = self
                .cells
                .iter()
                .filter_map(|(key, cell)| {
                    cell.oldest_specialized_publication()
                        .map(|publication| (publication, key.clone()))
                })
                .min();
            let Some((_, key)) = oldest else {
                break;
            };
            if self.cells[&key].retire_oldest_specialized().is_some() {
                retired += 1;
            }
        }
        if self.retained_code_bytes(current) > max_code_bytes {
            Err("generic native code exceeds the session code-memory budget")
        } else {
            Ok(retired)
        }
    }

    /// Retire replaceable specialized products to admit one completed
    /// replacement. The generic continuation is never retired.
    pub fn make_room_for_specialized(
        &mut self,
        key: &EntryKey,
        current: &DependencySnapshot,
        additional_bytes: u64,
        max_versions_per_entry: usize,
        max_code_bytes: u64,
    ) -> Result<usize, &'static str> {
        if max_versions_per_entry < 2 || max_code_bytes == 0 {
            return Err("specialized publication requires generic and specialized capacity");
        }
        let mut retired = 0;
        if let Some(cell) = self.cells.get(key) {
            while cell.retained_versions(current) + 1 > max_versions_per_entry {
                if cell.retire_oldest_specialized().is_none() {
                    return Err("generic native entry leaves no specialized version capacity");
                }
                retired += 1;
            }
        }
        while self
            .retained_code_bytes(current)
            .checked_add(additional_bytes)
            .is_none_or(|total| total > max_code_bytes)
        {
            let oldest = self
                .cells
                .iter()
                .filter_map(|(candidate, cell)| {
                    cell.oldest_specialized_publication()
                        .map(|publication| (publication, candidate.clone()))
                })
                .min();
            let Some((_, candidate)) = oldest else {
                return Err("generic native code and replacement exceed the code-memory budget");
            };
            self.cells[&candidate].retire_oldest_specialized();
            retired += 1;
        }
        Ok(retired)
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

    pub fn retained_code_bytes(&self, current: &DependencySnapshot) -> u64 {
        self.cells.values().fold(0_u64, |total, cell| {
            total.saturating_add(cell.retained_code_bytes(current))
        })
    }

    pub fn specialized_version_count(&self, current: &DependencySnapshot) -> usize {
        self.cells
            .values()
            .map(|cell| cell.specialized_version_count(current))
            .sum()
    }
}
