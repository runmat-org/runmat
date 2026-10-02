use std::collections::{btree_map::IntoKeys, BTreeMap};

use super::{GroupIndex, GroupKey, KeyOrder};
#[cfg(any(feature = "plot-core", test))]
use crate::builtins::common::grouping_keys::group_key_work_units;

macro_rules! work_progress {
    ($units:expr) => {{
        #[cfg(any(feature = "plot-core", test))]
        {
            BuildProgress::Work($units)
        }
        #[cfg(not(any(feature = "plot-core", test)))]
        {
            BuildProgress::Work
        }
    }};
}

pub(super) enum BuildProgress {
    #[cfg(any(feature = "plot-core", test))]
    Work(u64),
    #[cfg(not(any(feature = "plot-core", test)))]
    Work,
    Complete(GroupIndex),
}

enum Phase {
    CollectSorted,
    DrainSorted,
    CollectFirstAppearance,
    ValidateExplicit,
    IndexKeys,
    AssignRows,
    Complete,
}

/// A single resumable exact-key algorithm. Drivers decide whether to await a
/// service checkpoint after each bounded piece of work; ordering and errors
/// are independent of that execution policy.
pub(super) struct GroupIndexBuilder {
    rows: Vec<Option<GroupKey>>,
    keys: Vec<GroupKey>,
    ids: Vec<f64>,
    first_rows: Vec<Option<usize>>,
    row_groups: Vec<Vec<usize>>,
    sorted: BTreeMap<GroupKey, ()>,
    sorted_keys: Option<IntoKeys<GroupKey, ()>>,
    key_indices: BTreeMap<GroupKey, usize>,
    phase: Phase,
    cursor: usize,
}

impl GroupIndexBuilder {
    pub(super) fn new(rows: Vec<Option<GroupKey>>, order: KeyOrder) -> Self {
        let (phase, keys) = match order {
            KeyOrder::Sorted => (Phase::CollectSorted, Vec::new()),
            KeyOrder::FirstAppearance => (Phase::CollectFirstAppearance, Vec::new()),
            KeyOrder::Explicit(keys) => (Phase::ValidateExplicit, keys),
        };
        Self {
            rows,
            keys,
            ids: Vec::new(),
            first_rows: Vec::new(),
            row_groups: Vec::new(),
            sorted: BTreeMap::new(),
            sorted_keys: None,
            key_indices: BTreeMap::new(),
            phase,
            cursor: 0,
        }
    }

    pub(super) fn step(&mut self) -> Result<BuildProgress, String> {
        loop {
            match self.phase {
                Phase::CollectSorted => {
                    if let Some(row) = self.rows.get(self.cursor) {
                        if let Some(key) = row {
                            if !self.sorted.contains_key(key) {
                                self.sorted.insert(key.clone(), ());
                            }
                        }
                        let progress =
                            work_progress!(row.as_ref().map_or(1, |key| group_key_work_units(key)));
                        self.cursor += 1;
                        return Ok(progress);
                    }
                    self.sorted_keys = Some(std::mem::take(&mut self.sorted).into_keys());
                    self.phase = Phase::DrainSorted;
                }
                Phase::DrainSorted => {
                    if let Some(key) = self.sorted_keys.as_mut().and_then(Iterator::next) {
                        let progress = work_progress!(group_key_work_units(&key));
                        self.keys.push(key);
                        return Ok(progress);
                    }
                    self.sorted_keys = None;
                    self.cursor = 0;
                    self.phase = Phase::IndexKeys;
                }
                Phase::CollectFirstAppearance => {
                    if let Some(row) = self.rows.get(self.cursor) {
                        if let Some(key) = row {
                            if !self.key_indices.contains_key(key) {
                                self.key_indices.insert(key.clone(), self.keys.len());
                                self.keys.push(key.clone());
                            }
                        }
                        let progress =
                            work_progress!(row.as_ref().map_or(1, |key| group_key_work_units(key)));
                        self.cursor += 1;
                        return Ok(progress);
                    }
                    self.cursor = 0;
                    self.phase = Phase::IndexKeys;
                }
                Phase::ValidateExplicit => {
                    if let Some(key) = self.keys.get(self.cursor) {
                        let progress = work_progress!(group_key_work_units(key));
                        if self.key_indices.insert(key.clone(), self.cursor).is_some() {
                            return Err("explicit group order contains a duplicate key".into());
                        }
                        self.cursor += 1;
                        return Ok(progress);
                    }
                    self.cursor = 0;
                    self.phase = Phase::IndexKeys;
                }
                Phase::IndexKeys => {
                    if let Some(key) = self.keys.get(self.cursor) {
                        let progress = work_progress!(group_key_work_units(key));
                        if !self.key_indices.contains_key(key) {
                            self.key_indices.insert(key.clone(), self.cursor);
                        }
                        self.first_rows.push(None);
                        self.row_groups.push(Vec::new());
                        self.cursor += 1;
                        return Ok(progress);
                    }
                    self.cursor = 0;
                    self.phase = Phase::AssignRows;
                }
                Phase::AssignRows => {
                    if let Some(row) = self.rows.get_mut(self.cursor) {
                        let current = self.cursor;
                        let progress = if let Some(key) = row.take() {
                            let progress = work_progress!(group_key_work_units(&key));
                            let index = self.key_indices.get(&key).copied().ok_or_else(|| {
                                "group observation is absent from the explicit order".to_string()
                            })?;
                            self.ids.push(index as f64 + 1.0);
                            self.first_rows[index].get_or_insert(current);
                            self.row_groups[index].push(current);
                            progress
                        } else {
                            self.ids.push(f64::NAN);
                            work_progress!(1)
                        };
                        self.cursor += 1;
                        return Ok(progress);
                    }
                    self.phase = Phase::Complete;
                }
                Phase::Complete => {
                    return Ok(BuildProgress::Complete(GroupIndex {
                        ids: std::mem::take(&mut self.ids),
                        keys: std::mem::take(&mut self.keys),
                        first_rows: std::mem::take(&mut self.first_rows),
                        row_groups: std::mem::take(&mut self.row_groups),
                    }));
                }
            }
        }
    }
}
