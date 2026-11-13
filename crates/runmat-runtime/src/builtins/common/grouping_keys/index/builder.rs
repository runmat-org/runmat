use std::collections::{btree_map::IntoKeys, BTreeMap};

use super::{GroupIndex, GroupKey, KeyOrder};
use crate::builtins::common::grouping_keys::group_key_work_units;

pub(super) enum BuildProgress {
    Work,
    Complete(GroupIndex),
}

/// Scheduling cost is requested by the cooperative driver only. A synchronous
/// caller runs the same grouping state machine without measuring work units.
fn record_work(work_units: Option<&mut u64>, key: Option<&[super::KeyAtom]>) {
    if let Some(units) = work_units {
        *units = key.map_or(1, group_key_work_units);
    }
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

    pub(super) fn step(&mut self, work_units: Option<&mut u64>) -> Result<BuildProgress, String> {
        loop {
            match self.phase {
                Phase::CollectSorted => {
                    if let Some(row) = self.rows.get(self.cursor) {
                        if let Some(key) = row {
                            if !self.sorted.contains_key(key) {
                                self.sorted.insert(key.clone(), ());
                            }
                        }
                        record_work(work_units, row.as_deref());
                        self.cursor += 1;
                        return Ok(BuildProgress::Work);
                    }
                    self.sorted_keys = Some(std::mem::take(&mut self.sorted).into_keys());
                    self.phase = Phase::DrainSorted;
                }
                Phase::DrainSorted => {
                    if let Some(key) = self.sorted_keys.as_mut().and_then(Iterator::next) {
                        record_work(work_units, Some(&key));
                        self.keys.push(key);
                        return Ok(BuildProgress::Work);
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
                        record_work(work_units, row.as_deref());
                        self.cursor += 1;
                        return Ok(BuildProgress::Work);
                    }
                    self.cursor = 0;
                    self.phase = Phase::IndexKeys;
                }
                Phase::ValidateExplicit => {
                    if let Some(key) = self.keys.get(self.cursor) {
                        record_work(work_units, Some(key));
                        if self.key_indices.insert(key.clone(), self.cursor).is_some() {
                            return Err("explicit group order contains a duplicate key".into());
                        }
                        self.cursor += 1;
                        return Ok(BuildProgress::Work);
                    }
                    self.cursor = 0;
                    self.phase = Phase::IndexKeys;
                }
                Phase::IndexKeys => {
                    if let Some(key) = self.keys.get(self.cursor) {
                        record_work(work_units, Some(key));
                        if !self.key_indices.contains_key(key) {
                            self.key_indices.insert(key.clone(), self.cursor);
                        }
                        self.first_rows.push(None);
                        self.row_groups.push(Vec::new());
                        self.cursor += 1;
                        return Ok(BuildProgress::Work);
                    }
                    self.cursor = 0;
                    self.phase = Phase::AssignRows;
                }
                Phase::AssignRows => {
                    if let Some(row) = self.rows.get_mut(self.cursor) {
                        let current = self.cursor;
                        if let Some(key) = row.take() {
                            record_work(work_units, Some(&key));
                            let index = self.key_indices.get(&key).copied().ok_or_else(|| {
                                "group observation is absent from the explicit order".to_string()
                            })?;
                            self.ids.push(index as f64 + 1.0);
                            self.first_rows[index].get_or_insert(current);
                            self.row_groups[index].push(current);
                        } else {
                            record_work(work_units, None);
                            self.ids.push(f64::NAN);
                        }
                        self.cursor += 1;
                        return Ok(BuildProgress::Work);
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
