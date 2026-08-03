use std::collections::BTreeSet;

use runmat_execution::TaskId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct QueueEntry {
    pub priority: i16,
    pub enqueued_sequence: u64,
    pub task_id: TaskId,
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| self.enqueued_sequence.cmp(&other.enqueued_sequence))
            .then_with(|| self.task_id.cmp(&other.task_id))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReadyQueue {
    entries: BTreeSet<QueueEntry>,
}

impl ReadyQueue {
    pub fn insert(&mut self, entry: QueueEntry) {
        self.entries.insert(entry);
    }

    pub fn remove_task(&mut self, task_id: TaskId) {
        self.entries.retain(|entry| entry.task_id != task_id);
    }

    pub fn ordered(&self) -> impl Iterator<Item = QueueEntry> + '_ {
        self.entries.iter().copied()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn task_ids(&self) -> impl Iterator<Item = TaskId> + '_ {
        self.entries.iter().map(|entry| entry.task_id)
    }

    pub fn contains(&self, task_id: TaskId) -> bool {
        self.entries.iter().any(|entry| entry.task_id == task_id)
    }
}
