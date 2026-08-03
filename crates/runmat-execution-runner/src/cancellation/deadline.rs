use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::TaskId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeadlineIndex {
    by_deadline: BTreeMap<u64, BTreeSet<TaskId>>,
}

impl DeadlineIndex {
    pub fn insert(&mut self, task_id: TaskId, deadline_millis: Option<u64>) {
        if let Some(deadline) = deadline_millis {
            self.by_deadline
                .entry(deadline)
                .or_default()
                .insert(task_id);
        }
    }

    pub fn remove(&mut self, task_id: TaskId, deadline_millis: Option<u64>) {
        if let Some(deadline) = deadline_millis {
            if let Some(tasks) = self.by_deadline.get_mut(&deadline) {
                tasks.remove(&task_id);
                if tasks.is_empty() {
                    self.by_deadline.remove(&deadline);
                }
            }
        }
    }

    pub fn expired(&self, now_millis: u64) -> Vec<TaskId> {
        self.by_deadline
            .range(..=now_millis)
            .flat_map(|(_, tasks)| tasks.iter().copied())
            .collect()
    }
}
