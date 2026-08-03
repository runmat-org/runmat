use std::collections::BTreeMap;

use runmat_execution::state::TaskState;
use runmat_execution::TaskId;

use crate::driver::{DriverEvent, DriverEventKind};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReferenceModel {
    pub tasks: BTreeMap<TaskId, TaskState>,
}

impl ReferenceModel {
    pub fn apply(&mut self, event: &DriverEvent) {
        match event.kind {
            DriverEventKind::TaskSubmitted { task_id, state }
            | DriverEventKind::TaskStateChanged { task_id, state } => {
                self.tasks.insert(task_id, state);
            }
            DriverEventKind::ResultCommitted { task_id, .. } => {
                self.tasks.insert(task_id, TaskState::Succeeded);
            }
            _ => {}
        }
    }
}
