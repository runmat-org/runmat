use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::TaskId;
use serde::{Deserialize, Serialize};

use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TaskGraph {
    dependencies: BTreeMap<TaskId, BTreeSet<TaskId>>,
    dependents: BTreeMap<TaskId, BTreeSet<TaskId>>,
}

impl TaskGraph {
    pub fn insert(&mut self, task_id: TaskId, dependencies: BTreeSet<TaskId>) -> RunnerResult<()> {
        if self.dependencies.contains_key(&task_id) {
            return Err(RunnerError::Invalid(format!(
                "task {task_id} is already registered"
            )));
        }
        if dependencies.contains(&task_id) {
            return Err(RunnerError::DependencyCycle);
        }
        self.dependencies.insert(task_id, dependencies.clone());
        for dependency in &dependencies {
            self.dependents
                .entry(*dependency)
                .or_default()
                .insert(task_id);
        }
        if self.has_cycle() {
            self.dependencies.remove(&task_id);
            for dependency in dependencies {
                if let Some(dependents) = self.dependents.get_mut(&dependency) {
                    dependents.remove(&task_id);
                }
            }
            return Err(RunnerError::DependencyCycle);
        }
        Ok(())
    }

    pub fn dependencies(&self, task_id: TaskId) -> Option<&BTreeSet<TaskId>> {
        self.dependencies.get(&task_id)
    }

    pub fn dependents(&self, task_id: TaskId) -> impl Iterator<Item = TaskId> + '_ {
        self.dependents.get(&task_id).into_iter().flatten().copied()
    }

    fn has_cycle(&self) -> bool {
        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        self.dependencies
            .keys()
            .copied()
            .any(|task| self.visit(task, &mut visiting, &mut visited))
    }

    fn visit(
        &self,
        task: TaskId,
        visiting: &mut BTreeSet<TaskId>,
        visited: &mut BTreeSet<TaskId>,
    ) -> bool {
        if visited.contains(&task) {
            return false;
        }
        if !visiting.insert(task) {
            return true;
        }
        if self
            .dependencies
            .get(&task)
            .into_iter()
            .flatten()
            .filter(|dependency| self.dependencies.contains_key(dependency))
            .copied()
            .any(|dependency| self.visit(dependency, visiting, visited))
        {
            return true;
        }
        visiting.remove(&task);
        visited.insert(task);
        false
    }
}
