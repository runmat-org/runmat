//! Exact caller-declared input closures; the generic pool does not inspect domain payloads.
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::{value::ValueRef, Digest, TaskId};
use runmat_execution_artifact::ProgramExecutionRequest;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Default)]
pub(super) struct InputInventories {
    programs: HashMap<Digest, Vec<ValueRef>>,
    tasks: HashMap<TaskId, (Digest, Option<Vec<ValueRef>>)>,
}

fn identity(program: &ProgramExecutionRequest) -> NativeExecutionResult<Digest> {
    program.canonical_digest().map_err(invalid)
}

impl InputInventories {
    pub(super) fn register(
        &mut self,
        program: &ProgramExecutionRequest,
        references: Vec<ValueRef>,
    ) -> NativeExecutionResult<()> {
        let key = identity(program)?;
        let mut unique = BTreeMap::new();
        for reference in references {
            if let Some(previous) = unique.insert(reference.id, reference.clone()) {
                if previous != reference {
                    return Err(invalid("conflicting program input references"));
                }
            }
        }
        let references: Vec<_> = unique.into_values().collect();
        for root in super::super::value_transfer::collect_references(&program.arguments) {
            if !references.contains(&root) {
                return Err(invalid("program input inventory omits an argument root"));
            }
        }
        if self
            .programs
            .get(&key)
            .is_some_and(|existing| existing != &references)
            || self.tasks.values().any(|(existing_key, inventory)| {
                existing_key == &key && inventory.as_ref() != Some(&references)
            })
        {
            return Err(invalid(
                "program input inventory changed after registration or admission",
            ));
        }
        self.programs.insert(key, references);
        Ok(())
    }

    pub(super) fn bind(
        &mut self,
        task: TaskId,
        program: &ProgramExecutionRequest,
    ) -> NativeExecutionResult<()> {
        let key = identity(program)?;
        let binding = (key, self.programs.get(&key).cloned());
        if self
            .tasks
            .get(&task)
            .is_some_and(|existing| existing != &binding)
        {
            return Err(invalid(
                "task input inventory differs from its immutable admission",
            ));
        }
        self.tasks.insert(task, binding);
        Ok(())
    }

    pub(super) fn validate(
        &self,
        task: TaskId,
        program: &ProgramExecutionRequest,
    ) -> NativeExecutionResult<()> {
        let key = identity(program)?;
        if self
            .tasks
            .get(&task)
            .is_some_and(|existing| existing != &(key, self.programs.get(&key).cloned()))
        {
            return Err(invalid(
                "task input inventory differs from its immutable admission",
            ));
        }
        Ok(())
    }

    pub(super) fn get(&self, task: TaskId) -> Option<Vec<ValueRef>> {
        self.tasks
            .get(&task)
            .and_then(|(_, inventory)| inventory.clone())
    }
}

fn invalid(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
