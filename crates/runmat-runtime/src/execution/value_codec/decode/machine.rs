//! Shared synchronous/cooperative decoder state machine.
//!
//! The machine contains no scheduling policy. `advance` performs at most the
//! requested amount of task-accounted source work; the synchronous frontend
//! repeatedly calls it, while the invocation frontend checkpoints between
//! advances. Canonical constructors that still validate a completed shape or
//! repack a completed aggregate remain a separate bounded-admission frontier.

use super::*;
use crate::execution::{RootedValueChildren, RootedValueMember};
use runmat_execution::value::ValuePayload;
use std::num::NonZeroUsize;
use std::rc::Rc;

#[path = "machine/task.rs"]
mod task;
use task::Task;

const TASK_CHUNK_CAPACITY: usize = 64;

struct TaskStack<'a> {
    current: Vec<Task<'a>>,
    previous: Option<Box<TaskChunk<'a>>>,
}

struct TaskChunk<'a> {
    tasks: Vec<Task<'a>>,
    previous: Option<Box<TaskChunk<'a>>>,
}

impl<'a> TaskStack<'a> {
    fn new(task: Task<'a>) -> Self {
        let mut current = Vec::with_capacity(TASK_CHUNK_CAPACITY);
        current.push(task);
        Self {
            current,
            previous: None,
        }
    }

    #[cfg(test)]
    fn empty() -> Self {
        Self {
            current: Vec::new(),
            previous: None,
        }
    }

    pub(super) fn push(&mut self, task: Task<'a>) {
        if self.current.len() == TASK_CHUNK_CAPACITY {
            let chunk = TaskChunk {
                tasks: std::mem::replace(
                    &mut self.current,
                    Vec::with_capacity(TASK_CHUNK_CAPACITY),
                ),
                previous: self.previous.take(),
            };
            self.previous = Some(Box::new(chunk));
        }
        self.current.push(task);
    }

    fn pop(&mut self) -> Option<Task<'a>> {
        if let Some(task) = self.current.pop() {
            return Some(task);
        }
        let chunk = self.previous.take()?;
        let TaskChunk { tasks, previous } = *chunk;
        self.current = tasks;
        self.previous = previous;
        self.current.pop()
    }

    fn is_empty(&self) -> bool {
        self.current.is_empty() && self.previous.is_none()
    }
}

impl Drop for TaskStack<'_> {
    fn drop(&mut self) {
        let mut previous = self.previous.take();
        while let Some(mut chunk) = previous {
            previous = chunk.previous.take();
        }
    }
}

pub(super) enum DecodeProgress {
    Pending { completed_work: usize },
    Complete { completed_work: usize },
}

pub(super) type PartialValue = RootedValueMember;

/// Child values removed from the machine while an aggregate is assembled.
/// Their explicit roots remain alive until the completed parent has itself
/// taken ownership of those same roots through `push_aggregate`.
pub(super) struct TakenValues {
    values: std::vec::IntoIter<PartialValue>,
    roots: RootedValueChildren,
}

impl TakenValues {
    pub(super) fn from_partials(partials: Vec<PartialValue>) -> Self {
        Self {
            values: partials.into_iter(),
            roots: RootedValueChildren::new(),
        }
    }
    pub(super) fn next(&mut self) -> Option<Value> {
        let partial = self.values.next()?;
        Some(partial.into_child(&mut self.roots))
    }

    pub(super) fn remaining(&self) -> usize {
        self.values.len()
    }

    pub(super) fn take_roots(&mut self) -> RootedValueChildren {
        assert_eq!(
            self.values.len(),
            0,
            "all child values must be materialized"
        );
        std::mem::take(&mut self.roots)
    }
}

pub(super) struct DecodeMachine<'a> {
    tasks: TaskStack<'a>,
    values: Vec<PartialValue>,
    source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
    frame_sources: Option<&'a dyn super::InlineFrameSourceAuthority>,
}

impl<'a> DecodeMachine<'a> {
    pub(super) fn empty_taken_values() -> TakenValues {
        TakenValues {
            values: Vec::new().into_iter(),
            roots: RootedValueChildren::new(),
        }
    }

    pub(super) fn new(
        payload: &'a ValuePayload,
        path: String,
        source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
    ) -> Self {
        Self::new_with_frame_sources(payload, path, source_resolver, None)
    }

    pub(super) fn new_with_frame_sources(
        payload: &'a ValuePayload,
        path: String,
        source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
        frame_sources: Option<&'a dyn super::InlineFrameSourceAuthority>,
    ) -> Self {
        Self {
            tasks: TaskStack::new(Task::Decode { payload, path }),
            values: Vec::new(),
            source_resolver,
            frame_sources,
        }
    }

    pub(super) fn frame_sources(&self) -> Option<&'a dyn super::InlineFrameSourceAuthority> {
        self.frame_sources
    }

    pub(super) fn source_resolver(
        &self,
    ) -> Option<Rc<crate::user_functions::ExecutionSourceResolver>> {
        self.source_resolver.clone()
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: NonZeroUsize,
    ) -> Result<DecodeProgress, ValueCodecError> {
        let mut remaining = maximum_work.get();
        while remaining != 0 {
            let Some(task) = self.tasks.pop() else {
                debug_assert_eq!(self.values.len(), 1);
                return Ok(DecodeProgress::Complete {
                    completed_work: maximum_work.get() - remaining,
                });
            };
            let completed = task::advance(task, self, remaining)?;
            debug_assert!(completed != 0 && completed <= remaining);
            if self.tasks.is_empty() && self.values.len() == 1 {
                return Ok(DecodeProgress::Complete {
                    completed_work: maximum_work.get() - remaining + completed,
                });
            }
            remaining -= completed;
        }
        Ok(DecodeProgress::Pending {
            completed_work: maximum_work.get(),
        })
    }

    /// Transfer the completed value only after the caller's cooperation
    /// checkpoint. The machine keeps its explicit root alive across that await.
    pub(super) fn take_completed(&mut self) -> RootedValueMember {
        assert!(self.tasks.is_empty(), "decoder still has pending work");
        assert_eq!(self.values.len(), 1, "completed decoder owns one result");
        self.values.pop().expect("completed decoder result")
    }

    pub(super) fn push_value(&mut self, value: Value, path: &str) -> Result<(), ValueCodecError> {
        let value = RootedValueMember::root(value).map_err(|error| {
            ValueCodecError::invalid(path, format!("could not root decoded value: {error}"))
        })?;
        self.values.push(value);
        Ok(())
    }

    pub(super) fn push_aggregate(&mut self, value: Value, mut retained: TakenValues) {
        self.values
            .push(RootedValueMember::aggregate(value, retained.take_roots()));
    }

    pub(super) fn pop_partial(&mut self) -> PartialValue {
        self.values.pop().expect("decoded child result")
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use runmat_value::HandleRef;

    #[test]
    fn aggregate_transfers_child_roots_without_retracing_the_parent() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("aggregate".into()))
                .expect("allocate managed payload");
            let handle = initial.handle();
            let child = Value::HandleObject(HandleRef {
                class_name: "DecodedAggregateHandle".into(),
                target: handle,
                valid: true,
            });
            let child = RootedValueMember::root(child).expect("root decoded child");
            initial.unroot().expect("transfer initial root");

            let mut children = TakenValues::from_partials(vec![child]);
            runmat_gc::gc_collect_major().expect("collect with partial aggregate");
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            let cell = Value::Cell(
                CellArray::new_with_shape(
                    vec![children.next().expect("decoded child")],
                    vec![1, 1],
                )
                .expect("valid cell"),
            );
            let mut machine = DecodeMachine {
                tasks: TaskStack::empty(),
                values: Vec::new(),
                source_resolver: None,
                frame_sources: None,
            };
            machine.push_aggregate(cell, children);
            let DecodeProgress::Complete { .. } = machine
                .advance(NonZeroUsize::new(1).unwrap())
                .expect("complete aggregate")
            else {
                panic!("aggregate should be complete");
            };
            let decoded = machine.take_completed();

            runmat_gc::gc_collect_major().expect("collect with aggregate owner");
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            drop(decoded);
            runmat_gc::gc_collect_major().expect("collect after aggregate drop");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
