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
#[path = "machine/control.rs"]
mod control;
use control::{ControlBox, ControlIntoIter, ControlOwner, ControlPath, ControlVec};

const TASK_CHUNK_CAPACITY: usize = 64;

struct TaskStack<'a> {
    current: ControlVec<'a, Task<'a>>,
    previous: Option<ControlBox<TaskChunk<'a>>>,
    owner: ControlOwner<'a>,
}

struct TaskChunk<'a> {
    tasks: ControlVec<'a, Task<'a>>,
    previous: Option<ControlBox<TaskChunk<'a>>>,
}

impl<'a> TaskStack<'a> {
    fn new(task: Task<'a>, owner: ControlOwner<'a>) -> Result<Self, ValueCodecError> {
        let mut current = owner.vector(TASK_CHUNK_CAPACITY, "$")?;
        current.push(task, "$")?;
        Ok(Self {
            current,
            previous: None,
            owner,
        })
    }

    #[cfg(test)]
    fn empty(owner: ControlOwner<'a>) -> Result<Self, ValueCodecError> {
        Ok(Self {
            current: owner.vector(0, "$")?,
            previous: None,
            owner,
        })
    }

    pub(super) fn push(&mut self, task: Task<'a>) -> Result<(), ValueCodecError> {
        if self.current.len() == TASK_CHUNK_CAPACITY {
            // Complete both fallible allocations before moving the linked
            // chain. Failure must still retire a deep stack iteratively.
            let current = self.owner.vector(TASK_CHUNK_CAPACITY, "$")?;
            let mut chunk = self.owner.boxed(
                TaskChunk {
                    tasks: self.owner.vector(0, "$")?,
                    previous: None,
                },
                "$",
            )?;
            chunk.tasks = std::mem::replace(&mut self.current, current);
            chunk.previous = self.previous.take();
            self.previous = Some(chunk);
        }
        self.current.push(task, "$")
    }

    fn pop(&mut self) -> Option<Task<'a>> {
        if let Some(task) = self.current.pop() {
            return Some(task);
        }
        let chunk = self.previous.take()?;
        let TaskChunk { tasks, previous } = allocator_api2::boxed::Box::into_inner(chunk);
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
pub(super) struct TakenValues<'a> {
    values: Option<ControlIntoIter<PartialValue>>,
    roots: RootedValueChildren<'a>,
}

impl<'a> TakenValues<'a> {
    fn from_partials(
        partials: ControlVec<'a, PartialValue>,
        runtime: Option<&'a crate::context::RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let roots = match runtime {
            Some(runtime) => RootedValueChildren::in_context(runtime)
                .map_err(|source| ValueCodecError::allocation("$", source))?,
            None => RootedValueChildren::new(),
        };
        Ok(Self {
            values: Some(partials.into_iter()),
            roots,
        })
    }
    pub(super) fn next(&mut self) -> Result<Option<Value>, ValueCodecError> {
        let Some(partial) = self.values.as_mut().and_then(Iterator::next) else {
            return Ok(None);
        };
        partial
            .into_child(&mut self.roots)
            .map(Some)
            .map_err(|source| ValueCodecError::allocation("$", source))
    }

    pub(super) fn remaining(&self) -> usize {
        self.values.as_ref().map_or(0, ExactSizeIterator::len)
    }

    pub(super) fn take_roots(&mut self) -> RootedValueChildren<'a> {
        assert_eq!(self.remaining(), 0, "all child values must be materialized");
        std::mem::take(&mut self.roots)
    }
}

pub(super) struct DecodeMachine<'a> {
    tasks: TaskStack<'a>,
    values: ControlVec<'a, PartialValue>,
    control: ControlOwner<'a>,
    source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
    frame_sources: Option<&'a dyn super::InlineFrameSourceAuthority>,
    runtime: Option<&'a crate::context::RuntimeContext>,
}

impl<'a> DecodeMachine<'a> {
    pub(super) fn empty_taken_values() -> TakenValues<'a> {
        TakenValues {
            values: None,
            roots: RootedValueChildren::new(),
        }
    }

    pub(super) fn new_with_runtime(
        payload: &'a ValuePayload,
        path: &str,
        source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
        runtime: Option<&'a crate::context::RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        Self::construct(payload, path, source_resolver, None, runtime)
    }

    pub(super) fn new_with_frame_sources(
        payload: &'a ValuePayload,
        path: &str,
        source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
        frame_sources: Option<&'a dyn super::InlineFrameSourceAuthority>,
    ) -> Result<Self, ValueCodecError> {
        Self::construct(payload, path, source_resolver, frame_sources, None)
    }

    fn construct(
        payload: &'a ValuePayload,
        path: &str,
        source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
        frame_sources: Option<&'a dyn super::InlineFrameSourceAuthority>,
        runtime: Option<&'a crate::context::RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let control = ControlOwner::new(runtime)?;
        let path = control.path(path)?;
        Ok(Self {
            tasks: TaskStack::new(Task::Decode { payload, path }, control.clone())?,
            values: control.vector(0, "$")?,
            control,
            source_resolver,
            frame_sources,
            runtime,
        })
    }

    pub(super) fn frame_sources(&self) -> Option<&'a dyn super::InlineFrameSourceAuthority> {
        self.frame_sources
    }

    pub(super) fn runtime(&self) -> Option<&'a crate::context::RuntimeContext> {
        self.runtime
    }

    pub(super) fn boxed<T>(&self, value: T) -> Result<ControlBox<T>, ValueCodecError> {
        self.control.boxed(value, "$")
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
        let value = if let Some(runtime) = self.runtime {
            RootedValueMember::root_in_context(value, runtime).map_err(|error| match error {
                crate::execution::RootedValueAdmissionError::Host(source) => {
                    ValueCodecError::allocation(path, source)
                }
                crate::execution::RootedValueAdmissionError::Gc(source) => {
                    ValueCodecError::root_registration(path, source)
                }
            })?
        } else {
            RootedValueMember::root(value)
                .map_err(|source| ValueCodecError::root_registration(path, source))?
        };
        self.values.push(value, path)?;
        Ok(())
    }

    pub(super) fn push_aggregate(
        &mut self,
        value: Value,
        mut retained: TakenValues<'a>,
    ) -> Result<(), ValueCodecError> {
        self.values.push(
            RootedValueMember::aggregate(value, retained.take_roots())
                .map_err(|source| ValueCodecError::allocation("$", source))?,
            "$",
        )
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

            let control = ControlOwner::new(None).expect("unbounded decoder controls");
            let mut partials = control.vector(1, "$").unwrap();
            partials.push(child, "$").unwrap();
            let mut children = TakenValues::from_partials(partials, None).unwrap();
            runmat_gc::gc_collect_major().expect("collect with partial aggregate");
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            let cell = Value::Cell(
                CellArray::new_with_shape(
                    vec![children.next().unwrap().expect("decoded child")],
                    vec![1, 1],
                )
                .expect("valid cell"),
            );
            let mut machine = DecodeMachine {
                tasks: TaskStack::empty(control.clone()).unwrap(),
                values: control.vector(0, "$").unwrap(),
                control,
                source_resolver: None,
                frame_sources: None,
                runtime: None,
            };
            machine.push_aggregate(cell, children).unwrap();
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
