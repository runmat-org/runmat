//! Session-owned timer registry and lifecycle authority.
//!
//! A timer's object properties are presentation; this state owns its typed
//! scheduling mode, generation, task lease, and callback-once decisions.

use super::{RuntimeContext, RuntimeLocalTaskLease};
use crate::builtins::timing::timer::property::{
    TimerBusyMode, TimerExecutionMode, TimerVisibility,
};
use runmat_async::RuntimeErrorOwner;
use runmat_gc::ExplicitRoot;
use runmat_value::HandleRef;
use std::cell::RefCell;
use std::rc::Rc;

/// A generation's failure survives replacement of the timer's current record
/// while a waiter still holds this receipt; otherwise it drops with the run.
#[derive(Clone, Default)]
pub(crate) struct TimerFailureReceipt(Rc<RefCell<Option<RuntimeErrorOwner>>>);

impl TimerFailureReceipt {
    pub(crate) fn record(&self, error: RuntimeErrorOwner) {
        *self.0.borrow_mut() = Some(error);
    }

    pub(crate) fn take(&self) -> Option<RuntimeErrorOwner> {
        self.0.borrow_mut().take()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TimerPhase {
    Idle,
    Running,
    Stopped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TimerAdmissionError {
    ForeignOrDeleted,
    AlreadyRunning,
    GenerationExhausted,
}

pub(crate) struct TimerRecord {
    pub(crate) handle: HandleRef,
    // Session discovery keeps the object and its traced callback graph alive
    // after the constructor result leaves the workspace.
    _root: ExplicitRoot,
    pub(crate) generation: u64,
    pub(crate) phase: TimerPhase,
    pub(crate) busy_mode: TimerBusyMode,
    pub(crate) execution_mode: TimerExecutionMode,
    pub(crate) visibility: TimerVisibility,
    pub(crate) lease: Option<RuntimeLocalTaskLease>,
    pub(crate) stop_claimed: bool,
    pub(crate) failure: TimerFailureReceipt,
}

#[derive(Default)]
pub(crate) struct TimerSessionState {
    records: Vec<TimerRecord>,
    next_name: u64,
    next_generation: u64,
}

impl std::fmt::Debug for TimerSessionState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TimerSessionState")
            .field("timer_count", &self.records.len())
            .finish()
    }
}

impl TimerSessionState {
    pub(crate) fn next_name(&mut self) -> Option<String> {
        self.next_name = self.next_name.checked_add(1)?;
        Some(format!("timer-{}", self.next_name))
    }

    pub(crate) fn register(
        &mut self,
        handle: HandleRef,
        root: ExplicitRoot,
        visibility: TimerVisibility,
    ) {
        self.records.push(TimerRecord {
            handle,
            _root: root,
            generation: 0,
            phase: TimerPhase::Idle,
            busy_mode: TimerBusyMode::Drop,
            execution_mode: TimerExecutionMode::SingleShot,
            visibility,
            lease: None,
            stop_claimed: false,
            failure: TimerFailureReceipt::default(),
        });
    }

    pub(crate) fn handles(&self) -> Vec<HandleRef> {
        self.records
            .iter()
            .map(|entry| entry.handle.clone())
            .collect()
    }

    pub(crate) fn record(&self, handle: &HandleRef) -> Option<&TimerRecord> {
        self.records.iter().find(|entry| entry.handle == *handle)
    }

    pub(crate) fn record_mut(&mut self, handle: &HandleRef) -> Option<&mut TimerRecord> {
        self.records
            .iter_mut()
            .find(|entry| entry.handle == *handle)
    }

    pub(crate) fn admit_start(
        &mut self,
        handle: &HandleRef,
        busy_mode: TimerBusyMode,
        execution_mode: TimerExecutionMode,
        visibility: TimerVisibility,
    ) -> Result<(u64, TimerFailureReceipt), TimerAdmissionError> {
        let index = self
            .records
            .iter()
            .position(|entry| entry.handle == *handle)
            .ok_or(TimerAdmissionError::ForeignOrDeleted)?;
        if self.records[index].phase == TimerPhase::Running {
            return Err(TimerAdmissionError::AlreadyRunning);
        }
        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(TimerAdmissionError::GenerationExhausted)?;
        self.next_generation = generation;
        let entry = &mut self.records[index];
        entry.generation = generation;
        entry.phase = TimerPhase::Running;
        entry.busy_mode = busy_mode;
        entry.execution_mode = execution_mode;
        entry.visibility = visibility;
        entry.lease = None;
        entry.stop_claimed = false;
        entry.failure = TimerFailureReceipt::default();
        Ok((generation, entry.failure.clone()))
    }

    pub(crate) fn attach_lease(
        &mut self,
        handle: &HandleRef,
        generation: u64,
        lease: RuntimeLocalTaskLease,
    ) -> Result<(), RuntimeLocalTaskLease> {
        let Some(entry) = self.record_mut(handle) else {
            return Err(lease);
        };
        if entry.generation != generation || entry.phase != TimerPhase::Running {
            return Err(lease);
        }
        entry.lease = Some(lease);
        Ok(())
    }

    pub(crate) fn finish(&mut self, handle: &HandleRef, generation: u64) -> Option<bool> {
        let entry = self.record_mut(handle)?;
        if entry.generation != generation || entry.phase != TimerPhase::Running {
            return None;
        }
        entry.phase = TimerPhase::Stopped;
        entry.lease = None;
        if entry.stop_claimed {
            Some(false)
        } else {
            entry.stop_claimed = true;
            Some(true)
        }
    }

    pub(crate) fn stop(
        &mut self,
        handle: &HandleRef,
    ) -> Option<(Option<RuntimeLocalTaskLease>, bool)> {
        let entry = self.record_mut(handle)?;
        if entry.phase != TimerPhase::Running {
            return Some((None, false));
        }
        entry.phase = TimerPhase::Stopped;
        let lease = entry.lease.take();
        let call_stop = !entry.stop_claimed;
        entry.stop_claimed = true;
        Some((lease, call_stop))
    }

    pub(crate) fn delete(&mut self, handle: &HandleRef) -> Option<RuntimeLocalTaskLease> {
        let index = self
            .records
            .iter()
            .position(|entry| entry.handle == *handle)?;
        let entry = self.records.remove(index);
        entry.lease
    }
}

impl Drop for TimerSessionState {
    fn drop(&mut self) {
        for entry in self.records.drain(..) {
            if let Some(lease) = entry.lease {
                lease.cancel();
            }
        }
    }
}

impl RuntimeContext {
    pub(crate) fn timer_session(&self) -> &std::cell::RefCell<TimerSessionState> {
        &self.state().timers
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        handle_identity, test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };
    use runmat_value::{Closure, ObjectInstance, Value};

    #[test]
    fn session_root_retains_discoverable_timer_and_traced_callback_capture() {
        runmat_gc::gc_test_context(|| {
            let captured = runmat_gc::gc_allocate_rooted(Value::String("capture".into()))
                .expect("captured handle");
            let captured_handle = captured.handle();
            let resident = TestAllocationOwner::new()
                .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .expect("resident callback capture")
                .handle;
            let mut object = ObjectInstance::new("timer");
            object.replace_property_unbounded(
                "TimerFcn".into(),
                Value::Closure(Closure::from_data(runmat_value::ClosureData {
                    lexical_environment: None,
                    function_name: "callback".into(),
                    binding: None,
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                    captures: vec![
                        Value::HandleObject(HandleRef {
                            class_name: "Capture".into(),
                            target: captured_handle,
                            valid: true,
                        }),
                        Value::GpuTensor(resident.clone()),
                    ]
                    .into(),
                })),
            );
            let timer = runmat_gc::gc_allocate_rooted(Value::Object(object)).expect("timer object");
            let handle = HandleRef {
                class_name: "timer".into(),
                target: timer.handle(),
                valid: true,
            };
            let mut session = TimerSessionState::default();
            session.register(
                handle.clone(),
                runmat_gc::gc_root(handle.target).expect("session root"),
                TimerVisibility::On,
            );
            timer.unroot().expect("release constructor root");
            captured.unroot().expect("release callback creator root");

            runmat_gc::gc_collect_major().expect("collect while discoverable");
            assert!(runmat_gc::gc_clone_value(&handle.target).is_ok());
            assert!(runmat_gc::gc_clone_value(&captured_handle).is_ok());
            let mut resident_aliases = Vec::new();
            assert_eq!(
                runmat_gc::gc_visit_retained_values(&[], &mut |value| {
                    crate::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
                        resident_aliases.push(handle_identity(handle));
                    })
                    .ok()?;
                    Some(())
                }),
                Some(())
            );
            assert!(resident_aliases.contains(&handle_identity(&resident)));

            session.delete(&handle);
            resident_aliases.clear();
            assert_eq!(
                runmat_gc::gc_visit_retained_values(&[], &mut |value| {
                    crate::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
                        resident_aliases.push(handle_identity(handle));
                    })
                    .ok()?;
                    Some(())
                }),
                Some(())
            );
            assert!(!resident_aliases.contains(&handle_identity(&resident)));
            runmat_gc::gc_collect_major().expect("collect after deletion");
            assert!(runmat_gc::gc_clone_value(&handle.target).is_err());
            assert!(runmat_gc::gc_clone_value(&captured_handle).is_err());
        });
    }
}
