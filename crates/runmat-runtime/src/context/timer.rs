//! Session-owned timer registry and lifecycle authority.
//!
//! A timer's object properties are presentation; this state owns its typed
//! scheduling mode, generation, task lease, and callback-once decisions.

use super::{RuntimeContext, RuntimeLocalTaskLease};
use crate::builtins::timing::timer::property::{
    TimerBusyMode, TimerExecutionMode, TimerVisibility,
};
use crate::RuntimeError;
use runmat_value::HandleRef;

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
    pub(crate) generation: u64,
    pub(crate) phase: TimerPhase,
    pub(crate) busy_mode: TimerBusyMode,
    pub(crate) execution_mode: TimerExecutionMode,
    pub(crate) visibility: TimerVisibility,
    pub(crate) lease: Option<RuntimeLocalTaskLease>,
    pub(crate) stop_claimed: bool,
    pub(crate) terminal_error: Option<RuntimeError>,
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

    pub(crate) fn register(&mut self, handle: HandleRef, visibility: TimerVisibility) {
        self.records.push(TimerRecord {
            handle,
            generation: 0,
            phase: TimerPhase::Idle,
            busy_mode: TimerBusyMode::Drop,
            execution_mode: TimerExecutionMode::SingleShot,
            visibility,
            lease: None,
            stop_claimed: false,
            terminal_error: None,
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
    ) -> Result<u64, TimerAdmissionError> {
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
        entry.terminal_error = None;
        Ok(generation)
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

    pub(crate) fn finish(
        &mut self,
        handle: &HandleRef,
        generation: u64,
        error: Option<RuntimeError>,
    ) -> Option<bool> {
        let Some(entry) = self.record_mut(handle) else {
            return None;
        };
        if entry.generation != generation || entry.phase != TimerPhase::Running {
            return None;
        }
        entry.phase = TimerPhase::Stopped;
        entry.terminal_error = error;
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

    pub(crate) fn take_terminal_error(&mut self, handle: &HandleRef) -> Option<RuntimeError> {
        self.record_mut(handle)?.terminal_error.take()
    }

    pub(crate) fn delete(&mut self, handle: &HandleRef) -> Option<RuntimeLocalTaskLease> {
        let Some(index) = self
            .records
            .iter()
            .position(|entry| entry.handle == *handle)
        else {
            return None;
        };
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
            let _ = runmat_gc::gc_remove_root(entry.handle.target);
        }
    }
}

impl RuntimeContext {
    pub(crate) fn timer_session(&self) -> &std::cell::RefCell<TimerSessionState> {
        &self.state().timers
    }
}
