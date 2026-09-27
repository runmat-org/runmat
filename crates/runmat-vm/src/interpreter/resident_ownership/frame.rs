//! Terminal ownership of the pinned interpreter frame.
use super::visibility::AllocationCollector;
use super::*;
use crate::interpreter::dispatch::SequenceState;
use std::ptr::NonNull;

/// Same pinned-frame lifetime as its GC registrations. The inner instruction
/// future is dropped first; this guard precedes all frame values in drop order.
pub(crate) struct FrameValues {
    stack: NonNull<Vec<Value>>,
    vars: NonNull<Vec<Value>>,
    locals: NonNull<Vec<Value>>,
    sequence: NonNull<SequenceState>,
    registrations: Vec<RootId>,
    active: bool,
}
impl FrameValues {
    /// # Safety
    /// All referenced frame owners must remain at these addresses and may not
    /// be mutated concurrently with this guard's terminal cleanup.
    pub unsafe fn new(
        stack: &Vec<Value>,
        vars: &Vec<Value>,
        locals: &Vec<Value>,
        sequence: &SequenceState,
        registrations: Vec<RootId>,
    ) -> Self {
        Self {
            stack: NonNull::from(stack),
            vars: NonNull::from(vars),
            locals: NonNull::from(locals),
            sequence: NonNull::from(sequence),
            registrations,
            active: true,
        }
    }
    fn finish(&mut self, outputs: &[Value]) {
        if !self.active {
            return;
        }
        self.active = false;
        // SAFETY: construction requires all frame owners to outlive this guard,
        // and the caller has already dropped the suspended instruction future.
        let mut collector = AllocationCollector::default();
        let mut result = Ok(());
        let mut visit = |value: &Value| {
            if result.is_ok() {
                result = collector.visit(value);
            }
        };
        for value in unsafe { self.stack.as_ref() }
            .iter()
            .chain(unsafe { self.vars.as_ref() })
            .chain(unsafe { self.locals.as_ref() })
        {
            visit(value);
        }
        unsafe { self.sequence.as_ref() }.visit_retained_values(&mut visit);
        let collected = result.and_then(|()| collector.finish());
        match collected {
            Ok(owned) if owned.is_empty() => {}
            Ok(owned) => {
                let mut keep = HashSet::new();
                if outputs
                    .iter()
                    .all(|value| retain_value(value, &mut keep).is_some())
                {
                    release(owned, &self.registrations, &keep);
                } else {
                    log::warn!("VM completed-output aliases are unavailable");
                }
            }
            Err(reason) => log::warn!("VM frame ownership unavailable: {reason}"),
        }
    }
    pub fn complete(&mut self, outputs: &[Value]) {
        self.finish(outputs);
    }
}
impl Drop for FrameValues {
    fn drop(&mut self) {
        self.finish(&[]);
    }
}
