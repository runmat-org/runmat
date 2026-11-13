use crate::interpreter::vars::InterpreterVars;
use runmat_gc::{GcHandle, GcRoot, Trace, Tracer, ValueRoot};
use runmat_value::Value;
use std::ptr::NonNull;

pub(super) struct InterpreterVarsRoot(NonNull<InterpreterVars>);

impl InterpreterVarsRoot {
    /// # Safety
    /// The vars owner must remain pinned until this root is unregistered and
    /// scans must not overlap mutation of the owner.
    pub unsafe fn new(vars: &InterpreterVars) -> Self {
        Self(NonNull::from(vars))
    }

    fn visit(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for value in unsafe { self.0.as_ref() }.iter() {
            visitor(&value)?;
        }
        Some(())
    }
}

impl ValueRoot for InterpreterVarsRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        self.visit(visitor)
    }
}

impl GcRoot for InterpreterVarsRoot {
    fn scan(&self) -> Vec<GcHandle> {
        struct Collector(Vec<GcHandle>);
        impl Tracer for Collector {
            fn mark(&mut self, handle: GcHandle) {
                self.0.push(handle);
            }
        }
        let mut collector = Collector(Vec::new());
        let _ = self.visit(&mut |value| {
            value.trace(&mut collector);
            Some(())
        });
        collector.0
    }

    fn description(&self) -> String {
        "interpreter_vars".into()
    }
}
