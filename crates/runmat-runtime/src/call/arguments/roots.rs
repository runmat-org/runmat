//! Managed liveness for values owned by a suspended argument expansion.
//! Physical resources remain owned by Values; this registration contains only
//! compact GC identities, never clones of array payloads or device handles.
use crate::RuntimeError;
use runmat_gc_api::{GcHandle, GcRoot, RootId, Trace, Tracer};
use runmat_value::Value;
use std::{cell::RefCell, collections::HashSet, rc::Rc};

#[derive(Default)]
struct Handles {
    seen: HashSet<GcHandle>,
    values: Vec<GcHandle>,
}
impl Tracer for Handles {
    fn mark(&mut self, handle: GcHandle) {
        if self.seen.insert(handle) {
            self.values.push(handle);
        }
    }
}
struct Registered(Rc<RefCell<Handles>>);
impl GcRoot for Registered {
    fn scan(&self) -> Vec<GcHandle> {
        self.0.borrow().values.clone()
    }
    fn description(&self) -> String {
        "runtime argument expansion".into()
    }
}
impl runmat_gc::ValueRoot for Registered {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for handle in &self.0.try_borrow().ok()?.values {
            runmat_gc::gc_with_value(handle, |value| visitor(value)).ok()??;
        }
        Some(())
    }
}

pub(crate) struct ExpansionRoots {
    registration: RootId,
    handles: Rc<RefCell<Handles>>,
}
impl ExpansionRoots {
    pub(crate) fn new() -> Result<Self, RuntimeError> {
        let handles = Rc::new(RefCell::new(Handles::default()));
        let registration = runmat_gc::gc_register_value_root(Box::new(Registered(handles.clone())))
            .map_err(|error| {
                crate::build_runtime_error(format!("cannot root argument expansion: {error}"))
                    .build()
            })?;
        Ok(Self {
            registration,
            handles,
        })
    }
    pub(crate) fn retain_iter<'a>(&self, values: impl IntoIterator<Item = &'a Value>) {
        let mut handles = self.handles.borrow_mut();
        for value in values {
            value.trace(&mut *handles);
        }
    }
}
impl Drop for ExpansionRoots {
    fn drop(&mut self) {
        if let Err(error) = runmat_gc::gc_unregister_root(self.registration) {
            log::warn!("could not unregister argument expansion roots: {error}");
        }
    }
}

#[cfg(test)]
mod tests;
