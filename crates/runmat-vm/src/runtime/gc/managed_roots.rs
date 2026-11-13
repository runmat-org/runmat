//! Compact GC roots for values owned outside the pinned interpreter frame.
//! GPU allocations are retained by the values themselves, not by these roots.
use runmat_gc::{GcHandle, GcRoot, RootId, Trace, Tracer};
use runmat_value::Value;
use std::collections::HashSet;

#[cfg(test)]
mod tests;

pub(crate) struct ManagedRoots(Option<RootId>);

impl ManagedRoots {
    pub(crate) fn new(values: &[Value]) -> Result<Self, String> {
        Self::from_values(values.iter())
    }

    pub(crate) fn from_values<'a>(
        values: impl IntoIterator<Item = &'a Value>,
    ) -> Result<Self, String> {
        struct Handles(HashSet<GcHandle>);
        impl Tracer for Handles {
            fn mark(&mut self, handle: GcHandle) {
                self.0.insert(handle);
            }
        }
        let mut handles = Handles(HashSet::new());
        for value in values {
            value.trace(&mut handles);
        }
        if handles.0.is_empty() {
            return Ok(Self(None));
        }
        struct Root(Vec<GcHandle>);
        impl GcRoot for Root {
            fn scan(&self) -> Vec<GcHandle> {
                self.0.clone()
            }
            fn description(&self) -> String {
                "vm_owned_managed_values".into()
            }
        }
        runmat_gc::gc_register_root(Box::new(Root(handles.0.into_iter().collect())))
            .map(|id| Self(Some(id)))
            .map_err(|error| error.to_string())
    }

    pub(crate) fn from_traced(value: &impl Trace) -> Result<Self, String> {
        struct Handles(HashSet<GcHandle>);
        impl Tracer for Handles {
            fn mark(&mut self, handle: GcHandle) {
                self.0.insert(handle);
            }
        }
        let mut handles = Handles(HashSet::new());
        value.trace(&mut handles);
        if handles.0.is_empty() {
            return Ok(Self(None));
        }
        struct Root(Vec<GcHandle>);
        impl GcRoot for Root {
            fn scan(&self) -> Vec<GcHandle> {
                self.0.clone()
            }
            fn description(&self) -> String {
                "vm_owned_managed_values".into()
            }
        }
        runmat_gc::gc_register_root(Box::new(Root(handles.0.into_iter().collect())))
            .map(|id| Self(Some(id)))
            .map_err(|error| error.to_string())
    }
}

impl Drop for ManagedRoots {
    fn drop(&mut self) {
        if let Some(id) = self.0.take() {
            if let Err(error) = runmat_gc::gc_unregister_root(id) {
                log::warn!("VM managed root removal failed: {error}");
            }
        }
    }
}
