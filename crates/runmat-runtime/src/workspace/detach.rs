//! Exact synchronous retirement of current-workspace bindings before callbacks.

use std::collections::HashSet;

use runmat_execution::resource::ExactShared;
use runmat_gc::{GcHandle, GcRoot, Trace, Tracer, ValueRoot};
use runmat_value::Value;

use super::BaseWorkspaceError;

/// Select bindings from the *current* workspace, never a session-wide fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceDetachSelection {
    All,
    Names(Vec<String>),
    Except(Vec<String>),
}

impl WorkspaceDetachSelection {
    pub fn select(&self, entries: Vec<(String, Value)>) -> Vec<(String, Value)> {
        match self {
            Self::All => entries,
            Self::Names(names) => {
                let mut available = entries
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>();
                names
                    .iter()
                    .filter_map(|name| available.remove(name).map(|value| (name.clone(), value)))
                    .collect()
            }
            Self::Except(excluded) => {
                let excluded: HashSet<&str> = excluded.iter().map(String::as_str).collect();
                entries
                    .into_iter()
                    .filter(|(name, _)| !excluded.contains(name.as_str()))
                    .collect()
            }
        }
    }
}

/// Removed values plus every managed child root. This owner is created before
/// mutation and retained across any awaited cleanup callback.
#[derive(Debug)]
pub struct DetachedWorkspaceValues {
    // Unregister while `entries` still owns the values. Releasing the scanner's
    // shared owner must never run a final Foreign destructor under its lock.
    root: runmat_gc::ValueRootRegistration,
    entries: ExactShared<Vec<(String, Value)>>,
}

impl DetachedWorkspaceValues {
    pub fn new(entries: Vec<(String, Value)>) -> Result<Self, BaseWorkspaceError> {
        let entries = ExactShared::new(entries);
        let root = runmat_gc::ValueRootRegistration::new(Box::new(DetachedWorkspaceRoot(
            entries.clone(),
        )))?;
        Ok(Self { root, entries })
    }

    pub fn entries(&self) -> &[(String, Value)] {
        self.entries.get()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.get().is_empty()
    }

    /// Only a reachability audit that also retains this owner may exclude its
    /// registration; another root for the same handle remains visible.
    pub fn root_registration(&self) -> runmat_gc::RootId {
        self.root.id()
    }
}

/// GC scanning and retained-resource inspection borrow the same physical
/// entries returned to cleanup callers, without cloning their Value trees.
struct DetachedWorkspaceRoot(ExactShared<Vec<(String, Value)>>);

impl GcRoot for DetachedWorkspaceRoot {
    fn scan(&self) -> Vec<GcHandle> {
        struct Handles(Vec<GcHandle>);
        impl Tracer for Handles {
            fn mark(&mut self, handle: GcHandle) {
                self.0.push(handle);
            }
        }
        let mut handles = Handles(Vec::new());
        for (_, value) in self.0.get() {
            value.trace(&mut handles);
        }
        handles.0
    }

    fn description(&self) -> String {
        "detached workspace values".into()
    }

    fn estimated_size(&self) -> usize {
        std::mem::size_of_val(self.0.get().as_slice())
    }
}

impl ValueRoot for DetachedWorkspaceRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for (_, value) in self.0.get() {
            visitor(value)?;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn detached_root_projects_its_original_values_and_retires_with_owner() {
        runmat_gc::gc_test_context(|| {
            let allocation = runmat_gc::gc_allocate_rooted(Value::Num(73.0)).unwrap();
            let handle = allocation.handle();
            let detached = DetachedWorkspaceValues::new(vec![(
                "retained".into(),
                Value::HandleObject(runmat_value::HandleRef {
                    class_name: "DetachedRootFixture".into(),
                    target: handle,
                    valid: true,
                }),
            )])
            .unwrap();
            allocation.unroot().unwrap();
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());

            let expected = &detached.entries()[0].1;
            let mut found_original = false;
            runmat_gc::gc_visit_retained_values(&[], &mut |value| {
                found_original |= std::ptr::eq(value, expected);
                Some(())
            })
            .expect("detached values have a complete retained-root projection");
            assert!(
                found_original,
                "root projection must not visit a Value copy"
            );
            drop(detached);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
