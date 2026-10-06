use std::collections::HashSet;
use std::rc::Rc;

use runmat_gc::{ExplicitRoot, GcError, GcHandle, Trace, Tracer};
use runmat_value::{Value, ValueSequence};

use super::ExecutionServiceError;

/// A completed executor-owned value sequence whose reachable managed handles
/// remain GC roots until the completion record is discarded.
#[derive(Clone, Debug)]
pub struct RootedValueSequence {
    value: ValueSequence,
    _roots: RootedValueHandles,
}

impl RootedValueSequence {
    pub fn new(value: ValueSequence) -> Result<Self, ExecutionServiceError> {
        let roots = RootedValueHandles::for_traced(&value).map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "could not root completed future output: {error}"
            ))
        })?;
        Ok(Self {
            value,
            _roots: roots,
        })
    }

    pub fn value(&self) -> &ValueSequence {
        &self.value
    }
}

/// Managed handles reachable from owned Values while an asynchronous owner
/// retains them. This roots identities, not cloned payloads or device buffers.
#[derive(Clone, Debug)]
pub(crate) struct RootedValueHandles {
    _roots: Option<Rc<Vec<ExplicitRoot>>>,
}

impl RootedValueHandles {
    pub(crate) fn for_values<'a>(values: impl Iterator<Item = &'a Value>) -> Result<Self, GcError> {
        let mut collector = HandleCollector::default();
        for value in values {
            value.trace(&mut collector);
        }
        Self::from_handles(collector)
    }

    fn for_traced(value: &impl Trace) -> Result<Self, GcError> {
        let mut collector = HandleCollector::default();
        value.trace(&mut collector);
        Self::from_handles(collector)
    }

    fn from_handles(collector: HandleCollector) -> Result<Self, GcError> {
        if collector.handles.is_empty() {
            return Ok(Self { _roots: None });
        }
        let mut roots = Vec::with_capacity(collector.handles.len());
        for handle in collector.handles {
            roots.push(runmat_gc::gc_root(handle)?);
        }
        Ok(Self {
            _roots: Some(Rc::new(roots)),
        })
    }
}

#[derive(Default)]
struct HandleCollector {
    seen: HashSet<GcHandle>,
    handles: Vec<GcHandle>,
}

impl Tracer for HandleCollector {
    fn mark(&mut self, handle: GcHandle) {
        if self.seen.insert(handle) {
            self.handles.push(handle);
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use runmat_value::{HandleRef, Value};

    #[test]
    fn unmanaged_values_need_no_shared_root_owner() {
        let values = [Value::Num(7.0), Value::Bool(false)];
        let roots = RootedValueHandles::for_values(values.iter()).expect("no managed handles");
        assert!(roots._roots.is_none());
        assert!(roots.clone()._roots.is_none());
    }

    #[test]
    fn completed_sequence_roots_nested_handle_values_until_discarded() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("retained".into()))
                .expect("allocate rooted payload");
            let handle = initial.handle();
            let sequence = ValueSequence::comma_separated(vec![Value::HandleObject(HandleRef {
                class_name: "CompletionHandle".into(),
                target: handle,
                valid: true,
            })])
            .expect("valid completion sequence");
            let rooted = RootedValueSequence::new(sequence).expect("root completion");
            initial.unroot().expect("transfer root to completion");

            runmat_gc::gc_collect_major().expect("collect with live completion");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("completion remains rooted"),
                Value::String("retained".into())
            );

            drop(rooted);
            runmat_gc::gc_collect_major().expect("collect after completion discard");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}
