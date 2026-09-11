use std::collections::HashSet;
use std::rc::Rc;

use runmat_gc::{ExplicitRoot, GcHandle, Trace, Tracer};
use runmat_value::ValueSequence;

use super::ExecutionServiceError;

/// A completed executor-owned value sequence whose reachable managed handles
/// remain GC roots until the completion record is discarded.
#[derive(Clone, Debug)]
pub struct RootedValueSequence {
    value: ValueSequence,
    _roots: Rc<Vec<ExplicitRoot>>,
}

impl RootedValueSequence {
    pub fn new(value: ValueSequence) -> Result<Self, ExecutionServiceError> {
        let mut collector = HandleCollector::default();
        value.trace(&mut collector);
        let mut roots = Vec::with_capacity(collector.handles.len());
        for handle in collector.handles {
            roots.push(runmat_gc::gc_root(handle).map_err(|error| {
                ExecutionServiceError::Infrastructure(format!(
                    "could not root completed future output: {error}"
                ))
            })?);
        }
        Ok(Self {
            value,
            _roots: Rc::new(roots),
        })
    }

    pub fn value(&self) -> &ValueSequence {
        &self.value
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
