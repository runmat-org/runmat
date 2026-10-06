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

/// Owned values whose reachable managed handles remain roots while an async
/// operation retains the batch. This is the canonical input counterpart to
/// [`RootedValueSequence`].
#[derive(Clone, Debug)]
pub struct RootedValues {
    values: Vec<Value>,
    _roots: RootedValueHandles,
}

/// Incremental owner for values decoded by a potentially allocating codec.
/// Every value is rooted before decoding or allocating the next member.
#[derive(Debug, Default)]
pub struct RootedValuesBuilder {
    values: Vec<Value>,
    member_roots: Vec<RootedValueHandles>,
}

impl RootedValuesBuilder {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            member_roots: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, value: Value) -> Result<(), ExecutionServiceError> {
        let roots = RootedValueHandles::for_traced(&value).map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "could not root retained future input: {error}"
            ))
        })?;
        self.values.push(value);
        self.member_roots.push(roots);
        Ok(())
    }

    pub fn finish(self) -> Result<RootedValues, ExecutionServiceError> {
        let roots = RootedValueHandles::for_values(self.values.iter()).map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "could not root retained future inputs: {error}"
            ))
        })?;
        let values = RootedValues {
            values: self.values,
            _roots: roots,
        };
        drop(self.member_roots);
        Ok(values)
    }
}

impl RootedValues {
    pub fn new(values: Vec<Value>) -> Result<Self, ExecutionServiceError> {
        let roots = RootedValueHandles::for_values(values.iter()).map_err(|error| {
            ExecutionServiceError::Infrastructure(format!(
                "could not root retained future inputs: {error}"
            ))
        })?;
        Ok(Self {
            values,
            _roots: roots,
        })
    }

    pub fn values(&self) -> &[Value] {
        &self.values
    }

    pub fn into_values(self) -> Vec<Value> {
        self.values
    }
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

    pub(crate) fn for_traced(value: &impl Trace) -> Result<Self, GcError> {
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

    #[test]
    fn retained_input_values_root_nested_handles_until_discarded() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("input".into()))
                .expect("allocate rooted payload");
            let handle = initial.handle();
            let values = RootedValues::new(vec![Value::HandleObject(HandleRef {
                class_name: "InputHandle".into(),
                target: handle,
                valid: true,
            })])
            .expect("root retained inputs");
            initial.unroot().expect("transfer root to retained inputs");

            runmat_gc::gc_collect_major().expect("collect with live inputs");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("input remains rooted"),
                Value::String("input".into())
            );

            drop(values);
            runmat_gc::gc_collect_major().expect("collect after retained input discard");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn incremental_inputs_survive_collection_between_decoded_members() {
        runmat_gc::gc_test_context(|| {
            let first = runmat_gc::gc_allocate_rooted(Value::String("first".into()))
                .expect("allocate first payload");
            let first_handle = first.handle();
            let mut builder = RootedValuesBuilder::with_capacity(2);
            builder
                .push(Value::HandleObject(HandleRef {
                    class_name: "DecodedHandle".into(),
                    target: first_handle,
                    valid: true,
                }))
                .expect("root first decoded member");
            first.unroot().expect("transfer first root");
            runmat_gc::gc_collect_major().expect("collect between decoded members");
            assert_eq!(
                runmat_gc::gc_clone_value(&first_handle).expect("first member remains rooted"),
                Value::String("first".into())
            );

            let second = runmat_gc::gc_allocate_rooted(Value::String("second".into()))
                .expect("allocate second payload");
            let second_handle = second.handle();
            builder
                .push(Value::HandleObject(HandleRef {
                    class_name: "DecodedHandle".into(),
                    target: second_handle,
                    valid: true,
                }))
                .expect("root second decoded member");
            second.unroot().expect("transfer second root");
            let values = builder.finish().expect("finish rooted batch");

            runmat_gc::gc_collect_major().expect("collect completed batch");
            assert!(runmat_gc::gc_clone_value(&first_handle).is_ok());
            assert!(runmat_gc::gc_clone_value(&second_handle).is_ok());
            drop(values);
            runmat_gc::gc_collect_major().expect("collect after batch drop");
            assert!(runmat_gc::gc_clone_value(&first_handle).is_err());
            assert!(runmat_gc::gc_clone_value(&second_handle).is_err());
        });
    }
}
