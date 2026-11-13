use std::future::Future;

use runmat_gc_api::{Trace, Tracer};
use runmat_value::Value;

use super::{ObjectAccessContext, ObjectProtocol, ProtocolResolution};
use crate::execution::RootedValue;
use crate::object::indexing::ObjectSubscriptPath;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

#[derive(Clone, Debug)]
pub struct PreparedSubscriptReceiver {
    receiver: RootedValue,
    end: ProtocolResolution,
}

impl PreparedSubscriptReceiver {
    pub fn receiver(&self) -> &Value {
        &self.receiver
    }
}

impl Trace for PreparedSubscriptReceiver {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.receiver.value().trace(tracer);
    }
}

pub fn prepare_subscript_receiver(
    root: Value,
    materialized_prefix: Option<ObjectSubscriptPath>,
    access: ObjectAccessContext,
) -> impl Future<Output = Result<PreparedSubscriptReceiver, RuntimeError>> {
    let runtime = crate::context::legacy::active();
    async move {
        prepare_subscript_receiver_with_context(runtime.as_ref(), root, materialized_prefix, access)
            .await
    }
}

pub fn prepare_subscript_receiver_in_context<'runtime>(
    runtime: &'runtime crate::context::RuntimeContext,
    root: Value,
    materialized_prefix: Option<ObjectSubscriptPath>,
    access: ObjectAccessContext,
) -> impl Future<Output = Result<PreparedSubscriptReceiver, RuntimeError>> + use<'runtime> {
    prepare_subscript_receiver_with_context(Some(runtime), root, materialized_prefix, access)
}

/// Borrow the prepared destination's proof through the canonical path owner.
pub(crate) fn prepare_subscript_receiver_with_borrowed_access<'a>(
    runtime: &'a crate::context::RuntimeContext,
    root: Value,
    materialized_prefix: Option<ObjectSubscriptPath>,
    access: &'a ObjectAccessContext,
) -> impl Future<Output = Result<PreparedSubscriptReceiver, RuntimeError>> + 'a {
    super::context::scope(
        Some(runtime),
        prepare_subscript_receiver_in_scope(Some(runtime), root, materialized_prefix, access),
    )
}

fn prepare_subscript_receiver_with_context<'runtime>(
    runtime: Option<&'runtime crate::context::RuntimeContext>,
    root: Value,
    materialized_prefix: Option<ObjectSubscriptPath>,
    access: ObjectAccessContext,
) -> impl Future<Output = Result<PreparedSubscriptReceiver, RuntimeError>> + use<'runtime> {
    let work = async move {
        prepare_subscript_receiver_in_scope(runtime, root, materialized_prefix, &access).await
    };
    super::context::scope(runtime, work)
}

async fn prepare_subscript_receiver_in_scope(
    runtime: Option<&crate::context::RuntimeContext>,
    root: Value,
    materialized_prefix: Option<ObjectSubscriptPath>,
    access: &ObjectAccessContext,
) -> Result<PreparedSubscriptReceiver, RuntimeError> {
    let _root = match runtime {
        Some(context) => crate::execution::RootedValueLease::for_values_iterator_in_context(
            std::iter::once(&root),
            context,
        )?,
        None => crate::execution::RootedValueLease::for_values(std::iter::once(&root))
            .map_err(|error| error.into_runtime_error())?,
    };
    let receiver = match materialized_prefix {
        Some(prefix) => {
            super::path_execution::read_single_with_borrowed_access(runtime, root, prefix, access)
                .await?
        }
        None => RootedValue::in_optional_context(root, runtime)?,
    };
    let end = super::resolve_object_protocol(&receiver, ObjectProtocol::End, access)?;
    Ok(PreparedSubscriptReceiver { receiver, end })
}

pub fn resolve_subscript_end<'prepared>(
    prepared: &'prepared PreparedSubscriptReceiver,
    component_index: usize,
    component_count: usize,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'prepared> {
    let runtime = crate::context::legacy::active();
    async move {
        resolve_subscript_end_with_context(
            runtime.as_ref(),
            prepared,
            component_index,
            component_count,
        )
        .await
    }
}

pub fn resolve_subscript_end_in_context<'runtime, 'prepared>(
    runtime: &'runtime crate::context::RuntimeContext,
    prepared: &'prepared PreparedSubscriptReceiver,
    component_index: usize,
    component_count: usize,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'runtime, 'prepared> {
    resolve_subscript_end_with_context(Some(runtime), prepared, component_index, component_count)
}

fn resolve_subscript_end_with_context<'runtime, 'prepared>(
    runtime: Option<&'runtime crate::context::RuntimeContext>,
    prepared: &'prepared PreparedSubscriptReceiver,
    component_index: usize,
    component_count: usize,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + use<'runtime, 'prepared> {
    let work = resolve_subscript_end_in_scope(runtime, prepared, component_index, component_count);
    super::context::scope(runtime, work)
}

async fn resolve_subscript_end_in_scope(
    runtime: Option<&crate::context::RuntimeContext>,
    prepared: &PreparedSubscriptReceiver,
    component_index: usize,
    component_count: usize,
) -> Result<RootedValue, RuntimeError> {
    if let Some(method) = &prepared.end {
        return super::invoke_prepared_object_end_with_context(
            runtime,
            method,
            prepared.receiver.value().clone(),
            component_index,
            component_count,
        )
        .await;
    }
    let shape = crate::builtins::common::shape::stored_dimensions_borrowed(&prepared.receiver);
    let extent = crate::indexing::shape::selector_extent(&shape, component_count, component_index)?;
    RootedValue::new(exact_double(extent)?).map_err(|error| error.into_runtime_error())
}

fn exact_double(value: usize) -> Result<Value, RuntimeError> {
    let value = u64::try_from(value)
        .map_err(|_| semantic_error("ObjectEndIndexOverflow", "object end extent is too large"))?;
    if value > (1u64 << 53) {
        return Err(semantic_error(
            "ObjectEndIndexOverflow",
            "object end extent exceeds exact double range",
        ));
    }
    Ok(Value::Num(value as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn prepared_receiver_keeps_nested_handles_until_retirement() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("receiver".into()))
                .expect("allocate receiver payload");
            let handle = initial.handle();
            let source = Value::Cell(
                runmat_value::CellArray::new_with_shape(
                    vec![Value::HandleObject(runmat_value::HandleRef {
                        class_name: "PreparedReceiverPayload".into(),
                        target: handle,
                        valid: true,
                    })],
                    vec![1, 1],
                )
                .expect("receiver container"),
            );
            let prepared = pollster::block_on(prepare_subscript_receiver(
                source,
                None,
                ObjectAccessContext::default(),
            ))
            .expect("prepare receiver");
            initial.unroot().expect("transfer to prepared receiver");
            runmat_gc::gc_collect_major().expect("collect with live receiver");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("receiver keeps payload"),
                Value::String("receiver".into())
            );
            let end = pollster::block_on(resolve_subscript_end(&prepared, 0, 1))
                .expect("resolve container extent");
            assert_eq!(end.value(), &Value::Num(1.0));
            drop(end);
            drop(prepared);
            runmat_gc::gc_collect_major().expect("collect retired receiver");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn prepared_receiver_shape_keeps_low_rank_numeric_metadata() {
        for shape in [Vec::new(), vec![2]] {
            let count = shape.iter().product::<usize>();
            let real =
                Value::Tensor(runmat_value::Tensor::new(vec![1.0; count], shape.clone()).unwrap());
            let complex = Value::ComplexTensor(
                runmat_value::ComplexTensor::new(vec![(1.0, 2.0); count], shape.clone()).unwrap(),
            );
            assert_eq!(
                crate::builtins::common::shape::stored_dimensions_borrowed(&real).as_slice(),
                shape
            );
            assert_eq!(
                crate::builtins::common::shape::stored_dimensions_borrowed(&complex).as_slice(),
                shape
            );
        }
    }
}
