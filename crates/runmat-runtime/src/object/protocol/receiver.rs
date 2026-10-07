use runmat_gc_api::{Trace, Tracer};
use runmat_value::Value;

use super::{ObjectAccessContext, ObjectProtocol, ProtocolResolution};
use crate::object::indexing::ObjectSubscriptPath;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

#[derive(Clone, Debug)]
pub struct PreparedSubscriptReceiver {
    receiver: Value,
    end: ProtocolResolution,
}

impl PreparedSubscriptReceiver {
    pub fn receiver(&self) -> &Value {
        &self.receiver
    }
}

impl Trace for PreparedSubscriptReceiver {
    fn trace(&self, tracer: &mut dyn Tracer) {
        self.receiver.trace(tracer);
    }
}

pub async fn prepare_subscript_receiver(
    root: Value,
    materialized_prefix: Option<ObjectSubscriptPath>,
    access: ObjectAccessContext,
) -> Result<PreparedSubscriptReceiver, RuntimeError> {
    prepare_subscript_receiver_with_context(None, root, materialized_prefix, access).await
}

pub async fn prepare_subscript_receiver_in_context(
    runtime: &crate::context::RuntimeContext,
    root: Value,
    materialized_prefix: Option<ObjectSubscriptPath>,
    access: ObjectAccessContext,
) -> Result<PreparedSubscriptReceiver, RuntimeError> {
    prepare_subscript_receiver_with_context(Some(runtime), root, materialized_prefix, access).await
}

async fn prepare_subscript_receiver_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    root: Value,
    materialized_prefix: Option<ObjectSubscriptPath>,
    access: ObjectAccessContext,
) -> Result<PreparedSubscriptReceiver, RuntimeError> {
    let work = prepare_subscript_receiver_in_scope(runtime, root, materialized_prefix, access);
    super::context::scope(runtime, work).await
}

async fn prepare_subscript_receiver_in_scope(
    runtime: Option<&crate::context::RuntimeContext>,
    root: Value,
    materialized_prefix: Option<ObjectSubscriptPath>,
    access: ObjectAccessContext,
) -> Result<PreparedSubscriptReceiver, RuntimeError> {
    let receiver = match materialized_prefix {
        Some(prefix) => match runtime {
            Some(runtime) => {
                super::read_subscript_path_with_access_in_context(
                    runtime,
                    root,
                    prefix,
                    access.clone(),
                )
                .await?
            }
            None => super::read_subscript_path_with_access(root, prefix, access.clone()).await?,
        },
        None => root,
    };
    let end = super::resolve_object_protocol(&receiver, ObjectProtocol::End, &access)?;
    Ok(PreparedSubscriptReceiver { receiver, end })
}

pub async fn resolve_subscript_end(
    prepared: &PreparedSubscriptReceiver,
    component_index: usize,
    component_count: usize,
) -> Result<Value, RuntimeError> {
    resolve_subscript_end_with_context(None, prepared, component_index, component_count).await
}

pub async fn resolve_subscript_end_in_context(
    runtime: &crate::context::RuntimeContext,
    prepared: &PreparedSubscriptReceiver,
    component_index: usize,
    component_count: usize,
) -> Result<Value, RuntimeError> {
    resolve_subscript_end_with_context(Some(runtime), prepared, component_index, component_count)
        .await
}

async fn resolve_subscript_end_with_context(
    runtime: Option<&crate::context::RuntimeContext>,
    prepared: &PreparedSubscriptReceiver,
    component_index: usize,
    component_count: usize,
) -> Result<Value, RuntimeError> {
    let work = resolve_subscript_end_in_scope(runtime, prepared, component_index, component_count);
    super::context::scope(runtime, work).await
}

async fn resolve_subscript_end_in_scope(
    runtime: Option<&crate::context::RuntimeContext>,
    prepared: &PreparedSubscriptReceiver,
    component_index: usize,
    component_count: usize,
) -> Result<Value, RuntimeError> {
    if let ProtocolResolution::Method(method) = &prepared.end {
        return super::invoke_prepared_object_end_with_context(
            runtime,
            method,
            prepared.receiver.clone(),
            component_index,
            component_count,
        )
        .await;
    }
    let shape = value_shape(&prepared.receiver);
    let extent = crate::indexing::shape::selector_extent(&shape, component_count, component_index)?;
    exact_double(extent)
}

fn value_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(value) => value.shape().to_vec(),
        Value::ComplexTensor(value) => value.shape().to_vec(),
        Value::SparseTensor(value) => value.shape(),
        Value::GpuTensor(value) => value.shape().to_vec(),
        Value::StringArray(value) => value.shape.clone(),
        Value::LogicalArray(value) => value.shape.clone(),
        Value::CharArray(value) => value.shape().to_vec(),
        Value::Cell(value) => value.shape.clone(),
        Value::StructArray(value) => value.shape().to_vec(),
        Value::ObjectArray(value) => value.shape().to_vec(),
        Value::SymbolicArray(value) => value.shape.clone(),
        _ => vec![1, 1],
    }
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
            assert_eq!(value_shape(&real), shape);
            assert_eq!(value_shape(&complex), shape);
        }
    }
}
