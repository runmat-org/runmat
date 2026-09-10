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
    let receiver = match materialized_prefix {
        Some(prefix) => {
            super::read_subscript_path_with_access(root, prefix, access.clone()).await?
        }
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
    if let ProtocolResolution::Method(method) = &prepared.end {
        return super::invoke_prepared_object_end(
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
        Value::Tensor(value) => value.shape.clone(),
        Value::ComplexTensor(value) => value.shape.clone(),
        Value::SparseTensor(value) => value.shape(),
        Value::GpuTensor(value) => value.shape.clone(),
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
