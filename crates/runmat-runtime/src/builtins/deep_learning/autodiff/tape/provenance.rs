//! Identity of a dlarray's Data at the moment its tape node was recorded.
//!
//! Dense numeric values already share the node's pinned host allocation.
//! Scalars and logical arrays need their source identity retained because
//! converting them to a numerical tensor creates a different allocation.

use runmat_value::{IntValue, LogicalArray, NumericDType, Value};

use super::NodeValue;
use super::DLARRAY_CLASS;

pub(super) enum DataProvenance {
    Dense,
    Number(u64),
    Integer(IntValue),
    Boolean(bool),
    Logical(LogicalArray),
    Resident(runmat_accelerate_api::GpuHandleIdentity),
}

impl DataProvenance {
    pub(super) fn from_data(value: &Value) -> Option<Self> {
        match effective_data(value)? {
            Value::Tensor(_) => Some(Self::Dense),
            Value::Num(value) => Some(Self::Number(value.to_bits())),
            Value::Int(value) => Some(Self::Integer(value.clone())),
            Value::Bool(value) => Some(Self::Boolean(*value)),
            Value::LogicalArray(value) => Some(Self::Logical(value.clone())),
            Value::GpuTensor(value) => Some(Self::Resident(value.view_identity())),
            _ => None,
        }
    }

    pub(super) fn matches(&self, value: &Value, node: &NodeValue) -> bool {
        let Some(value) = effective_data(value) else {
            return false;
        };
        match (self, value) {
            (Self::Dense, Value::Tensor(current)) if node.host().is_some() => {
                let node = node.host().expect("host provenance guards host node");
                current.shape() == node.shape()
                    && current.numeric_dtype() == node.numeric_dtype()
                    && current
                        .host_buffer()
                        .shares_allocation_with(node.host_buffer())
            }
            (Self::Number(bits), Value::Num(current)) => {
                current.to_bits() == *bits
                    && node.shape() == [1, 1]
                    && node.numeric_dtype() == Some(NumericDType::F64)
            }
            (Self::Integer(original), Value::Int(current)) => {
                current == original && node.shape() == [1, 1]
            }
            (Self::Boolean(original), Value::Bool(current)) => {
                current == original
                    && node.shape() == [1, 1]
                    && node.numeric_dtype() == Some(NumericDType::F64)
            }
            (Self::Logical(original), Value::LogicalArray(current)) => {
                current.shape == original.shape
                    && current.shape.as_slice() == node.shape()
                    && node.numeric_dtype() == Some(NumericDType::F64)
                    && current.data.shares_allocation_with(&original.data)
            }
            (Self::Resident(identity), Value::GpuTensor(current)) => {
                matches!(node, NodeValue::Resident(recorded) if recorded.view_identity() == *identity)
                    && current.view_identity() == *identity
                    && current.shape() == node.shape()
            }
            _ => false,
        }
    }
}

fn effective_data(mut value: &Value) -> Option<&Value> {
    while let Value::Object(object) = value {
        if !object.class_name().is(DLARRAY_CLASS) {
            return None;
        }
        value = object.properties().get("Data")?;
    }
    Some(value)
}
