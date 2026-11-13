use runmat_value::{NumericScalar, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;

use super::error;

pub(super) fn values(value: &Value, context: &str) -> BuiltinResult<Vec<NumericScalar>> {
    match value {
        Value::Num(value) => Ok(vec![NumericScalar::F64(*value)]),
        Value::Int(value) => Ok(vec![NumericScalar::from(value.clone())]),
        Value::Bool(value) => Ok(vec![NumericScalar::F64(f64::from(u8::from(*value)))]),
        Value::Tensor(tensor) => (0..tensor_utils::tensor_element_len(tensor))
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| error::internal(format!("{context}: invalid numeric element")))
            })
            .collect(),
        Value::LogicalArray(array) => Ok(array
            .data
            .iter()
            .map(|flag| NumericScalar::F64(f64::from(u8::from(*flag != 0))))
            .collect()),
        Value::SparseTensor(sparse) => values(
            &Value::Tensor(sparse.to_dense().map_err(error::internal)?),
            context,
        ),
        other => Err(error::invalid(format!(
            "discretize: {context} must be real numeric or logical, got {other:?}"
        ))),
    }
}
