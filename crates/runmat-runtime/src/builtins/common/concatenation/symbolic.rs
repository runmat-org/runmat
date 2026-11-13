//! Symbolic concatenation conversion, independent of builtin routing.

use super::{concat_error, BuiltinResult};
use runmat_value::{SymbolicExpr, Value};

fn numeric_scalar_to_symbolic(value: &Value) -> Option<SymbolicExpr> {
    match value {
        Value::Symbolic(expr) => Some(expr.clone()),
        Value::Num(n) => Some(SymbolicExpr::constant(*n)),
        Value::Int(i) => Some(SymbolicExpr::constant(i.to_f64())),
        Value::Bool(flag) => Some(SymbolicExpr::constant(if *flag { 1.0 } else { 0.0 })),
        _ => None,
    }
}

fn normalize_symbolic_concat_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() == 1 && shape[0] != 1 {
        vec![1, shape[0]]
    } else if crate::builtins::common::shape::is_scalar_shape(shape) {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

#[derive(Clone)]
pub(crate) struct SymbolicConcatBlock {
    pub(crate) data: runmat_value::ExactArrayStorage<SymbolicExpr>,
    pub(crate) shape: Vec<usize>,
}

impl SymbolicConcatBlock {
    fn new(data: Vec<SymbolicExpr>, shape: Vec<usize>) -> BuiltinResult<Self> {
        Self::from_shared(
            runmat_value::ExactArrayStorage::shared_unbounded(data),
            shape,
        )
    }
    fn from_shared(
        data: runmat_value::ExactArrayStorage<SymbolicExpr>,
        shape: Vec<usize>,
    ) -> BuiltinResult<Self> {
        let expected_len = checked_shape_len(&shape)?;
        if data.len() != expected_len {
            return Err(concat_error(format!(
                "Symbolic array data length {} does not match shape {:?} (expected {} elements)",
                data.len(),
                shape,
                expected_len
            )));
        }
        Ok(Self { data, shape })
    }
}

fn checked_shape_len(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| concat_error("Symbolic array dimensions overflow"))
    })
}

pub(crate) fn symbolic_block_from_value(
    value: &Value,
) -> BuiltinResult<Option<SymbolicConcatBlock>> {
    match value {
        Value::SymbolicArray(array) => {
            let shape = normalize_symbolic_concat_shape(&array.shape);
            if shape[0] == 0 && shape[1] == 0 {
                return Ok(None);
            }
            SymbolicConcatBlock::from_shared(array.data_owner().clone(), shape).map(Some)
        }
        Value::Tensor(tensor) => {
            let shape = normalize_symbolic_concat_shape(tensor.shape());
            if shape[0] == 0 && shape[1] == 0 {
                return Ok(None);
            }
            let data = tensor
                .materialize_f64()
                .iter()
                .map(|value| SymbolicExpr::constant(*value))
                .collect();
            SymbolicConcatBlock::new(data, shape).map(Some)
        }
        Value::LogicalArray(array) => {
            let shape = normalize_symbolic_concat_shape(&array.shape);
            if shape[0] == 0 && shape[1] == 0 {
                return Ok(None);
            }
            let data = array
                .data
                .iter()
                .map(|value| SymbolicExpr::constant(if *value == 0 { 0.0 } else { 1.0 }))
                .collect();
            SymbolicConcatBlock::new(data, shape).map(Some)
        }
        _ => {
            if let Some(expr) = numeric_scalar_to_symbolic(value) {
                SymbolicConcatBlock::new(vec![expr], vec![1, 1]).map(Some)
            } else {
                Err(concat_error(format!(
                    "Cannot concatenate value of type {value:?} with symbolic array"
                )))
            }
        }
    }
}
