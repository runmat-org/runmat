//! Symbolic assembly shares the checked N-D column-major concatenation owner.

use crate::builtins::common::concatenation::symbolic::symbolic_block_from_value;
use crate::BuiltinResult;
use runmat_value::{SymbolicArray, Value};

pub(crate) fn concatenate<'a>(
    dim_zero: usize,
    values: impl Iterator<Item = &'a Value>,
) -> BuiltinResult<Value> {
    let mut blocks = Vec::new();
    for value in values {
        if let Some(block) = symbolic_block_from_value(value)? {
            blocks.push(block);
        }
    }
    if blocks.is_empty() {
        return SymbolicArray::new(Vec::new(), vec![0, 0])
            .map(Value::SymbolicArray)
            .map_err(super::cat_err);
    }
    let shapes: Vec<_> = blocks.iter().map(|block| block.shape.as_slice()).collect();
    let data: Vec<_> = blocks.iter().map(|block| block.data.as_slice()).collect();
    let rank = shapes
        .iter()
        .map(|shape| shape.len())
        .max()
        .unwrap_or(2)
        .max(dim_zero + 1);
    // Only [] is neutral for the existing symbolic conversion. Shaped empty
    // arrays still participate in non-concatenated dimension validation.
    let (data, mut shape) =
        super::concat_column_major_with_neutral(dim_zero, &shapes, &data, "cat", |shape| {
            shape[0] == 0 && shape[1] == 0
        })?;
    // Symbolic arrays preserve explicitly authored trailing dimensions.
    shape.resize(rank, 1);
    SymbolicArray::new(data, shape)
        .map(Value::SymbolicArray)
        .map_err(super::cat_err)
}
