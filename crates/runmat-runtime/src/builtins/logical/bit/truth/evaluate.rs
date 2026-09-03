use runmat_builtins::LogicalBinaryOperator;

use crate::builtins::common::broadcast::{broadcast_index, compute_strides};
use crate::builtins::common::tensor;

use super::operand::LogicalBuffer;

pub(super) fn binary(
    left: &LogicalBuffer,
    right: &LogicalBuffer,
    shape: &[usize],
    operation: LogicalBinaryOperator,
) -> Vec<u8> {
    let total = tensor::element_count(shape);
    let left_strides = compute_strides(&left.shape);
    let right_strides = compute_strides(&right.shape);
    (0..total)
        .map(|linear| {
            let left = expanded_bit(left, &left_strides, shape, linear);
            let right = expanded_bit(right, &right_strides, shape, linear);
            u8::from(match operation {
                LogicalBinaryOperator::And => left && right,
                LogicalBinaryOperator::Or => left || right,
                LogicalBinaryOperator::Xor => left ^ right,
            })
        })
        .collect()
}

fn expanded_bit(
    buffer: &LogicalBuffer,
    strides: &[usize],
    output_shape: &[usize],
    linear: usize,
) -> bool {
    if buffer.data.is_empty() {
        return false;
    }
    let index = broadcast_index(linear, output_shape, &buffer.shape, strides);
    buffer.data.get(index).is_some_and(|bit| *bit != 0)
}
