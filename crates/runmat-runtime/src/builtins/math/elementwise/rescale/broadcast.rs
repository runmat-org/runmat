use crate::builtins::common::broadcast::{align_shape, compute_strides, BroadcastPlan};
use crate::BuiltinResult;

use super::error;

pub(super) struct OperandBroadcast {
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl OperandBroadcast {
    pub fn new(shape: &[usize], rank: usize) -> Self {
        let shape = align_shape(shape, rank);
        let strides = compute_strides(&shape);
        Self { shape, strides }
    }

    pub fn index(&self, mut linear: usize, output_shape: &[usize]) -> usize {
        let mut offset = 0usize;
        for (dimension, &output_extent) in output_shape.iter().enumerate() {
            let coordinate = if output_extent == 0 {
                0
            } else {
                linear % output_extent
            };
            if output_extent != 0 {
                linear /= output_extent;
            }
            let mapped = if self.shape[dimension] <= 1 {
                0
            } else {
                coordinate
            };
            offset += mapped * self.strides[dimension];
        }
        offset
    }
}

pub(super) fn output_shape(input: &[usize], bounds: &[&[usize]]) -> BuiltinResult<Vec<usize>> {
    let mut output = input.to_vec();
    for shape in bounds {
        let plan =
            BroadcastPlan::new(&output, shape).map_err(|failure| error::size_mismatch(&failure))?;
        output = plan.output_shape().to_vec();
    }
    Ok(output)
}

pub(super) fn element_count(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count.checked_mul(dimension).ok_or_else(|| {
            error::internal(format!(
                "output shape {shape:?} exceeds supported element count"
            ))
        })
    })
}
