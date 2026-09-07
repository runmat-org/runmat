use std::borrow::Cow;

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::super::broadcast::OperandBroadcast;
use super::super::error;
use super::super::operands::{BoundOperand, RescaleInput};
use super::range::ScaleInput;

pub(super) struct PreparedOperands<'a> {
    values: [Cow<'a, [f64]>; 5],
    broadcasts: [OperandBroadcast; 5],
}

impl<'a> PreparedOperands<'a> {
    pub(super) fn new(
        input: &'a RescaleInput,
        lower: &'a BoundOperand,
        upper: &'a BoundOperand,
        input_min: &'a BoundOperand,
        input_max: &'a BoundOperand,
        output_shape: &[usize],
    ) -> Self {
        let tensors = [
            &input.tensor,
            &lower.tensor,
            &upper.tensor,
            &input_min.tensor,
            &input_max.tensor,
        ];
        Self {
            values: tensors.map(tensor::tensor_values_f64_cow),
            broadcasts: tensors
                .map(|tensor| OperandBroadcast::new(&tensor.shape, output_shape.len())),
        }
    }

    pub(super) fn at(&self, linear: usize, shape: &[usize]) -> BuiltinResult<ScaleInput> {
        let mut values = [0.0; 5];
        for (index, output) in values.iter_mut().enumerate() {
            let offset = self.broadcasts[index].index(linear, shape);
            *output = *self.values[index]
                .get(offset)
                .ok_or_else(|| error::internal("broadcast offset exceeded operand storage"))?;
        }
        Ok(ScaleInput::from(values))
    }
}
