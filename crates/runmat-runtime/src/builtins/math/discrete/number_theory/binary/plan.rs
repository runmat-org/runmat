use super::{binary_error, BinaryContext, BinaryInput};
use crate::BuiltinResult;

pub(in crate::builtins::math::discrete) struct SameSizeOrScalarPlan {
    pub(in crate::builtins::math::discrete) output_shape: Vec<usize>,
    len: usize,
    left_scalar: bool,
    right_scalar: bool,
}

impl SameSizeOrScalarPlan {
    pub(in crate::builtins::math::discrete) fn new(
        left: &BinaryInput,
        right: &BinaryInput,
        context: &'static BinaryContext,
    ) -> BuiltinResult<Self> {
        let left_scalar = left.is_scalar();
        let right_scalar = right.is_scalar();
        let output_shape = if left.shape == right.shape {
            left.shape.clone()
        } else if left_scalar {
            right.shape.clone()
        } else if right_scalar {
            left.shape.clone()
        } else {
            return Err(binary_error(
                context,
                context.size_mismatch,
                "inputs must be the same size or one input must be scalar",
            ));
        };
        Ok(Self {
            len: element_count(&output_shape),
            output_shape,
            left_scalar,
            right_scalar,
        })
    }

    pub(in crate::builtins::math::discrete) fn len(&self) -> usize {
        self.len
    }

    pub(in crate::builtins::math::discrete) fn iter(
        &self,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.len).map(|index| {
            (
                if self.left_scalar { 0 } else { index },
                if self.right_scalar { 0 } else { index },
            )
        })
    }
}

pub(super) fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}
