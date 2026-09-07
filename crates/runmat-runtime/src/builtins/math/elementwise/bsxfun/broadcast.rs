pub(super) struct BroadcastPlan {
    output_shape: Vec<usize>,
    left_shape: Vec<usize>,
    right_shape: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
    len: usize,
}

impl BroadcastPlan {
    pub(super) fn new(left_shape: &[usize], right_shape: &[usize]) -> crate::BuiltinResult<Self> {
        let rank = left_shape.len().max(right_shape.len());
        let left_shape = crate::builtins::common::broadcast::align_shape(left_shape, rank);
        let right_shape = crate::builtins::common::broadcast::align_shape(right_shape, rank);
        let mut output_shape = Vec::with_capacity(rank);
        for dimension in 0..rank {
            let left = left_shape[dimension];
            let right = right_shape[dimension];
            if left == right {
                output_shape.push(left);
            } else if left == 1 {
                output_shape.push(right);
            } else if right == 1 {
                output_shape.push(left);
            } else {
                return Err(super::error::size(format!(
                    "non-singleton dimension mismatch (dimension {}: {} vs {})",
                    dimension + 1,
                    left,
                    right
                )));
            }
        }
        let len = checked_element_count(&output_shape).map_err(super::error::size)?;
        let left_strides = checked_strides(&left_shape).map_err(super::error::size)?;
        let right_strides = checked_strides(&right_shape).map_err(super::error::size)?;
        Ok(Self {
            output_shape,
            left_shape,
            right_shape,
            left_strides,
            right_strides,
            len,
        })
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        (0..self.len).map(|index| {
            (
                index,
                input_index(
                    index,
                    &self.output_shape,
                    &self.left_shape,
                    &self.left_strides,
                ),
                input_index(
                    index,
                    &self.output_shape,
                    &self.right_shape,
                    &self.right_strides,
                ),
            )
        })
    }

    pub(super) fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
}

fn checked_element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().copied().try_fold(1usize, |count, extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| "output size exceeds platform limits".to_string())
    })
}

fn checked_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| "input size exceeds platform limits".to_string())?;
    }
    Ok(strides)
}

fn input_index(
    output_index: usize,
    output_shape: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
) -> usize {
    let mut remaining = output_index;
    let mut offset = 0usize;
    for dimension in 0..output_shape.len() {
        let coordinate = remaining % output_shape[dimension];
        remaining /= output_shape[dimension];
        if input_shape[dimension] > 1 {
            offset += coordinate * input_strides[dimension];
        }
    }
    offset
}
