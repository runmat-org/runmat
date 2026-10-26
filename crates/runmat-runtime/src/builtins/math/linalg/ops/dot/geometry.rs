//! Same real-dot row geometry for actual host arithmetic and model routing.
use super::*;
pub(in crate::builtins::math) struct DotGeometry {
    output: Vec<usize>,
    count: usize,
    factor: usize,
    stride: usize,
    elementwise: bool,
}
impl DotGeometry {
    pub(super) fn new(
        left: &[usize],
        right: &[usize],
        dim: Option<usize>,
    ) -> DotKernelResult<Self> {
        let count = |shape: &[usize]| {
            if shape.contains(&0) {
                return Ok(0);
            }
            shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| DotKernelError::internal("dot geometry exceeds addressable size"))
        };
        if dim.is_none() && is_vector_shape(left) && is_vector_shape(right) {
            let factor = count(left)?;
            if factor != count(right)? {
                return Err(DotKernelError::invalid_input());
            }
            return Ok(Self {
                output: vec![1, 1],
                count: 1,
                factor,
                stride: 1,
                elementwise: false,
            });
        }
        if left != right {
            return Err(DotKernelError::invalid_input());
        }
        let axis = dim.unwrap_or_else(|| default_dimension(left)) - 1;
        let mut output = left.to_vec();
        let (factor, stride) = if axis < left.len() {
            output[axis] = 1;
            (left[axis], count(&left[..axis])?)
        } else {
            (1, count(left)?)
        };
        Ok(Self {
            count: count(&output)?,
            output,
            factor,
            stride,
            elementwise: axis >= left.len(),
        })
    }
    pub(super) fn elementwise(&self) -> bool {
        self.elementwise
    }
    pub(in crate::builtins::math) fn output_shape(&self) -> &[usize] {
        &self.output
    }
    pub(in crate::builtins::math) fn outputs(&self) -> usize {
        self.count
    }
    pub(in crate::builtins::math) fn factor(&self) -> usize {
        self.factor
    }
    pub(in crate::builtins::math) fn source(&self, row: usize, axis: usize) -> Option<usize> {
        if row >= self.count || axis >= self.factor || self.stride == 0 {
            return None;
        }
        row.checked_rem(self.stride)?
            .checked_add(axis.checked_mul(self.stride)?)?
            .checked_add(
                (row / self.stride)
                    .checked_mul(self.stride)?
                    .checked_mul(self.factor)?,
            )
    }
    pub(in crate::builtins::math) fn output(&self, source: usize) -> Option<usize> {
        if self.factor == 0 || self.stride == 0 {
            return None;
        }
        let block = self.stride.checked_mul(self.factor)?;
        let row = source % self.stride + (source / block) * self.stride;
        (row < self.count).then_some(row)
    }
}
pub(in crate::builtins::math) async fn dot_geometry(
    left: &[usize],
    right: &[usize],
    arguments: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<DotGeometry> {
    if arguments.len() > 1 {
        return Err(dot_invalid_argument("dot: too many input arguments"));
    }
    let dim = match arguments.first() {
        Some(value) => Some(parse_dimension_arg(value).await?),
        None => None,
    };
    let left = if left.is_empty() { &[0, 0][..] } else { left };
    let right = if right.is_empty() { &[0, 0][..] } else { right };
    DotGeometry::new(left, right, dim).map_err(Into::into)
}
