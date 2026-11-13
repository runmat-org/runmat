//! Difference geometry shared with actual numeric stage indexing.
use super::*;
pub(in crate::builtins::math) struct DifferenceGeometry {
    input: Vec<usize>,
    axis: usize,
    order: usize,
    stride: usize,
}
impl DifferenceGeometry {
    pub(super) fn new(shape: &[usize], order: usize, dim: Option<usize>) -> BuiltinResult<Self> {
        let axis = dim
            .unwrap_or_else(|| default_dimension(shape))
            .saturating_sub(1);
        let mut input = shape.to_vec();
        while input.len() <= axis {
            input.push(1);
        }
        let stride = product(&input[..axis]);
        Ok(Self {
            input,
            axis,
            order,
            stride,
        })
    }
    pub(in crate::builtins::math) fn stages(&self) -> usize {
        self.order.min(self.input[self.axis])
    }
    pub(in crate::builtins::math) fn output_shape(&self) -> Vec<usize> {
        self.shape_after(self.order)
    }
    pub(in crate::builtins::math) fn shape_after(&self, steps: usize) -> Vec<usize> {
        let mut shape = self.input.clone();
        shape[self.axis] = shape[self.axis].saturating_sub(steps);
        shape
    }
    pub(in crate::builtins::math) fn pair(
        &self,
        step: usize,
        row: usize,
    ) -> Option<(usize, usize)> {
        let extent = self.input[self.axis].checked_sub(step + 1)?;
        if extent == 0 || self.stride == 0 {
            return None;
        }
        let block = self.stride.checked_mul(extent)?;
        let first = row.checked_add((row / block).checked_mul(self.stride)?)?;
        Some((first, first.checked_add(self.stride)?))
    }
}
pub(in crate::builtins::math) fn difference_geometry(
    shape: &[usize],
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<DifferenceGeometry> {
    if args.iter().any(is_empty_array) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DIFF_EMPTY_CONTROL_EXTENSION,
            NAME,
        )?;
    }
    let (order, dim) = parse_arguments(args)?;
    if order == 0 {
        crate::compatibility::ensure_builtin_extension_enabled(&DIFF_ZERO_ORDER_EXTENSION, NAME)?;
    }
    DifferenceGeometry::new(shape, order, dim)
}
