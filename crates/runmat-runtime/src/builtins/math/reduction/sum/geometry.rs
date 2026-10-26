//! Metadata-only access to sum's same public grammar and index arithmetic.
//! No numerical tensor or provider operand is synthesized.
use super::*;
pub(in crate::builtins::math) struct ReductionGeometry {
    input: Vec<usize>,
    output: Vec<usize>,
    mask: Vec<bool>,
    coordinates: Vec<usize>,
    output_coordinates: Vec<usize>,
    count: usize,
    factor: usize,
}
impl ReductionGeometry {
    pub(in crate::builtins::math) fn output_shape(&self) -> &[usize] {
        &self.output
    }
    pub(in crate::builtins::math) fn is_identity(&self) -> bool {
        self.input == self.output
    }
    pub(in crate::builtins::math) fn factor(&self) -> usize {
        self.factor
    }
    pub(in crate::builtins::math) fn rank(&self) -> usize {
        self.input.len()
    }
    /// Inverse of project: one original source row in this output's fiber.
    pub(in crate::builtins::math) fn source(&self, output: usize, ordinal: usize) -> Option<usize> {
        if ordinal >= self.factor || self.count == 0 {
            return None;
        }
        let mut output = output;
        let mut reduced = ordinal;
        let mut source = 0usize;
        let mut stride = 1usize;
        for axis in 0..self.input.len() {
            let extent = self.input[axis];
            let coordinate = if self.mask[axis] {
                let value = reduced % extent;
                reduced /= extent;
                value
            } else {
                let value = output % extent;
                output /= extent;
                value
            };
            source = source.checked_add(coordinate.checked_mul(stride)?)?;
            stride = stride.checked_mul(extent)?;
        }
        (output == 0 && reduced == 0 && source < self.count).then_some(source)
    }
    pub(in crate::builtins::math) fn project(&mut self, input: usize) -> Option<usize> {
        if input >= self.count {
            return None;
        }
        linear_to_multi(input, &self.input, &mut self.coordinates);
        for (axis, coordinate) in self.coordinates.iter().enumerate() {
            self.output_coordinates[axis] = if self.mask[axis] { 0 } else { *coordinate };
        }
        Some(multi_to_linear(&self.output_coordinates, &self.output))
    }
}
pub(in crate::builtins::math) async fn reduction_geometry(
    shape: &[usize],
    arguments: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<ReductionGeometry> {
    crate::context::host_allocation::require_unbounded_workspace(
        runtime,
        "sum reduction geometry",
        "finite metadata-only reduction planning is not certified",
    )?;
    crate::context::RuntimeScopeFactory::new(runtime.cloned(), move || async move {
        let parsed = parse_arguments(arguments).await?;
        let dims = resolve_dims(shape, &parsed.selection)?;
        reduction_geometry_for_axes(shape, &dims.dims_in_bounds)
    })
    .construct()
    .await
}

pub(in crate::builtins::math) fn reduction_geometry_for_axes(
    shape: &[usize],
    axes: &[usize],
) -> BuiltinResult<ReductionGeometry> {
    let input = if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    };
    let count = if input.contains(&0) {
        0
    } else {
        input
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| sum_invalid_input("reduction geometry exceeds addressable size"))?
    };
    let mut output = input.clone();
    let mut mask = vec![false; input.len()];
    for &axis in axes {
        output[axis] = 1;
        mask[axis] = true;
    }
    let out_count = if output.contains(&0) {
        0
    } else {
        output
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| sum_invalid_input("reduction output exceeds addressable size"))?
    };
    let factor = if out_count == 0 || axes.iter().any(|&axis| input[axis] == 0) {
        0
    } else {
        axes.iter()
            .try_fold(1usize, |n, &axis| n.checked_mul(input[axis]))
            .ok_or_else(|| sum_invalid_input("reduction factor exceeds addressable size"))?
    };
    let coordinates = vec![0usize; input.len()];
    let output_coordinates = vec![0usize; input.len()];
    Ok(ReductionGeometry {
        input,
        output,
        mask,
        coordinates,
        output_coordinates,
        count,
        factor,
    })
}
