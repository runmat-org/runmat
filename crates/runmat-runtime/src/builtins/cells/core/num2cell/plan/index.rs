use super::super::error;

pub(super) fn source_index(
    output_shape: &[usize],
    slice_shape: &[usize],
    source_shape: &[usize],
    output_count: usize,
    slice_count: usize,
    output_linear: usize,
    slice_linear: usize,
    mut slice_destination: impl FnMut(usize) -> Option<usize>,
) -> crate::BuiltinResult<usize> {
    if output_linear >= output_count || slice_linear >= slice_count {
        return Err(error::internal(
            "partition index exceeds the planned extent",
        ));
    }
    let mut output_suffix = output_count;
    let mut source_stride = 1usize;
    let mut source_linear = 0usize;
    for axis in 0..source_shape.len() {
        let output_extent = output_shape[axis];
        output_suffix /= output_extent;
        let output_coord = (output_linear / output_suffix) % output_extent;
        let coordinate = if let Some(destination) = slice_destination(axis) {
            let slice_prefix = slice_shape[..destination]
                .iter()
                .try_fold(1usize, |stride, extent| stride.checked_mul(*extent))
                .ok_or_else(|| error::internal("partition stride exceeds platform limits"))?;
            (slice_linear / slice_prefix) % slice_shape[destination]
        } else {
            output_coord
        };
        source_linear = source_linear
            .checked_add(
                coordinate
                    .checked_mul(source_stride)
                    .ok_or_else(|| error::internal("partition index overflow"))?,
            )
            .ok_or_else(|| error::internal("partition index overflow"))?;
        source_stride = source_stride
            .checked_mul(source_shape[axis])
            .ok_or_else(|| error::internal("partition stride exceeds platform limits"))?;
    }
    Ok(source_linear)
}

pub(super) fn row_major_coords(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        if shape[axis] != 0 {
            coords[axis] = linear % shape[axis];
            linear /= shape[axis];
        }
    }
    coords
}

pub(in crate::builtins::cells::core::num2cell) fn column_major_coords(
    mut linear: usize,
    shape: &[usize],
) -> Vec<usize> {
    shape
        .iter()
        .map(|extent| {
            let coordinate = if *extent == 0 { 0 } else { linear % *extent };
            if *extent != 0 {
                linear /= *extent;
            }
            coordinate
        })
        .collect()
}

pub(super) fn column_major_linear(
    coords: &[usize],
    shape: &[usize],
) -> crate::BuiltinResult<usize> {
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (&coordinate, &extent) in coords.iter().zip(shape) {
        linear = linear
            .checked_add(
                coordinate
                    .checked_mul(stride)
                    .ok_or_else(|| error::internal("index overflow"))?,
            )
            .ok_or_else(|| error::internal("index overflow"))?;
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| error::internal("index overflow"))?;
    }
    Ok(linear)
}
