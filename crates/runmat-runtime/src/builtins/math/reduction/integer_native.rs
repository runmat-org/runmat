//! Exact native integer reductions shared by MATLAB reduction builtins.

use runmat_builtins::{IntValue, IntegerStorage, Tensor, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtremaDirection {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExtremaComparison {
    Natural,
    Absolute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CumulativeOperation {
    Sum,
    Product,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CumulativeDirection {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CumulativeExtremaDirection {
    Min,
    Max,
}

pub(crate) struct IntegerExtrema {
    pub(crate) values: Value,
    pub(crate) indices: Value,
}

/// Reduces an integer tensor by saturated native addition along zero-based
/// dimensions. Callers select this only for MATLAB's explicit `"native"`
/// output mode; default reductions intentionally retain their double output.
pub(crate) fn sum(
    storage: &IntegerStorage,
    shape: &[usize],
    reduced_dims: &[usize],
) -> Result<Value, String> {
    if reduced_dims.is_empty() {
        return integer_storage_into_value(storage.clone(), shape.to_vec());
    }

    let shape = normalized_shape(shape);
    let mut output_shape = shape.clone();
    for &dim in reduced_dims {
        if dim < output_shape.len() {
            output_shape[dim] = 1;
        }
    }
    let output_len = element_count(&output_shape);
    let mut coords = vec![0usize; shape.len()];
    let mut output_coords = vec![0usize; shape.len()];
    let mut reduced = vec![false; shape.len()];
    for &dim in reduced_dims {
        if dim < reduced.len() {
            reduced[dim] = true;
        }
    }

    macro_rules! reduce {
        ($values:expr, $variant:ident, $zero:expr) => {{
            let mut output = vec![$zero; output_len];
            for (linear, &value) in $values.iter().enumerate() {
                linear_to_multi(linear, &shape, &mut coords);
                for (dim, &coord) in coords.iter().enumerate() {
                    output_coords[dim] = if reduced[dim] { 0 } else { coord };
                }
                let output_index = multi_to_linear(&output_coords, &output_shape);
                output[output_index] = output[output_index].saturating_add(value);
            }
            IntegerStorage::$variant(output)
        }};
    }

    let output = match storage {
        IntegerStorage::I8(values) => reduce!(values, I8, 0i8),
        IntegerStorage::I16(values) => reduce!(values, I16, 0i16),
        IntegerStorage::I32(values) => reduce!(values, I32, 0i32),
        IntegerStorage::I64(values) => reduce!(values, I64, 0i64),
        IntegerStorage::U8(values) => reduce!(values, U8, 0u8),
        IntegerStorage::U16(values) => reduce!(values, U16, 0u16),
        IntegerStorage::U32(values) => reduce!(values, U32, 0u32),
        IntegerStorage::U64(values) => reduce!(values, U64, 0u64),
    };
    integer_storage_into_value(output, output_shape)
}

/// Reduces an integer tensor by saturated native multiplication along
/// zero-based dimensions for MATLAB's explicit `"native"` output mode.
pub(crate) fn product(
    storage: &IntegerStorage,
    shape: &[usize],
    reduced_dims: &[usize],
) -> Result<Value, String> {
    if reduced_dims.is_empty() {
        return integer_storage_into_value(storage.clone(), shape.to_vec());
    }

    let shape = normalized_shape(shape);
    let mut output_shape = shape.clone();
    for &dim in reduced_dims {
        if dim < output_shape.len() {
            output_shape[dim] = 1;
        }
    }
    let output_len = element_count(&output_shape);
    let mut coords = vec![0usize; shape.len()];
    let mut output_coords = vec![0usize; shape.len()];
    let mut reduced = vec![false; shape.len()];
    for &dim in reduced_dims {
        if dim < reduced.len() {
            reduced[dim] = true;
        }
    }

    macro_rules! reduce {
        ($values:expr, $variant:ident, $one:expr) => {{
            let mut output = vec![$one; output_len];
            for (linear, &value) in $values.iter().enumerate() {
                linear_to_multi(linear, &shape, &mut coords);
                for (dim, &coord) in coords.iter().enumerate() {
                    output_coords[dim] = if reduced[dim] { 0 } else { coord };
                }
                let output_index = multi_to_linear(&output_coords, &output_shape);
                output[output_index] = output[output_index].saturating_mul(value);
            }
            IntegerStorage::$variant(output)
        }};
    }

    let output = match storage {
        IntegerStorage::I8(values) => reduce!(values, I8, 1i8),
        IntegerStorage::I16(values) => reduce!(values, I16, 1i16),
        IntegerStorage::I32(values) => reduce!(values, I32, 1i32),
        IntegerStorage::I64(values) => reduce!(values, I64, 1i64),
        IntegerStorage::U8(values) => reduce!(values, U8, 1u8),
        IntegerStorage::U16(values) => reduce!(values, U16, 1u16),
        IntegerStorage::U32(values) => reduce!(values, U32, 1u32),
        IntegerStorage::U64(values) => reduce!(values, U64, 1u64),
    };
    integer_storage_into_value(output, output_shape)
}

/// Reduces an integer tensor by an exact mean in the source integer class.
///
/// MATLAB's `mean(A, "native")` performs the operation in the input class.
/// Accumulating in the native class would overflow before the division, so the
/// calculation uses a widened exact accumulator and converts only the final,
/// rounded mean back to the source class. Half values round away from zero.
pub(crate) fn mean(
    storage: &IntegerStorage,
    shape: &[usize],
    reduced_dims: &[usize],
) -> Result<Value, String> {
    if reduced_dims.is_empty() {
        return integer_storage_into_value(storage.clone(), shape.to_vec());
    }

    let shape = normalized_shape(shape);
    let mut output_shape = shape.clone();
    for &dim in reduced_dims {
        if dim < output_shape.len() {
            output_shape[dim] = 1;
        }
    }
    let output_len = element_count(&output_shape);
    let mut coords = vec![0usize; shape.len()];
    let mut output_coords = vec![0usize; shape.len()];
    let mut reduced = vec![false; shape.len()];
    for &dim in reduced_dims {
        if dim < reduced.len() {
            reduced[dim] = true;
        }
    }

    macro_rules! reduce_signed {
        ($values:expr, $variant:ident, $ty:ty) => {{
            let mut sums = vec![0i128; output_len];
            let mut counts = vec![0usize; output_len];
            for (linear, &value) in $values.iter().enumerate() {
                linear_to_multi(linear, &shape, &mut coords);
                for (dim, &coord) in coords.iter().enumerate() {
                    output_coords[dim] = if reduced[dim] { 0 } else { coord };
                }
                let output_index = multi_to_linear(&output_coords, &output_shape);
                sums[output_index] += value as i128;
                counts[output_index] += 1;
            }
            IntegerStorage::$variant(
                sums.into_iter()
                    .zip(counts)
                    .map(|(sum, count)| rounded_signed_mean(sum, count) as $ty)
                    .collect(),
            )
        }};
    }

    macro_rules! reduce_unsigned {
        ($values:expr, $variant:ident, $ty:ty) => {{
            let mut sums = vec![0u128; output_len];
            let mut counts = vec![0usize; output_len];
            for (linear, &value) in $values.iter().enumerate() {
                linear_to_multi(linear, &shape, &mut coords);
                for (dim, &coord) in coords.iter().enumerate() {
                    output_coords[dim] = if reduced[dim] { 0 } else { coord };
                }
                let output_index = multi_to_linear(&output_coords, &output_shape);
                sums[output_index] += value as u128;
                counts[output_index] += 1;
            }
            IntegerStorage::$variant(
                sums.into_iter()
                    .zip(counts)
                    .map(|(sum, count)| rounded_unsigned_mean(sum, count) as $ty)
                    .collect(),
            )
        }};
    }

    let output = match storage {
        IntegerStorage::I8(values) => reduce_signed!(values, I8, i8),
        IntegerStorage::I16(values) => reduce_signed!(values, I16, i16),
        IntegerStorage::I32(values) => reduce_signed!(values, I32, i32),
        IntegerStorage::I64(values) => reduce_signed!(values, I64, i64),
        IntegerStorage::U8(values) => reduce_unsigned!(values, U8, u8),
        IntegerStorage::U16(values) => reduce_unsigned!(values, U16, u16),
        IntegerStorage::U32(values) => reduce_unsigned!(values, U32, u32),
        IntegerStorage::U64(values) => reduce_unsigned!(values, U64, u64),
    };
    integer_storage_into_value(output, output_shape)
}

fn rounded_signed_mean(sum: i128, count: usize) -> i128 {
    if count == 0 {
        return 0;
    }
    let divisor = count as i128;
    let quotient = sum / divisor;
    let remainder = sum % divisor;
    if remainder.unsigned_abs() * 2 >= count as u128 {
        quotient + remainder.signum()
    } else {
        quotient
    }
}

fn rounded_unsigned_mean(sum: u128, count: usize) -> u128 {
    if count == 0 {
        return 0;
    }
    let divisor = count as u128;
    let quotient = sum / divisor;
    let remainder = sum % divisor;
    if remainder * 2 >= divisor {
        quotient + 1
    } else {
        quotient
    }
}

/// Reduces native integer storage without reading its lossy floating-point
/// compatibility view. The caller owns MATLAB argument parsing and supplies
/// the already-resolved dimension plan so `min` and `max` retain identical
/// shape and index semantics.
pub(crate) fn extrema(
    storage: &IntegerStorage,
    shape: &[usize],
    output_shape: Vec<usize>,
    reduced_dims: &[usize],
    dims_mask: &[bool],
    reduce_strides: &[usize],
    reduce_all: bool,
    linear_index: bool,
    direction: ExtremaDirection,
    comparison: ExtremaComparison,
) -> Result<IntegerExtrema, String> {
    let output_len = element_count(&output_shape);
    if storage.is_empty() || output_len == 0 {
        return Err("integer extrema requires a non-empty reduction input".to_string());
    }

    let shape = normalized_shape(shape);
    let output_strides = strides(&output_shape);
    let mut best = vec![0usize; output_len];
    let mut has_value = vec![false; output_len];
    let mut coords = vec![0usize; shape.len()];

    for linear in 0..storage.len() {
        let output_index = output_index(&coords, &output_strides, dims_mask);
        let candidate = storage_value(storage, linear);
        if !has_value[output_index]
            || should_replace(
                &storage_value(storage, best[output_index]),
                &candidate,
                direction,
                comparison,
            )
        {
            best[output_index] = linear;
            has_value[output_index] = true;
        }
        increment_coords(&mut coords, &shape);
    }

    let values = selected_storage(storage, &best);
    let mut indices = Vec::with_capacity(output_len);
    for &full_index in &best {
        let index = if linear_index || reduce_all {
            full_index
        } else if reduced_dims.is_empty() {
            0
        } else {
            reduction_index(full_index, &shape, reduced_dims, reduce_strides)
        };
        indices.push((index + 1) as f64);
    }

    Ok(IntegerExtrema {
        values: integer_storage_into_value(values, output_shape.clone())?,
        indices: numeric_tensor_into_value(indices, output_shape)?,
    })
}

pub(crate) fn empty_like(storage: &IntegerStorage, shape: Vec<usize>) -> Result<Value, String> {
    integer_storage_into_value(empty_storage_like(storage), shape)
}

/// Performs a MATLAB cumulative reduction in the input integer class. Integer
/// `cumsum` and `cumprod` preserve their class, unlike scalar reductions whose
/// default output is double.
pub(crate) fn cumulative(
    storage: &IntegerStorage,
    shape: &[usize],
    dim: usize,
    direction: CumulativeDirection,
    operation: CumulativeOperation,
) -> Result<Value, String> {
    if dim == 0 {
        return Err("cumulative integer reduction dimension must be >= 1".to_string());
    }
    let shape = normalized_shape(shape);
    if storage.is_empty() || dim > shape.len() {
        return integer_storage_into_value(storage.clone(), shape);
    }

    let dim_index = dim - 1;
    let segment_len = shape[dim_index];
    if segment_len == 0 {
        return integer_storage_into_value(storage.clone(), shape);
    }
    let stride_before = element_count(&shape[..dim_index]);
    let stride_after = element_count(&shape[dim..]);
    let block = stride_before.saturating_mul(segment_len);

    macro_rules! scan {
        ($values:expr, $variant:ident, $identity:expr, $method:ident) => {{
            let mut output = vec![$identity; $values.len()];
            for after in 0..stride_after {
                let base = after * block;
                for before in 0..stride_before {
                    let mut accumulator = $identity;
                    match direction {
                        CumulativeDirection::Forward => {
                            for offset in 0..segment_len {
                                let index = base + before + offset * stride_before;
                                accumulator = accumulator.$method($values[index]);
                                output[index] = accumulator;
                            }
                        }
                        CumulativeDirection::Reverse => {
                            for offset in (0..segment_len).rev() {
                                let index = base + before + offset * stride_before;
                                accumulator = accumulator.$method($values[index]);
                                output[index] = accumulator;
                            }
                        }
                    }
                }
            }
            IntegerStorage::$variant(output)
        }};
    }

    let output = match (storage, operation) {
        (IntegerStorage::I8(values), CumulativeOperation::Sum) => {
            scan!(values, I8, 0i8, saturating_add)
        }
        (IntegerStorage::I16(values), CumulativeOperation::Sum) => {
            scan!(values, I16, 0i16, saturating_add)
        }
        (IntegerStorage::I32(values), CumulativeOperation::Sum) => {
            scan!(values, I32, 0i32, saturating_add)
        }
        (IntegerStorage::I64(values), CumulativeOperation::Sum) => {
            scan!(values, I64, 0i64, saturating_add)
        }
        (IntegerStorage::U8(values), CumulativeOperation::Sum) => {
            scan!(values, U8, 0u8, saturating_add)
        }
        (IntegerStorage::U16(values), CumulativeOperation::Sum) => {
            scan!(values, U16, 0u16, saturating_add)
        }
        (IntegerStorage::U32(values), CumulativeOperation::Sum) => {
            scan!(values, U32, 0u32, saturating_add)
        }
        (IntegerStorage::U64(values), CumulativeOperation::Sum) => {
            scan!(values, U64, 0u64, saturating_add)
        }
        (IntegerStorage::I8(values), CumulativeOperation::Product) => {
            scan!(values, I8, 1i8, saturating_mul)
        }
        (IntegerStorage::I16(values), CumulativeOperation::Product) => {
            scan!(values, I16, 1i16, saturating_mul)
        }
        (IntegerStorage::I32(values), CumulativeOperation::Product) => {
            scan!(values, I32, 1i32, saturating_mul)
        }
        (IntegerStorage::I64(values), CumulativeOperation::Product) => {
            scan!(values, I64, 1i64, saturating_mul)
        }
        (IntegerStorage::U8(values), CumulativeOperation::Product) => {
            scan!(values, U8, 1u8, saturating_mul)
        }
        (IntegerStorage::U16(values), CumulativeOperation::Product) => {
            scan!(values, U16, 1u16, saturating_mul)
        }
        (IntegerStorage::U32(values), CumulativeOperation::Product) => {
            scan!(values, U32, 1u32, saturating_mul)
        }
        (IntegerStorage::U64(values), CumulativeOperation::Product) => {
            scan!(values, U64, 1u64, saturating_mul)
        }
    };
    integer_storage_into_value(output, shape)
}

/// Computes cumulative extrema in exact integer storage, with MATLAB's
/// one-based position indices along the scanned dimension.
pub(crate) fn cumulative_extrema(
    storage: &IntegerStorage,
    shape: &[usize],
    dim: usize,
    direction: CumulativeDirection,
    extrema: CumulativeExtremaDirection,
) -> Result<IntegerExtrema, String> {
    if dim == 0 {
        return Err("cumulative integer extrema dimension must be >= 1".to_string());
    }
    let shape = normalized_shape(shape);
    if storage.is_empty() {
        return Ok(IntegerExtrema {
            values: integer_storage_into_value(storage.clone(), shape.clone())?,
            indices: numeric_tensor_into_value(Vec::new(), shape)?,
        });
    }
    if dim > shape.len() {
        return Ok(IntegerExtrema {
            values: integer_storage_into_value(storage.clone(), shape.clone())?,
            indices: numeric_tensor_into_value(vec![1.0; storage.len()], shape)?,
        });
    }

    let dim_index = dim - 1;
    let segment_len = shape[dim_index];
    let stride_before = element_count(&shape[..dim_index]);
    let stride_after = element_count(&shape[dim..]);
    let block = stride_before.saturating_mul(segment_len);
    let mut selected = vec![0usize; storage.len()];
    let mut indices = vec![0.0f64; storage.len()];

    for after in 0..stride_after {
        let base = after * block;
        for before in 0..stride_before {
            let mut current = 0usize;
            let mut has_value = false;
            macro_rules! scan {
                ($offsets:expr) => {
                    for offset in $offsets {
                        let index = base + before + offset * stride_before;
                        if !has_value
                            || should_replace(
                                &storage_value(storage, current),
                                &storage_value(storage, index),
                                match extrema {
                                    CumulativeExtremaDirection::Min => ExtremaDirection::Min,
                                    CumulativeExtremaDirection::Max => ExtremaDirection::Max,
                                },
                                ExtremaComparison::Natural,
                            )
                        {
                            current = index;
                            has_value = true;
                        }
                        selected[index] = current;
                        let position = (current - base - before) / stride_before + 1;
                        indices[index] = position as f64;
                    }
                };
            }
            match direction {
                CumulativeDirection::Forward => scan!(0..segment_len),
                CumulativeDirection::Reverse => scan!((0..segment_len).rev()),
            }
        }
    }
    Ok(IntegerExtrema {
        values: integer_storage_into_value(selected_storage(storage, &selected), shape.clone())?,
        indices: numeric_tensor_into_value(indices, shape)?,
    })
}

pub(crate) fn storage_from_scalar(value: &IntValue) -> IntegerStorage {
    match value {
        IntValue::I8(value) => IntegerStorage::I8(vec![*value]),
        IntValue::I16(value) => IntegerStorage::I16(vec![*value]),
        IntValue::I32(value) => IntegerStorage::I32(vec![*value]),
        IntValue::I64(value) => IntegerStorage::I64(vec![*value]),
        IntValue::U8(value) => IntegerStorage::U8(vec![*value]),
        IntValue::U16(value) => IntegerStorage::U16(vec![*value]),
        IntValue::U32(value) => IntegerStorage::U32(vec![*value]),
        IntValue::U64(value) => IntegerStorage::U64(vec![*value]),
    }
}

fn integer_storage_into_value(storage: IntegerStorage, shape: Vec<usize>) -> Result<Value, String> {
    if storage.len() == 1 {
        return Ok(Value::Int(storage_value(&storage, 0)));
    }
    Ok(Value::Tensor(Tensor::new_integer(storage, shape)?))
}

fn numeric_tensor_into_value(data: Vec<f64>, shape: Vec<usize>) -> Result<Value, String> {
    if data.len() == 1 {
        return Ok(Value::Num(data[0]));
    }
    Ok(Value::Tensor(Tensor::new(data, shape)?))
}

fn empty_storage_like(storage: &IntegerStorage) -> IntegerStorage {
    match storage {
        IntegerStorage::I8(_) => IntegerStorage::I8(Vec::new()),
        IntegerStorage::I16(_) => IntegerStorage::I16(Vec::new()),
        IntegerStorage::I32(_) => IntegerStorage::I32(Vec::new()),
        IntegerStorage::I64(_) => IntegerStorage::I64(Vec::new()),
        IntegerStorage::U8(_) => IntegerStorage::U8(Vec::new()),
        IntegerStorage::U16(_) => IntegerStorage::U16(Vec::new()),
        IntegerStorage::U32(_) => IntegerStorage::U32(Vec::new()),
        IntegerStorage::U64(_) => IntegerStorage::U64(Vec::new()),
    }
}

fn selected_storage(storage: &IntegerStorage, indices: &[usize]) -> IntegerStorage {
    macro_rules! select {
        ($values:expr, $variant:ident) => {
            IntegerStorage::$variant(indices.iter().map(|&index| $values[index]).collect())
        };
    }
    match storage {
        IntegerStorage::I8(values) => select!(values, I8),
        IntegerStorage::I16(values) => select!(values, I16),
        IntegerStorage::I32(values) => select!(values, I32),
        IntegerStorage::I64(values) => select!(values, I64),
        IntegerStorage::U8(values) => select!(values, U8),
        IntegerStorage::U16(values) => select!(values, U16),
        IntegerStorage::U32(values) => select!(values, U32),
        IntegerStorage::U64(values) => select!(values, U64),
    }
}

fn should_replace(
    current: &IntValue,
    candidate: &IntValue,
    direction: ExtremaDirection,
    comparison: ExtremaComparison,
) -> bool {
    let ordering = match comparison {
        ExtremaComparison::Natural => numeric_value(candidate).cmp(&numeric_value(current)),
        ExtremaComparison::Absolute => absolute_value(candidate)
            .cmp(&absolute_value(current))
            .then_with(|| numeric_value(candidate).cmp(&numeric_value(current))),
    };
    match direction {
        ExtremaDirection::Min => ordering.is_lt(),
        ExtremaDirection::Max => ordering.is_gt(),
    }
}

fn numeric_value(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => *value as i128,
        IntValue::I16(value) => *value as i128,
        IntValue::I32(value) => *value as i128,
        IntValue::I64(value) => *value as i128,
        IntValue::U8(value) => *value as i128,
        IntValue::U16(value) => *value as i128,
        IntValue::U32(value) => *value as i128,
        IntValue::U64(value) => *value as i128,
    }
}

fn absolute_value(value: &IntValue) -> u128 {
    let numeric = numeric_value(value);
    if numeric < 0 {
        (-numeric) as u128
    } else {
        numeric as u128
    }
}

fn storage_value(storage: &IntegerStorage, index: usize) -> IntValue {
    match storage {
        IntegerStorage::I8(values) => IntValue::I8(values[index]),
        IntegerStorage::I16(values) => IntValue::I16(values[index]),
        IntegerStorage::I32(values) => IntValue::I32(values[index]),
        IntegerStorage::I64(values) => IntValue::I64(values[index]),
        IntegerStorage::U8(values) => IntValue::U8(values[index]),
        IntegerStorage::U16(values) => IntValue::U16(values[index]),
        IntegerStorage::U32(values) => IntValue::U32(values[index]),
        IntegerStorage::U64(values) => IntValue::U64(values[index]),
    }
}

fn normalized_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn linear_to_multi(index: usize, shape: &[usize], output: &mut [usize]) {
    let mut remainder = index;
    for (dim, &size) in shape.iter().enumerate() {
        output[dim] = if size == 0 { 0 } else { remainder % size };
        if size != 0 {
            remainder /= size;
        }
    }
}

fn multi_to_linear(coords: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for (dim, &size) in shape.iter().enumerate() {
        index += coords[dim] * stride;
        stride *= size;
    }
    index
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dimension in shape {
        result.push(stride);
        stride = stride.saturating_mul(dimension.max(1));
    }
    result
}

fn output_index(coords: &[usize], output_strides: &[usize], dims_mask: &[bool]) -> usize {
    output_strides
        .iter()
        .enumerate()
        .map(|(dimension, &stride)| {
            if dims_mask.get(dimension).copied().unwrap_or(false) {
                0
            } else {
                coords[dimension] * stride
            }
        })
        .sum()
}

fn reduction_index(
    full_index: usize,
    shape: &[usize],
    reduced_dims: &[usize],
    reduce_strides: &[usize],
) -> usize {
    let mut coords = vec![0usize; shape.len()];
    linear_to_multi(full_index, shape, &mut coords);
    reduced_dims
        .iter()
        .zip(reduce_strides)
        .map(|(&dimension, &stride)| coords[dimension] * stride)
        .sum()
}

fn increment_coords(coords: &mut [usize], shape: &[usize]) {
    for (dimension, &size) in shape.iter().enumerate() {
        if size == 0 {
            continue;
        }
        coords[dimension] += 1;
        if coords[dimension] < size {
            return;
        }
        coords[dimension] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_preserves_uint64_bits_and_saturates() {
        let result =
            sum(&IntegerStorage::U64(vec![u64::MAX, 1, 5, 7]), &[2, 2], &[0]).expect("native sum");
        assert_eq!(
            result,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 12]), vec![1, 2])
                    .expect("expected tensor")
            )
        );
    }

    #[test]
    fn sum_handles_multidimensional_reduction_and_out_of_bounds_dim() {
        let storage = IntegerStorage::I16(vec![10, 20, 30, 40]);
        assert_eq!(
            sum(&storage, &[2, 2], &[1]).expect("row reduction"),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![40, 60]), vec![2, 1])
                    .expect("expected tensor")
            )
        );
        assert_eq!(
            sum(&storage, &[2, 2], &[]).expect("out of bounds"),
            Value::Tensor(Tensor::new_integer(storage, vec![2, 2]).expect("unchanged tensor"))
        );
    }

    #[test]
    fn product_preserves_native_type_identity_and_saturates() {
        let result = product(&IntegerStorage::U8(vec![2, 200, 3, 2]), &[2, 2], &[0])
            .expect("native product");
        assert_eq!(
            result,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![255, 6]), vec![1, 2])
                    .expect("expected tensor")
            )
        );
    }

    #[test]
    fn extrema_preserves_uint64_and_matlab_dimension_indices() {
        let storage = IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX, 3, 2]);
        let min = extrema(
            &storage,
            &[2, 2],
            vec![1, 2],
            &[0],
            &[true, false],
            &[1],
            false,
            false,
            ExtremaDirection::Min,
            ExtremaComparison::Natural,
        )
        .expect("min");
        assert_eq!(
            min.values,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX - 1, 2]), vec![1, 2])
                    .expect("values")
            )
        );
        assert_eq!(
            min.indices,
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap())
        );
    }

    #[test]
    fn extrema_absolute_comparison_handles_int64_minimum_without_overflow() {
        let storage = IntegerStorage::I64(vec![i64::MIN, -3, 3]);
        let result = extrema(
            &storage,
            &[3, 1],
            vec![1, 1],
            &[0],
            &[true, false],
            &[1],
            false,
            false,
            ExtremaDirection::Max,
            ExtremaComparison::Absolute,
        )
        .expect("max by absolute value");
        assert_eq!(result.values, Value::Int(IntValue::I64(i64::MIN)));
        assert_eq!(result.indices, Value::Num(1.0));
    }

    #[test]
    fn cumulative_scans_preserve_class_direction_and_saturation() {
        let sum = cumulative(
            &IntegerStorage::U8(vec![250, 10, 2, 3]),
            &[2, 2],
            1,
            CumulativeDirection::Forward,
            CumulativeOperation::Sum,
        )
        .expect("cumsum");
        assert_eq!(
            sum,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![250, 255, 2, 5]), vec![2, 2])
                    .expect("sum")
            )
        );

        let product = cumulative(
            &IntegerStorage::I8(vec![2, 100, 3, 2]),
            &[2, 2],
            1,
            CumulativeDirection::Reverse,
            CumulativeOperation::Product,
        )
        .expect("cumprod");
        assert_eq!(
            product,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I8(vec![127, 100, 6, 2]), vec![2, 2])
                    .expect("product")
            )
        );

        assert_eq!(
            cumulative(
                &IntegerStorage::U64(vec![u64::MAX, 1]),
                &[1, 2],
                3,
                CumulativeDirection::Forward,
                CumulativeOperation::Sum,
            )
            .expect("out-of-range dimension"),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1]), vec![1, 2])
                    .expect("unchanged input")
            )
        );
    }
}
