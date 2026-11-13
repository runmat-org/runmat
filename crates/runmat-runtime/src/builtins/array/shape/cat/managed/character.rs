use super::*;
use crate::builtins::common::concatenation::char_from_f64_code_unit;
use runmat_value::Utf16CodeUnit;

fn source_rank(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::CharArray(chars) => Ok(chars.shape().len().max(2)),
        Value::Tensor(tensor) => Ok(tensor.shape().len().max(2)),
        Value::Num(_) | Value::Int(_) => Ok(2),
        _ => Err(cat_err(
            "cat: selected character owner received an incompatible class",
        )),
    }
}

fn source_extent(value: &Value, axis: usize) -> BuiltinResult<usize> {
    match value {
        Value::CharArray(chars) if chars.shape().is_empty() => Ok(if axis < 2 { 0 } else { 1 }),
        Value::CharArray(chars) if chars.shape().len() == 1 => Ok(match axis {
            0 => 1,
            1 => chars.shape()[0],
            _ => 1,
        }),
        Value::CharArray(chars) => Ok(chars.shape().get(axis).copied().unwrap_or(1)),
        Value::Tensor(tensor) if tensor.shape().is_empty() => Ok(if axis < 2 { 0 } else { 1 }),
        Value::Tensor(tensor) if tensor.shape().len() == 1 => Ok(match axis {
            0 => 1,
            1 => tensor.shape()[0],
            _ => 1,
        }),
        Value::Tensor(tensor) => Ok(tensor.shape().get(axis).copied().unwrap_or(1)),
        Value::Num(_) | Value::Int(_) => Ok(1),
        _ => Err(cat_err(
            "cat: selected character owner received an incompatible class",
        )),
    }
}

fn source_count(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::CharArray(chars) => Ok(chars.data().len()),
        Value::Tensor(tensor) => Ok(tensor.len()),
        Value::Num(_) | Value::Int(_) => Ok(1),
        _ => Err(cat_err(
            "cat: selected character owner received an incompatible class",
        )),
    }
}

fn active(value: &Value, rank: usize, has_nonempty: bool) -> BuiltinResult<bool> {
    if !has_nonempty {
        return Ok(true);
    }
    for axis in 0..rank {
        if source_extent(value, axis)? == 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn output_extent(
    values: FunctionArgumentView<'_>,
    axis: usize,
    dim: usize,
    rank: usize,
    has_nonempty: bool,
) -> BuiltinResult<usize> {
    let mut first = None;
    let mut sum = 0usize;
    for (index, value) in values.iter().enumerate() {
        if !active(value, rank, has_nonempty)? {
            continue;
        }
        let extent = source_extent(value, axis)?;
        if axis == dim {
            sum = sum
                .checked_add(extent)
                .ok_or_else(|| cat_err("cat: concatenated dimension exceeds maximum size"))?;
        } else if let Some(reference) = first {
            if extent != reference {
                return Err(cat_err(format!(
                    "cat: dimension {} mismatch between input 1 and input {}",
                    axis + 1,
                    index + 1
                )));
            }
        } else {
            first = Some(extent);
        }
    }
    Ok(if axis == dim { sum } else { first.unwrap_or(1) })
}

fn source_char(value: &Value, index: usize) -> BuiltinResult<Utf16CodeUnit> {
    match value {
        Value::CharArray(chars) => {
            let page_len = chars
                .rows
                .checked_mul(chars.cols)
                .ok_or_else(|| cat_err("cat: character page size overflow"))?;
            if page_len == 0 {
                return Err(cat_err(
                    "cat: empty character source was selected for copying",
                ));
            }
            let page = index / page_len;
            let within = index % page_len;
            let row = within % chars.rows;
            let col = within / chars.rows;
            let physical = page
                .checked_mul(page_len)
                .and_then(|base| {
                    row.checked_mul(chars.cols)
                        .and_then(|offset| base.checked_add(offset))
                })
                .and_then(|base| base.checked_add(col))
                .ok_or_else(|| cat_err("cat: character source index overflow"))?;
            chars
                .data()
                .get(physical)
                .copied()
                .ok_or_else(|| cat_err("cat: character source geometry changed"))
        }
        Value::Tensor(tensor) => tensor
            .numeric_value_at(index)
            .ok_or_else(|| cat_err("cat: numeric source geometry changed"))
            .and_then(|scalar| char_from_f64_code_unit(scalar.materialize_f64(), "cat")),
        Value::Num(number) if index == 0 => char_from_f64_code_unit(*number, "cat"),
        Value::Int(number) if index == 0 => char_from_f64_code_unit(number.to_f64(), "cat"),
        _ => Err(cat_err("cat: character source geometry changed")),
    }
}

pub(in super::super) fn character(
    context: &RuntimeContext,
    dim: usize,
    values: FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    if values.is_empty() {
        return Err(cat_err("cat: no inputs to concatenate"));
    }
    let mut rank = dim
        .checked_add(1)
        .ok_or_else(|| cat_err("cat: dimension exceeds maximum size"))?;
    let mut has_nonempty = false;
    for value in values {
        rank = rank.max(source_rank(value)?);
        has_nonempty |= source_count(value)? != 0;
    }
    for value in values {
        let mut count = 1usize;
        let mut zero = false;
        for axis in 0..rank {
            let extent = source_extent(value, axis)?;
            zero |= extent == 0;
            if !zero {
                count = count
                    .checked_mul(extent)
                    .ok_or_else(|| cat_err("cat: input dimensions exceed maximum size"))?;
            }
        }
        if (if zero { 0 } else { count }) != source_count(value)? {
            return Err(cat_err(
                "cat: input shape does not match its character payload",
            ));
        }
        if let Value::CharArray(chars) = value {
            let expected = match chars.shape() {
                [] => (0, 0),
                [length] => (1, *length),
                [rows, cols, ..] => (*rows, *cols),
            };
            if (chars.rows, chars.cols) != expected {
                return Err(cat_err(
                    "cat: character shape cache does not match its dimensions",
                ));
            }
        }
        if matches!(value, Value::Tensor(_) | Value::Num(_) | Value::Int(_)) {
            for index in 0..source_count(value)? {
                if index % CHECKPOINT_CHUNK == 0 && context.is_cancelled() {
                    return Err(admission_error(context, HostAllocationError::Cancelled));
                }
                source_char(value, index)?;
            }
        }
    }
    let mut has_zero = false;
    for axis in 0..rank {
        has_zero |= output_extent(values, axis, dim, rank, has_nonempty)? == 0;
    }
    let mut count: usize = if has_zero { 0 } else { 1 };
    if !has_zero {
        for axis in 0..rank {
            count = count
                .checked_mul(output_extent(values, axis, dim, rank, has_nonempty)?)
                .ok_or_else(|| cat_err("cat: resulting array exceeds maximum size"))?;
        }
    }
    let minimum_rank = rank.min((dim + 1).max(2));
    let mut output_rank = minimum_rank;
    for axis in minimum_rank..rank {
        if output_extent(values, axis, dim, rank, has_nonempty)? != 1 {
            output_rank = axis + 1;
        }
    }
    let final_rank = output_rank.max(2);
    let output = crate::builtins::common::character_output::build(
        context,
        final_rank,
        count,
        0,
        |axis| {
            if rank == 1 {
                Ok(if axis == 0 {
                    1
                } else {
                    output_extent(values, 0, dim, rank, has_nonempty)?
                })
            } else {
                output_extent(values, axis, dim, rank, has_nonempty)
            }
        },
        |data, shape, receipt| {
            let inner = if dim == 0 {
                1
            } else {
                (0..dim).try_fold(1usize, |product, axis| {
                    product
                        .checked_mul(output_extent(values, axis, dim, rank, has_nonempty)?)
                        .ok_or_else(|| cat_err("cat: inner dimension overflow"))
                })?
            };
            let outer = if dim + 1 >= rank {
                1
            } else {
                ((dim + 1)..rank).try_fold(1usize, |product, axis| {
                    product
                        .checked_mul(output_extent(values, axis, dim, rank, has_nonempty)?)
                        .ok_or_else(|| cat_err("cat: outer dimension overflow"))
                })?
            };
            let rows = shape[0];
            let cols = shape[1];
            let page_len = rows
                .checked_mul(cols)
                .ok_or_else(|| cat_err("cat: character page size overflow"))?;
            let mut destination = 0usize;
            for outer_index in 0..outer {
                for value in values {
                    if !active(value, rank, has_nonempty)? {
                        continue;
                    }
                    let chunk = source_extent(value, dim)?
                        .checked_mul(inner)
                        .ok_or_else(|| cat_err("cat: source chunk overflow"))?;
                    let offset = outer_index
                        .checked_mul(chunk)
                        .ok_or_else(|| cat_err("cat: source offset overflow"))?;
                    let end = offset
                        .checked_add(chunk)
                        .ok_or_else(|| cat_err("cat: source range overflow"))?;
                    for source_index in offset..end {
                        if destination.is_multiple_of(CHECKPOINT_CHUNK) {
                            receipt
                                .checkpoint(context)
                                .map_err(|cause| admission_error(context, cause))?;
                        }
                        let page = if page_len == 0 {
                            0
                        } else {
                            destination / page_len
                        };
                        let within = if page_len == 0 {
                            0
                        } else {
                            destination % page_len
                        };
                        let row_major = if rows == 0 {
                            0
                        } else {
                            page * page_len + (within % rows) * cols + within / rows
                        };
                        *data.get_mut(row_major).ok_or_else(|| {
                            cat_err("cat: character destination geometry changed")
                        })? = source_char(value, source_index)?;
                        destination += 1;
                    }
                }
            }
            if destination != count {
                return Err(cat_err(
                    "cat: assembly did not fill admitted character output",
                ));
            }
            Ok(())
        },
        crate::builtins::common::character_output::CharacterBuildErrors {
            admission: admission_error,
            invalid: |detail| cat_err(format!("cat: {detail}")),
        },
    )?;
    Ok(Value::CharArray(output))
}
