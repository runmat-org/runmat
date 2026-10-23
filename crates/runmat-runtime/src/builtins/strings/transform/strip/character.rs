//! Page-aware character stripping with one admitted output backing.

use super::*;
use crate::builtins::common::character_output::{build, CharacterBuildErrors};
use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};

const CHECKPOINT_CHUNK: usize = 4096;

pub(super) async fn strip_char_array(
    runtime: Option<&RuntimeContext>,
    source: CharArray,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    source.validate_layout().map_err(invalid)?;
    let expectation = PatternExpectation::with_len(source.rows);
    let (direction, patterns) = parse_arguments(args, &expectation).await?;
    let rows = source.rows;
    let cols = source.cols;
    if rows == 0 {
        return publish_empty(runtime, source);
    }
    let pages = if source.shape.len() <= 2 {
        1
    } else {
        source.shape[2..]
            .iter()
            .try_fold(1usize, |product, &extent| product.checked_mul(extent))
            .ok_or_else(|| invalid("Character page geometry overflow"))?
    };
    let page_len = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("Character page geometry overflow"))?;
    let row_count = rows
        .checked_mul(pages)
        .ok_or_else(|| invalid("Character row geometry overflow"))?;
    if row_count == 0 {
        return publish_empty(runtime, source);
    }
    let source_len = page_len
        .checked_mul(pages)
        .ok_or_else(|| invalid("Character payload geometry overflow"))?;
    if source_len != source.data().len() {
        return Err(invalid("Character shape does not match payload"));
    }

    let mut width = 0usize;
    for index in 0..row_count {
        checkpoint(runtime, index)?;
        let start = index
            .checked_mul(cols)
            .ok_or_else(|| invalid("Character row offset overflow"))?;
        let (left, right) = trimmed_bounds(
            &source.data()[start..start + cols],
            direction,
            patterns.pattern_for_index(index % rows),
        );
        width = width.max(right - left);
    }
    let count = row_count
        .checked_mul(width)
        .ok_or_else(|| invalid("Character output size overflow"))?;
    let output_shape = |axis: usize| {
        if source.shape.len() == 1 || axis == 1 {
            width
        } else {
            source.shape[axis]
        }
    };
    let fill = |output: &mut [Utf16CodeUnit],
                receipt: Option<&HostAllocationReceipt>|
     -> BuiltinResult<()> {
        for index in 0..row_count {
            if let Some(receipt) = receipt {
                if index.is_multiple_of(CHECKPOINT_CHUNK) {
                    let runtime = runtime
                        .ok_or_else(|| invalid("Managed character receipt lacks runtime"))?;
                    receipt
                        .checkpoint(runtime)
                        .map_err(|cause| strip_allocation_error(runtime, cause))?;
                }
            } else {
                checkpoint(runtime, index)?;
            }
            let source_start = index * cols;
            let (left, right) = trimmed_bounds(
                &source.data()[source_start..source_start + cols],
                direction,
                patterns.pattern_for_index(index % rows),
            );
            let target_start = index * width;
            output[target_start..target_start + right - left]
                .copy_from_slice(&source.data()[source_start + left..source_start + right]);
        }
        Ok(())
    };
    let policy = HostAllocationReceipt::policy(runtime).map_err(|cause| match runtime {
        Some(runtime) => strip_allocation_error(runtime, cause),
        None => invalid(format!("Character output policy: {cause}")),
    })?;
    if policy == HostAllocationPolicy::Managed {
        let runtime = runtime.ok_or_else(|| invalid("Managed character output lacks runtime"))?;
        let output = build(
            runtime,
            source.shape.len(),
            count,
            0,
            |axis| Ok(output_shape(axis)),
            |data, _, receipt| fill(data, Some(receipt)),
            CharacterBuildErrors {
                admission: strip_allocation_error,
                invalid,
            },
        )?;
        return Ok(Value::CharArray(output));
    }
    let mut data = vec![Utf16CodeUnit::from_raw(b' ' as u16); count];
    fill(&mut data, None)?;
    let shape = (0..source.shape.len()).map(output_shape).collect();
    CharArray::from_code_units_with_shape(data, shape)
        .map(Value::CharArray)
        .map_err(invalid)
}

fn publish_empty(runtime: Option<&RuntimeContext>, source: CharArray) -> BuiltinResult<Value> {
    let policy = HostAllocationReceipt::policy(runtime).map_err(|cause| match runtime {
        Some(runtime) => strip_allocation_error(runtime, cause),
        None => invalid(format!("Character output policy: {cause}")),
    })?;
    if policy == HostAllocationPolicy::Unbounded
        || (source.shape.is_admitted() && source.data_owner().is_exact())
    {
        return Ok(Value::CharArray(source));
    }
    let runtime = runtime.ok_or_else(|| invalid("Managed character output lacks runtime"))?;
    build(
        runtime,
        source.shape.len(),
        0,
        0,
        |axis| Ok(source.shape[axis]),
        |_, _, _| Ok(()),
        CharacterBuildErrors {
            admission: strip_allocation_error,
            invalid,
        },
    )
    .map(Value::CharArray)
}

pub(super) fn trimmed_bounds(
    row: &[Utf16CodeUnit],
    direction: StripDirection,
    pattern: PatternRef<'_>,
) -> (usize, usize) {
    let matches = |value: Utf16CodeUnit| match pattern {
        PatternRef::Default => char::from_u32(value.raw() as u32).is_some_and(|value| {
            crate::builtins::strings::common::trimmable_whitespace(
                value,
                !crate::compatibility::runmat_extensions_enabled(),
            )
        }),
        PatternRef::CustomText(chars) => chars.encode_utf16().any(|unit| unit == value.raw()),
        PatternRef::CustomChars(chars) => chars.contains(&value),
    };
    let mut left = 0;
    let mut right = row.len();
    if direction != StripDirection::Right {
        while left < right && matches(row[left]) {
            left += 1;
        }
    }
    if direction != StripDirection::Left {
        while right > left && matches(row[right - 1]) {
            right -= 1;
        }
    }
    (left, right)
}

fn invalid(detail: impl Into<String>) -> RuntimeError {
    strip_error_with_message(
        format!("{BUILTIN_NAME}: {}", detail.into()),
        &STRIP_ERROR_INTERNAL,
    )
}

fn checkpoint(runtime: Option<&RuntimeContext>, index: usize) -> BuiltinResult<()> {
    if index.is_multiple_of(CHECKPOINT_CHUNK) {
        if let Some(runtime) = runtime {
            if runtime.is_cancelled() {
                return Err(strip_allocation_error(
                    runtime,
                    HostAllocationError::Cancelled,
                ));
            }
        }
    }
    Ok(())
}
