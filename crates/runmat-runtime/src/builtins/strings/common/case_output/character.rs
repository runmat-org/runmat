//! Borrowed UTF-16 rows to one exact case-mapped character output.

use super::{unicode16, TextCase, CHECKPOINT_CHUNK};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{
    CharArray, ExactArrayInitializationError, ExactArrayStorage, SharedShape, Utf16CodeUnit,
    Utf16DecodeError,
};

fn scalar(row: &[Utf16CodeUnit], index: usize) -> Result<(char, usize), Utf16DecodeError> {
    char::decode_utf16(row[index..].iter().map(|unit| unit.raw()))
        .next()
        .expect("nonempty row suffix")
        .map(|character| (character, character.len_utf16()))
        .map_err(|error| Utf16DecodeError {
            index,
            unpaired_surrogate: error.unpaired_surrogate(),
        })
}

fn visit(
    row: &[Utf16CodeUnit],
    mode: TextCase,
    suffix: Option<&[u8]>,
    mut emit: impl FnMut(char) -> BuiltinResult<()>,
    mut checkpoint: impl FnMut() -> BuiltinResult<()>,
    invalid: &impl Fn(String) -> RuntimeError,
) -> BuiltinResult<bool> {
    let mut index = 0;
    let mut has_sigma = false;
    let mut prefix_cased = false;
    let mut next_checkpoint = 0;
    while index < row.len() {
        if index >= next_checkpoint {
            checkpoint()?;
            next_checkpoint = index.saturating_add(CHECKPOINT_CHUNK);
        }
        let (character, width) = scalar(row, index).map_err(|error| invalid(error.to_string()))?;
        has_sigma |= character == 'Σ';
        match mode {
            TextCase::Lower
                if character == 'Σ'
                    && prefix_cased
                    && suffix.is_some_and(|flags| flags[index] == 0) =>
            {
                emit('ς')?
            }
            TextCase::Lower => {
                for mapped in character.to_lowercase() {
                    emit(mapped)?;
                }
            }
            TextCase::Upper => {
                for mapped in character.to_uppercase() {
                    emit(mapped)?;
                }
            }
        }
        if !unicode16::is_case_ignorable(character) {
            prefix_cased = unicode16::is_cased(character);
        }
        index += width;
    }
    Ok(has_sigma)
}

// Input was strictly validated by the sizing pass. Walk backwards in code-unit
// geometry; a supplementary scalar contributes one contextual state update.
fn suffix_flags(
    row: &[Utf16CodeUnit],
    flags: &mut [u8],
    mut checkpoint: impl FnMut() -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    let mut end = row.len();
    let mut next_cased = false;
    let mut seen = 0usize;
    while end != 0 {
        if seen.is_multiple_of(CHECKPOINT_CHUNK) {
            checkpoint()?;
        }
        let start = if (0xDC00..=0xDFFF).contains(&row[end - 1].raw()) {
            end - 2
        } else {
            end - 1
        };
        let (character, _) =
            scalar(row, start).expect("sizing pass validated immutable UTF-16 row");
        flags[start] = u8::from(next_cased);
        if !unicode16::is_case_ignorable(character) {
            next_cased = unicode16::is_cased(character);
        }
        end = start;
        seen += 1;
    }
    Ok(())
}

pub(crate) fn transform_character_case(
    runtime: Option<&RuntimeContext>,
    input: CharArray,
    mode: TextCase,
    admission: fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
    invalid: impl Fn(String) -> RuntimeError,
) -> BuiltinResult<CharArray> {
    let checkpoint = || {
        if let Some(runtime) = runtime.filter(|runtime| runtime.is_cancelled()) {
            return Err(admission(runtime, HostAllocationError::Cancelled));
        }
        Ok(())
    };
    checkpoint()?;
    if input.data().is_empty() {
        return Ok(input);
    }
    let mut columns = input.cols;
    let mut sigma = false;
    for row in input.data().chunks_exact(input.cols) {
        checkpoint()?;
        let mut width = 0usize;
        let has_sigma = visit(
            row,
            mode,
            None,
            |character| {
                width = width
                    .checked_add(character.len_utf16())
                    .ok_or_else(|| invalid("Character output size overflow".into()))?;
                Ok(())
            },
            checkpoint,
            &invalid,
        )?;
        sigma |= mode == TextCase::Lower && has_sigma;
        columns = columns.max(width);
    }
    // Physical rows are contiguous within each page. The same row mapper
    // visits all pages, with one common padded column extent.
    let count = (input.data().len() / input.cols)
        .checked_mul(columns)
        .ok_or_else(|| invalid("Character output size overflow".into()))?;
    let rank = if columns == input.cols {
        input.shape().len()
    } else {
        input.shape().len().max(2)
    };
    let shape_at = |axis| {
        if axis == 1 {
            columns
        } else {
            input.shape().get(axis).copied().unwrap_or(1)
        }
    };
    let scratch_len = if sigma { input.cols } else { 0 };
    let fill = |output: &mut [Utf16CodeUnit],
                flags: &mut [u8],
                check: &dyn Fn() -> BuiltinResult<()>|
     -> BuiltinResult<()> {
        for (row_index, row) in input.data().chunks_exact(input.cols).enumerate() {
            check()?;
            let suffix = if sigma {
                suffix_flags(row, flags, check)?;
                Some(&flags[..])
            } else {
                None
            };
            let mut cursor = row_index * columns;
            visit(
                row,
                mode,
                suffix,
                |character| {
                    let mut encoded = [0u16; 2];
                    for &unit in character.encode_utf16(&mut encoded).iter() {
                        output[cursor] = Utf16CodeUnit::from_raw(unit);
                        cursor += 1;
                    }
                    Ok(())
                },
                check,
                &invalid,
            )?;
        }
        Ok(())
    };
    let ordinary = || {
        let mut data = runmat_value::exact_capacity_vec::<Utf16CodeUnit>(count)
            .map_err(|error| invalid(error.to_string()))?;
        data.resize(count, Utf16CodeUnit::from_raw(32));
        let mut flags = runmat_value::exact_capacity_vec::<u8>(scratch_len)
            .map_err(|error| invalid(error.to_string()))?;
        flags.resize(scratch_len, 0);
        fill(&mut data, &mut flags, &checkpoint)?;
        let mut shape = runmat_value::exact_capacity_vec::<usize>(rank)
            .map_err(|error| invalid(error.to_string()))?;
        shape.extend((0..rank).map(shape_at));
        CharArray::from_code_units_with_shape(data, shape).map_err(&invalid)
    };
    let Some(runtime) = runtime else {
        return ordinary();
    };
    let payload = ExactArrayStorage::<Utf16CodeUnit>::requested_bytes(count)
        .map_err(|error| admission(runtime, HostAllocationError::Numeric(error)))?;
    let shape = SharedShape::requested_bytes_for_rank(rank)
        .map_err(|error| admission(runtime, HostAllocationError::Numeric(error)))?;
    let bytes = payload
        .checked_add(shape)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
    let retained =
        HostAllocationPlan::retained_exact_output_bytes(bytes, usize::from(count != 0) + 1)
            .map_err(|error| admission(runtime, error))?;
    let flags_scratch = if scratch_len == 0 {
        0
    } else {
        HostTypedWorkspace::<u8>::quoted_bytes(scratch_len)
            .map_err(|error| admission(runtime, error))?
    };
    let shape_scratch = HostTypedWorkspace::<usize>::quoted_bytes(rank)
        .map_err(|error| admission(runtime, error))?;
    let scratch = flags_scratch
        .checked_add(shape_scratch)
        .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
    let plan =
        HostAllocationPlan::new(scratch, retained).map_err(|error| admission(runtime, error))?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|error| admission(runtime, error))?
    else {
        return ordinary();
    };
    let mut shape_workspace = receipt
        .temporary_typed::<usize>(runtime, rank)
        .map_err(|error| admission(runtime, error))?;
    for (axis, extent) in shape_workspace.as_mut_slice().iter_mut().enumerate() {
        if axis.is_multiple_of(CHECKPOINT_CHUNK) {
            receipt
                .checkpoint(runtime)
                .map_err(|error| admission(runtime, error))?;
        }
        *extent = shape_at(axis);
    }
    let shape = receipt
        .with_output_allocation(runtime, |permit| {
            SharedShape::new_admitted_from_slice(shape_workspace.as_slice(), permit)
        })
        .map_err(|error| admission(runtime, error))?;
    let mut flags = if scratch_len == 0 {
        None
    } else {
        Some(
            receipt
                .temporary_typed::<u8>(runtime, scratch_len)
                .map_err(|error| admission(runtime, error))?,
        )
    };
    let mut data = receipt
        .with_output_construction(runtime, |permit| {
            ExactArrayStorage::try_initialize(count, permit, |index| {
                if index.is_multiple_of(CHECKPOINT_CHUNK) {
                    receipt.checkpoint(runtime)?;
                }
                Ok::<_, HostAllocationError>(Utf16CodeUnit::from_raw(32))
            })
        })
        .map_err(|error| admission(runtime, error))?
        .map_err(|error| match error {
            ExactArrayInitializationError::Allocation(error) => {
                admission(runtime, HostAllocationError::Numeric(error))
            }
            ExactArrayInitializationError::Element(error) => admission(runtime, error),
        })?;
    let check = || {
        receipt
            .checkpoint(runtime)
            .map_err(|error| admission(runtime, error))
    };
    fill(
        data.try_as_mut_slice_unique()
            .expect("fresh character output"),
        flags
            .as_mut()
            .map(|flags| flags.as_mut_slice())
            .unwrap_or(&mut []),
        &check,
    )?;
    check()?;
    CharArray::from_admitted(data, shape).map_err(invalid)
}
