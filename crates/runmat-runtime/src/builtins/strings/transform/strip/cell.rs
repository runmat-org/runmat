//! Exact cell header and child publication for selected Host strip work.

use super::*;
use crate::context::host_allocation::{
    HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use runmat_value::{
    exact_capacity_vec, ExactArrayInitializationError, ExactArrayStorage,
    ExactHostAllocationPermit, OutputListStorage, SharedShape, Utf16CodeUnit,
};

const CHECKPOINT_CHUNK: usize = 4096;

pub(super) async fn strip_cell_array(
    runtime: Option<&RuntimeContext>,
    cell: CellArray,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let expectation = PatternExpectation::with_shape(cell.data().len(), cell.shape.as_slice());
    let (direction, patterns) = parse_arguments(args, &expectation).await?;
    let policy =
        HostAllocationReceipt::policy(runtime).map_err(|cause| allocation(runtime, cause))?;
    if policy == HostAllocationPolicy::Unbounded {
        let mut values = Vec::with_capacity(cell.data().len());
        for (index, value) in cell.data().iter().enumerate() {
            checkpoint(runtime, index)?;
            values.push(ordinary_child(
                value,
                direction,
                patterns.pattern_for_index(index),
            )?);
        }
        return CellArray::new_with_shape(values, cell.shape.as_slice().to_vec())
            .map(Value::Cell)
            .map_err(invalid);
    }
    let runtime = runtime.ok_or_else(|| invalid("Managed cell output lacks runtime"))?;
    let mut bytes = OutputListStorage::requested_bytes(cell.data().len())
        .map_err(|cause| strip_allocation_error(runtime, HostAllocationError::Numeric(cause)))?;
    let mut claims = usize::from(!cell.data().is_empty());
    let reuse_shape = cell.shape.is_admitted();
    if !reuse_shape {
        add_quote(
            &mut bytes,
            &mut claims,
            SharedShape::requested_bytes_for_rank(cell.shape.len()).map_err(|cause| {
                strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
            })?,
            1,
            runtime,
        )?;
    }
    for (index, value) in cell.data().iter().enumerate() {
        checkpoint(Some(runtime), index)?;
        quote_child(
            value,
            direction,
            patterns.pattern_for_index(index),
            &mut bytes,
            &mut claims,
            runtime,
        )?;
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes)
            .map_err(|_| strip_allocation_error(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|cause| strip_allocation_error(runtime, cause))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained)
            .map_err(|cause| strip_allocation_error(runtime, cause))?,
    )
    .map_err(|cause| strip_allocation_error(runtime, cause))?
    .ok_or_else(|| strip_allocation_error(runtime, HostAllocationError::MissingManagedOwner))?;
    let output = receipt
        .with_output_construction(runtime, |permit| {
            let shape = if reuse_shape {
                cell.shape.clone()
            } else {
                SharedShape::new_admitted_from_slice(cell.shape.as_slice(), permit).map_err(
                    |cause| strip_allocation_error(runtime, HostAllocationError::Numeric(cause)),
                )?
            };
            let data = OutputListStorage::try_initialize(cell.data().len(), permit, |index| {
                if index.is_multiple_of(CHECKPOINT_CHUNK) {
                    receipt
                        .checkpoint(runtime)
                        .map_err(|cause| strip_allocation_error(runtime, cause))?;
                }
                admitted_child(
                    &cell.data()[index],
                    direction,
                    patterns.pattern_for_index(index),
                    permit,
                    runtime,
                )
            })
            .map_err(|cause| match cause {
                ExactArrayInitializationError::Allocation(cause) => {
                    strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
                }
                ExactArrayInitializationError::Element(cause) => cause,
            })?;
            CellArray::from_admitted(data, shape)
                .map(Value::Cell)
                .map_err(invalid)
        })
        .map_err(|cause| strip_allocation_error(runtime, cause))??;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| strip_allocation_error(runtime, cause))?;
    Ok(output)
}

fn text_child(value: &Value) -> Option<&StringElement> {
    match value {
        Value::String(text) => Some(text),
        Value::StringArray(array) if array.data().len() == 1 => array.data().first(),
        _ => None,
    }
}

fn character_child(value: &Value) -> BuiltinResult<Option<&CharArray>> {
    let Value::CharArray(array) = value else {
        return Ok(None);
    };
    if array.validate_layout().is_err() || array.shape.len() > 2 || array.rows > 1 {
        return Err(strip_error(&STRIP_ERROR_CELL_ELEMENT));
    }
    Ok(Some(array))
}

fn selected_text<'a>(
    text: &'a StringElement,
    direction: StripDirection,
    pattern: PatternRef<'_>,
) -> Option<&'a str> {
    text.as_str()
        .map(|text| strip_slice(text, direction, pattern))
}

fn quote_child(
    value: &Value,
    direction: StripDirection,
    pattern: PatternRef<'_>,
    bytes: &mut usize,
    claims: &mut usize,
    runtime: &RuntimeContext,
) -> BuiltinResult<()> {
    if let Some(text) = text_child(value) {
        let Some(selected) = selected_text(text, direction, pattern) else {
            return Ok(());
        };
        if Some(selected) != text.as_str() || !text.has_retained_claim() {
            let requested =
                Utf8Text::requested_bytes_for_capacity(selected.len()).map_err(|cause| {
                    strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
                })?;
            add_quote(
                bytes,
                claims,
                requested,
                usize::from(!selected.is_empty()),
                runtime,
            )?;
        }
        return Ok(());
    }
    if let Some(array) = character_child(value)? {
        if array.rows == 0 {
            if !array.shape.is_admitted() {
                let shape =
                    SharedShape::requested_bytes_for_rank(array.shape.len()).map_err(|cause| {
                        strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
                    })?;
                add_quote(bytes, claims, shape, 1, runtime)?;
            }
            return Ok(());
        }
        let (left, right) =
            character::trimmed_bounds(array.data_owner().as_slice(), direction, pattern);
        let count = right - left;
        let payload =
            ExactArrayStorage::<Utf16CodeUnit>::requested_bytes(count).map_err(|cause| {
                strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
            })?;
        let shape = SharedShape::requested_bytes_for_rank(array.shape.len()).map_err(|cause| {
            strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
        })?;
        add_quote(
            bytes,
            claims,
            payload
                .checked_add(shape)
                .ok_or_else(|| strip_allocation_error(runtime, HostAllocationError::Overflow))?,
            usize::from(count != 0) + 1,
            runtime,
        )?;
        return Ok(());
    }
    Err(strip_error(&STRIP_ERROR_CELL_ELEMENT))
}

fn admitted_child(
    value: &Value,
    direction: StripDirection,
    pattern: PatternRef<'_>,
    permit: &dyn ExactHostAllocationPermit,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    if let Some(text) = text_child(value) {
        let Some(selected) = selected_text(text, direction, pattern) else {
            return Ok(Value::String(StringElement::Missing));
        };
        if Some(selected) == text.as_str() && text.has_retained_claim() {
            return Ok(Value::String(text.clone()));
        }
        let mut bytes = exact_capacity_vec::<u8>(selected.len()).map_err(|cause| {
            strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
        })?;
        bytes.extend_from_slice(selected.as_bytes());
        // SAFETY: selected is a borrowed UTF-8 substring copied unchanged.
        let text = Utf8Text::new(unsafe { String::from_utf8_unchecked(bytes) });
        return text
            .try_admit_exact(permit)
            .map(|text| Value::String(StringElement::Text(text)))
            .map_err(|failure| {
                strip_allocation_error(runtime, HostAllocationError::Numeric(failure.source()))
            });
    }
    if let Some(array) = character_child(value)? {
        if array.rows == 0 {
            if array.shape.is_admitted() && array.data_owner().is_exact() {
                return Ok(Value::CharArray(array.clone()));
            }
            let shape = if array.shape.is_admitted() {
                array.shape.clone()
            } else {
                SharedShape::new_admitted_from_slice(array.shape.as_slice(), permit).map_err(
                    |cause| strip_allocation_error(runtime, HostAllocationError::Numeric(cause)),
                )?
            };
            let data = if array.data_owner().is_exact() {
                array.data_owner().clone()
            } else {
                ExactArrayStorage::try_initialize(0, permit, |_| {
                    Ok::<Utf16CodeUnit, RuntimeError>(Utf16CodeUnit::from_raw(b' ' as u16))
                })
                .map_err(|cause| match cause {
                    ExactArrayInitializationError::Allocation(cause) => {
                        strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
                    }
                    ExactArrayInitializationError::Element(cause) => cause,
                })?
            };
            return CharArray::from_admitted(data, shape)
                .map(Value::CharArray)
                .map_err(invalid);
        }
        let (left, right) =
            character::trimmed_bounds(array.data_owner().as_slice(), direction, pattern);
        let width = right - left;
        let row = [width];
        let matrix = [1, width];
        let dimensions: &[usize] = if array.shape.len() == 1 {
            &row
        } else {
            &matrix
        };
        let shape = SharedShape::new_admitted_from_slice(dimensions, permit).map_err(|cause| {
            strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
        })?;
        let data = ExactArrayStorage::try_initialize(width, permit, |index| {
            Ok::<Utf16CodeUnit, RuntimeError>(array.data()[left + index])
        })
        .map_err(|cause| match cause {
            ExactArrayInitializationError::Allocation(cause) => {
                strip_allocation_error(runtime, HostAllocationError::Numeric(cause))
            }
            ExactArrayInitializationError::Element(cause) => cause,
        })?;
        return CharArray::from_admitted(data, shape)
            .map(Value::CharArray)
            .map_err(invalid);
    }
    Err(strip_error(&STRIP_ERROR_CELL_ELEMENT))
}

fn ordinary_child(
    value: &Value,
    direction: StripDirection,
    pattern: PatternRef<'_>,
) -> BuiltinResult<Value> {
    if let Some(text) = text_child(value) {
        return Ok(Value::String(
            match selected_text(text, direction, pattern) {
                Some(text) => StringElement::text(text.to_owned()),
                None => StringElement::Missing,
            },
        ));
    }
    if let Some(array) = character_child(value)? {
        if array.rows == 0 {
            return Ok(Value::CharArray(array.clone()));
        }
        let (left, right) =
            character::trimmed_bounds(array.data_owner().as_slice(), direction, pattern);
        let width = right - left;
        let dimensions = if array.shape.len() == 1 {
            vec![width]
        } else {
            vec![1, width]
        };
        return CharArray::from_code_units_with_shape(
            array.data()[left..right].to_vec(),
            dimensions,
        )
        .map(Value::CharArray)
        .map_err(invalid);
    }
    Err(strip_error(&STRIP_ERROR_CELL_ELEMENT))
}

fn add_quote(
    bytes: &mut usize,
    claims: &mut usize,
    payload: usize,
    count: usize,
    runtime: &RuntimeContext,
) -> BuiltinResult<()> {
    *bytes = bytes
        .checked_add(payload)
        .ok_or_else(|| strip_allocation_error(runtime, HostAllocationError::Overflow))?;
    *claims = claims
        .checked_add(count)
        .ok_or_else(|| strip_allocation_error(runtime, HostAllocationError::Overflow))?;
    Ok(())
}

fn allocation(runtime: Option<&RuntimeContext>, cause: HostAllocationError) -> RuntimeError {
    match runtime {
        Some(runtime) => strip_allocation_error(runtime, cause),
        None => invalid(format!("Cell allocation policy: {cause}")),
    }
}

fn invalid(detail: impl Into<String>) -> RuntimeError {
    strip_error_with_message(
        format!("{BUILTIN_NAME}: {}", detail.into()),
        &STRIP_ERROR_INTERNAL,
    )
}

fn checkpoint(runtime: Option<&RuntimeContext>, index: usize) -> BuiltinResult<()> {
    if index.is_multiple_of(CHECKPOINT_CHUNK) && runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(allocation(runtime, HostAllocationError::Cancelled));
    }
    Ok(())
}
