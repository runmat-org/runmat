//! Claimed source-map, text output, and logical mask for string filling.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_value::{ExactHostLogicalTensorAllocation, LogicalArray, StringArray, Value};

use super::super::numeric::is_missing_text;
use super::super::text_output::{selected_string_array, TextLiteral};
use super::super::{internal_error, invalid_argument, unsupported_type};
use super::lines::FillLines;
use super::text::{fill_selection_map, selected, NeighborSelection};
use super::{FillMethod, FillOptions};

pub(super) fn fill_missing_string_array(
    runtime: &RuntimeContext,
    array: StringArray,
    options: &FillOptions,
    lines: &FillLines,
) -> BuiltinResult<(Value, LogicalArray)> {
    let method = match &options.method {
        FillMethod::Constant(_) => NeighborSelection::Constant,
        FillMethod::Previous => NeighborSelection::Previous,
        FillMethod::Next => NeighborSelection::Next,
        FillMethod::Nearest => NeighborSelection::Nearest,
        _ => {
            return Err(unsupported_type(
                "fillmissing: string arrays support constant, previous, next, and nearest",
            ))
        }
    };
    let literal = match &options.method {
        FillMethod::Constant(value) => match value {
            Value::String(value) => value
                .as_str()
                .map_or(TextLiteral::Missing, TextLiteral::Utf8),
            Value::StringArray(value) if value.data().len() == 1 => value.data()[0]
                .as_str()
                .map_or(TextLiteral::Missing, TextLiteral::Utf8),
            Value::CharArray(value) if value.rows == 1 => TextLiteral::Chars(value.data()),
            _ => {
                return Err(invalid_argument(
                    "fillmissing: string constant must be text",
                ))
            }
        },
        _ => TextLiteral::Missing,
    };
    let admission = |cause| super::super::errors::allocation_error(runtime, "fillmissing", cause);
    let count = array.data().len();
    let scratch = HostTypedWorkspace::<usize>::quoted_bytes(count).map_err(admission)?;
    let scratch_plan = HostAllocationPlan::new(scratch, 0).map_err(admission)?;
    let scratch_receipt = HostAllocationReceipt::admit(Some(runtime), scratch_plan)
        .map_err(admission)?
        .ok_or_else(|| admission(HostAllocationError::MissingManagedOwner))?;
    let mut selections = scratch_receipt
        .temporary_typed::<usize>(runtime, count)
        .map_err(admission)?;
    fill_selection_map(&array, lines, method, selections.as_mut_slice(), |_| {
        scratch_receipt.checkpoint(runtime).map_err(admission)
    })?;
    let output = selected_string_array(
        Some(runtime),
        array,
        literal,
        |_, index| selected(selections.as_slice()[index]),
        |runtime, cause| super::super::errors::allocation_error(runtime, "fillmissing", cause),
        internal_error,
    )?;
    let mask_bytes =
        HostAllocationPlan::retained_logical_tensor_bytes(&output.shape).map_err(admission)?;
    let mask_plan = HostAllocationPlan::new(0, mask_bytes).map_err(admission)?;
    let mask_receipt = HostAllocationReceipt::admit(Some(runtime), mask_plan)
        .map_err(admission)?
        .ok_or_else(|| admission(HostAllocationError::MissingManagedOwner))?;
    let mut allocation = mask_receipt
        .with_output_allocation(runtime, |permit| {
            ExactHostLogicalTensorAllocation::new(&output.shape, permit)
        })
        .map_err(admission)?;
    for (index, flag) in allocation.as_mut_slice().iter_mut().enumerate() {
        if index % 4096 == 0 {
            mask_receipt.checkpoint(runtime).map_err(admission)?;
        }
        *flag = u8::from(
            selections.as_slice()[index] != index && !is_missing_text(&output.data()[index]),
        );
    }
    let mask = mask_receipt
        .with_output_allocation(runtime, |_| allocation.into_array())
        .map_err(admission)?;
    mask_receipt.checkpoint(runtime).map_err(admission)?;
    Ok((Value::StringArray(output), mask))
}
