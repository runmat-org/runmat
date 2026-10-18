//! Checked simultaneous host envelope for one 2-D tensor-product spline.
//! Values already owned by the caller are excluded; new parser/download buffers
//! enter through `input_scratch_bytes`. Exact derivative claims consume the
//! stage envelope, while final output claims consume separate headroom.

use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};
use runmat_value::NumericDType;

pub(crate) fn tensor_spline_allocation_plan(
    rows: usize,
    columns: usize,
    lanes: usize,
    output_count: usize,
    output_rank: usize,
    dtype: NumericDType,
    input_scratch_bytes: u64,
) -> Result<HostAllocationPlan, HostAllocationError> {
    let grid_count = rows
        .checked_mul(columns)
        .ok_or(HostAllocationError::Overflow)?;
    let derivative = HostAllocationPlan::numeric_workspace_bytes(NumericDType::F64, grid_count)?;
    let derivative_total = derivative
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_mul(lanes as u64))
        .ok_or(HostAllocationError::Overflow)?;
    let longest = rows.max(columns) as u64;
    // During the shared not-a-knot solve, h/secants, four tridiagonal arrays,
    // interior, second derivatives, and coefficient output have at most twelve
    // f64 elements per break concurrently. The four knot fields remain live.
    // This is a reserved Vec-request envelope, not an exact allocator claim.
    let coefficient_peak = longest
        .checked_mul(12)
        .and_then(|elements| elements.checked_mul(8))
        .ok_or(HostAllocationError::Overflow)?;
    let scratch = input_scratch_bytes
        .checked_add(derivative_total)
        .and_then(|bytes| bytes.checked_add(coefficient_peak))
        .and_then(|bytes| bytes.checked_add(16)) // gridded query coordinate pair
        .ok_or(HostAllocationError::Overflow)?;
    let output = if lanes == 1 && output_count == 1 && dtype == NumericDType::F64 {
        0
    } else if lanes == 2 {
        HostAllocationPlan::retained_complex_tensor_bytes_for_dimensions(
            dtype,
            output_count,
            output_rank.max(2),
        )?
    } else {
        HostAllocationPlan::retained_tensor_bytes_for_dimensions(
            dtype,
            output_count,
            output_rank.max(2),
        )?
    };
    HostAllocationPlan::new(scratch, output)
}
