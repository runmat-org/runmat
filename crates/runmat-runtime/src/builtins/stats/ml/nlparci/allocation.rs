//! Checked host envelope for the arrays owned by one nlparci call.
//! Existing host tensors and scalar f64 values are caller-owned; resident downloads, floating
//! conversions, complete-row packing, and SVD work overlap with the result.

use super::admission::{dimensions, Method};
use super::*;
use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};
use runmat_value::{ExactHostTensorAllocation, NumericDType};

fn overflow() -> RuntimeError {
    invalid("nlparci host allocation dimensions overflow")
}

fn bytes(count: usize, element_size: usize) -> BuiltinResult<u64> {
    let count = u64::try_from(count).map_err(|_| overflow())?;
    let size = u64::try_from(element_size).map_err(|_| overflow())?;
    count.checked_mul(size).ok_or_else(overflow)
}

fn add(total: &mut u64, next: u64) -> BuiltinResult<()> {
    *total = total.checked_add(next).ok_or_else(overflow)?;
    Ok(())
}

fn resident_download(value: &Value) -> BuiltinResult<u64> {
    if !matches!(value, Value::GpuTensor(_)) {
        return Ok(0);
    }
    let (rows, cols, dtype) = dimensions(value)?;
    let count = rows.checked_mul(cols).ok_or_else(overflow)?;
    let requested = ExactHostTensorAllocation::requested_bytes_for_dimensions(dtype, count, 2)
        .map_err(|_| overflow())?;
    u64::try_from(requested).map_err(|_| overflow())
}

fn floating_conversion(value: &Value) -> BuiltinResult<u64> {
    let (rows, cols, dtype) = dimensions(value)?;
    if dtype != NumericDType::F32 {
        return Ok(0);
    }
    bytes(
        rows.checked_mul(cols).ok_or_else(overflow)?,
        size_of::<f64>(),
    )
}

pub(super) fn plan(
    beta: &Value,
    residual: &Value,
    source: &Value,
    alpha: Option<&Value>,
    method: Method,
    produces_output: bool,
) -> BuiltinResult<HostAllocationPlan> {
    let (beta_rows, beta_cols, dtype) = dimensions(beta)?;
    let p = beta_rows.checked_mul(beta_cols).ok_or_else(overflow)?;
    let (residual_rows, residual_cols, _) = dimensions(residual)?;
    let n = residual_rows
        .checked_mul(residual_cols)
        .ok_or_else(overflow)?;
    let mut scratch = 0u64;
    for value in [Some(beta), Some(residual), Some(source), alpha]
        .into_iter()
        .flatten()
    {
        add(&mut scratch, resident_download(value)?)?;
    }
    for value in [beta, residual, source] {
        add(&mut scratch, floating_conversion(value)?)?;
    }
    // The index mask and standard errors coexist with the compact Jacobian.
    add(&mut scratch, bytes(n, size_of::<usize>())?)?;
    add(&mut scratch, bytes(p, size_of::<f64>())?)?;
    if method == Method::Jacobian {
        let cells = n.checked_mul(p).ok_or_else(overflow)?;
        let square = p.checked_mul(p).ok_or_else(overflow)?;
        // DMatrix owns the packed J. Nalgebra's bidiagonal/SVD factors may
        // coexist with it; this intentionally overcharges their O(np+p²)
        // floating workspace, without claiming an exact process-RSS bound.
        add(&mut scratch, bytes(cells, 64)?)?;
        add(&mut scratch, bytes(square, 64)?)?;
        add(&mut scratch, bytes(n.max(p), 64)?)?;
    }
    let retained = if produces_output {
        HostAllocationPlan::retained_tensor_bytes(dtype, &[p, 2]).map_err(map_plan_error)?
    } else {
        0
    };
    HostAllocationPlan::new(scratch, retained).map_err(map_plan_error)
}

fn map_plan_error(error: HostAllocationError) -> RuntimeError {
    let mut result = overflow();
    result.source = Some(Box::new(error));
    result
}
