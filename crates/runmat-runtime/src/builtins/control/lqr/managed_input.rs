//! Claimed, move-only real matrix inputs for the managed Riccati worker.
//!
//! This is deliberately separate from the ordinary `DMatrix` parser. The
//! source `Value` is read once into the exact scratch owner; no intermediate
//! tensor, floating-point `Vec`, or nalgebra matrix is constructed.

use runmat_value::Value;

use super::{RiccatiApi, RiccatiIssue};
use crate::builtins::control::tf_model::state_matrix_dimensions;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

const CHECKPOINT_ELEMENTS: usize = 4096;

pub(super) struct ManagedRealMatrix {
    rows: usize,
    cols: usize,
    values: HostTypedWorkspace<f64>,
}

impl ManagedRealMatrix {
    pub(super) fn quoted_bytes(rows: usize, cols: usize) -> Result<u64, HostAllocationError> {
        let len = rows
            .checked_mul(cols)
            .ok_or(HostAllocationError::Overflow)?;
        HostTypedWorkspace::<f64>::quoted_bytes(len)
    }

    /// Source residency and class are checked by the common caller before
    /// this host-only transfer. The retained stage receipt is the sole owner
    /// of the exact scratch claim, including across the blocking worker.
    pub(super) fn from_host_value(
        label: &'static str,
        source: Value,
        rows: usize,
        cols: usize,
        receipt: &HostAllocationReceipt,
        runtime: &RuntimeContext,
        api: RiccatiApi,
    ) -> BuiltinResult<Self> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| api.admission(Some(runtime), HostAllocationError::Overflow))?;
        let mut values = receipt
            .temporary_typed::<f64>(runtime, len)
            .map_err(|error| api.admission(Some(runtime), error))?;
        let destination = values.as_mut_slice();
        match &source {
            Value::Tensor(tensor) => {
                if tensor.shape().len() > 2
                    || state_matrix_dimensions(tensor) != (rows, cols)
                    || tensor.len() != len
                {
                    return Err(api.error(
                        RiccatiIssue::InvalidDimensions,
                        format!("{label} shape changed during Riccati preparation"),
                    ));
                }
                if let Some(native) = tensor.as_f64_slice() {
                    for (target, chunk) in destination
                        .chunks_mut(CHECKPOINT_ELEMENTS)
                        .zip(native.chunks(CHECKPOINT_ELEMENTS))
                    {
                        receipt
                            .checkpoint(runtime)
                            .map_err(|error| api.admission(Some(runtime), error))?;
                        if chunk.iter().any(|value| !value.is_finite()) {
                            return Err(api.error(
                                RiccatiIssue::UnsupportedInput,
                                format!("{label} must contain finite real values"),
                            ));
                        }
                        target.copy_from_slice(chunk);
                    }
                } else {
                    for (start, chunk) in destination.chunks_mut(CHECKPOINT_ELEMENTS).enumerate() {
                        receipt
                            .checkpoint(runtime)
                            .map_err(|error| api.admission(Some(runtime), error))?;
                        for (offset, target) in chunk.iter_mut().enumerate() {
                            let index = start * CHECKPOINT_ELEMENTS + offset;
                            let value = tensor
                                .numeric_value_at(index)
                                .ok_or_else(|| {
                                    api.error(
                                        RiccatiIssue::InvalidDimensions,
                                        format!(
                                            "{label} storage changed during Riccati preparation"
                                        ),
                                    )
                                })?
                                .materialize_f64();
                            if !value.is_finite() {
                                return Err(api.error(
                                    RiccatiIssue::UnsupportedInput,
                                    format!("{label} must contain finite real values"),
                                ));
                            }
                            *target = value;
                        }
                    }
                }
            }
            Value::Num(number) if len == 1 && number.is_finite() => destination[0] = *number,
            Value::Int(integer) if len == 1 => destination[0] = integer.to_f64(),
            _ => {
                return Err(api.error(
                    RiccatiIssue::UnsupportedInput,
                    format!("{label} must be a real numeric host matrix"),
                ))
            }
        }
        receipt
            .checkpoint(runtime)
            .map_err(|error| api.admission(Some(runtime), error))?;
        Ok(Self { rows, cols, values })
    }

    pub(super) fn into_parts(self) -> (usize, usize, HostTypedWorkspace<f64>) {
        (self.rows, self.cols, self.values)
    }

    pub(super) fn implicit(
        rows: usize,
        cols: usize,
        identity: bool,
        receipt: &HostAllocationReceipt,
        runtime: &RuntimeContext,
        api: RiccatiApi,
    ) -> BuiltinResult<Self> {
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| api.admission(Some(runtime), HostAllocationError::Overflow))?;
        let mut values = receipt
            .temporary_typed::<f64>(runtime, len)
            .map_err(|error| api.admission(Some(runtime), error))?;
        if identity {
            if rows != cols {
                return Err(api.error(RiccatiIssue::InvalidDimensions, "implicit R must be square"));
            }
            for diagonal in 0..rows {
                if diagonal % CHECKPOINT_ELEMENTS == 0 {
                    receipt
                        .checkpoint(runtime)
                        .map_err(|error| api.admission(Some(runtime), error))?;
                }
                values.as_mut_slice()[diagonal * rows + diagonal] = 1.0;
            }
        }
        receipt
            .checkpoint(runtime)
            .map_err(|error| api.admission(Some(runtime), error))?;
        Ok(Self { rows, cols, values })
    }
}
