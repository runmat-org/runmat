//! Checked, source-shape-derived host geometry for state polynomial work.
use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};
use num_complex::Complex64;
use runmat_value::{NumericDType, Value};

use super::{conversion_error, ConversionErrorKind, ConversionResult};
use crate::BuiltinResult;

pub(crate) struct SsPolynomialGeometry {
    pub(super) states: usize,
    pub(super) inputs: usize,
    pub(super) outputs: usize,
}

impl SsPolynomialGeometry {
    pub(in crate::builtins::control) fn states(&self) -> usize {
        self.states
    }

    pub(in crate::builtins::control) fn inputs(&self) -> usize {
        self.inputs
    }

    pub(crate) fn from_values(values: [&Value; 4], builtin: &'static str) -> BuiltinResult<Self> {
        Self::from_values_checked(values).map_err(|error| error.into_runtime(builtin))
    }

    fn from_values_checked(values: [&Value; 4]) -> ConversionResult<Self> {
        let [a, b, c, d] = values.map(matrix_extent);
        let a = a?;
        let b = b?;
        let c = c?;
        let d = d?;
        if a.0 != a.1 || b.0 != a.0 || c.1 != a.0 || d != (c.0, b.1) {
            return Err(conversion_error(
                "state matrices have incompatible dimensions",
                ConversionErrorKind::InvalidModel,
            ));
        }
        Ok(Self {
            states: a.0,
            inputs: b.1,
            outputs: c.0,
        })
    }

    /// Selected input: all output rows. The coefficient grid remains scratch
    /// until construction of the exact retained numeric result.
    pub(crate) fn selected_input_plan(
        &self,
        dtype: NumericDType,
        requested: usize,
    ) -> Result<HostAllocationPlan, HostAllocationError> {
        let degree = self
            .states
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let square = self
            .states
            .checked_mul(self.states)
            .ok_or(HostAllocationError::Overflow)?;
        let complex_bytes = |len: usize| -> Result<u64, HostAllocationError> {
            let bytes = len
                .checked_mul(std::mem::size_of::<Complex64>())
                .ok_or(HostAllocationError::Overflow)?;
            u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)
        };
        let add = |a: u64, b: u64| a.checked_add(b).ok_or(HostAllocationError::Overflow);
        let denominator = complex_bytes(degree)?;
        let numerator_payload = complex_bytes(
            self.outputs
                .checked_mul(degree)
                .ok_or(HostAllocationError::Overflow)?,
        )?;
        let numerator_headers = u64::try_from(
            self.outputs
                .checked_mul(std::mem::size_of::<Vec<Complex64>>())
                .ok_or(HostAllocationError::Overflow)?,
        )
        .map_err(|_| HostAllocationError::Overflow)?;
        let retained_coefficients = add(add(denominator, numerator_payload)?, numerator_headers)?;
        let buffers = if self.states == 0 {
            0
        } else {
            add(
                HostAllocationPlan::complex_workspace_bytes(square)?,
                HostAllocationPlan::complex_workspace_bytes(square)?,
            )?
        };
        let scratch = add(retained_coefficients, buffers)?;
        let output = if requested == 0 {
            0
        } else {
            let numerator =
                HostAllocationPlan::retained_tensor_bytes(dtype, &[self.outputs, degree])?;
            let denominator = if requested >= 2 {
                HostAllocationPlan::retained_tensor_bytes(dtype, &[1, degree])?
            } else {
                0
            };
            add(numerator, denominator)?
        };
        HostAllocationPlan::new(scratch, output)
    }

    /// Coefficient-grid scratch for all input/output channels. Published
    /// CellArray backing is deliberately outside this numeric-stage plan.
    pub(crate) fn coefficient_grid_plan(&self) -> Result<HostAllocationPlan, HostAllocationError> {
        let degree = self
            .states
            .checked_add(1)
            .ok_or(HostAllocationError::Overflow)?;
        let channels = self
            .inputs
            .checked_mul(self.outputs)
            .ok_or(HostAllocationError::Overflow)?;
        let coefficients = channels
            .checked_mul(degree)
            .and_then(|count| count.checked_add(degree))
            .ok_or(HostAllocationError::Overflow)?;
        let coefficient_bytes = coefficients
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(HostAllocationError::Overflow)?;
        let headers = channels
            .checked_mul(std::mem::size_of::<Vec<Complex64>>())
            .ok_or(HostAllocationError::Overflow)?;
        let square = self
            .states
            .checked_mul(self.states)
            .ok_or(HostAllocationError::Overflow)?;
        let workspaces = if self.states == 0 {
            0
        } else {
            HostAllocationPlan::complex_workspace_bytes(square)?
                .checked_mul(2)
                .ok_or(HostAllocationError::Overflow)?
        };
        let scratch = u64::try_from(coefficient_bytes)
            .ok()
            .and_then(|count| count.checked_add(u64::try_from(headers).ok()?))
            .and_then(|count| count.checked_add(workspaces))
            .ok_or(HostAllocationError::Overflow)?;
        HostAllocationPlan::new(scratch, 0)
    }
}

fn matrix_extent(value: &Value) -> ConversionResult<(usize, usize)> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::ComplexTensor(tensor) => tensor.shape(),
        Value::GpuTensor(handle) => handle.shape(),
        Value::LogicalArray(array) => &array.shape,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => return Ok((1, 1)),
        _ => {
            return Err(conversion_error(
                "state matrix must be numeric",
                ConversionErrorKind::InvalidModel,
            ))
        }
    };
    if shape.len() > 2 {
        return Err(conversion_error(
            "state matrix must be 2-D",
            ConversionErrorKind::InvalidModel,
        ));
    }
    Ok(match shape {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns] => (*rows, *columns),
        _ => unreachable!(),
    })
}
