//! Final exact output owners for the shared managed CARE/DARE path.
//!
//! Numerical work and validation retain their scratch claims until these
//! canonical Value tensors have been filled and the receipt is checkpointed.

use num_complex::Complex64;
use runmat_value::{
    ExactHostComplexTensorAllocation, ExactHostNumericAllocationError, ExactHostTensorAllocation,
    NumericDType, OutputListInitializationError, OutputListStorage, Value,
};

use super::{RiccatiApi, RiccatiIssue};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

const CHECKPOINT_ELEMENTS: usize = 4096;

pub(super) struct ManagedRiccatiOutputView<'a> {
    pub(super) gain: &'a [f64],
    pub(super) gain_shape: [usize; 2],
    pub(super) solution: &'a [f64],
    pub(super) solution_shape: [usize; 2],
    pub(super) poles: &'a [Complex64],
}

#[derive(Clone, Copy)]
enum OutputKind {
    Gain,
    Solution,
    Poles,
}

fn output_kind(api: RiccatiApi, index: usize) -> OutputKind {
    match (api, index) {
        (RiccatiApi::Care, 0) => OutputKind::Solution,
        (RiccatiApi::Care, 1) => OutputKind::Poles,
        (RiccatiApi::Care, 2) => OutputKind::Gain,
        (_, 0) => OutputKind::Gain,
        (_, 1) => OutputKind::Solution,
        (_, 2) => OutputKind::Poles,
        _ => unreachable!("validated output count"),
    }
}

impl ManagedRiccatiOutputView<'_> {
    /// Pole class is unknown before solving. Admit the exact larger of the
    /// two canonical layouts; the unused typed headroom is refunded.
    pub(super) fn retained_upper_bytes(
        api: RiccatiApi,
        count: usize,
        n: usize,
        m: usize,
    ) -> Result<u64, HostAllocationError> {
        if count > 3 {
            return Err(HostAllocationError::Overflow);
        }
        let real_poles = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[n, 1])?;
        let complex_poles =
            HostAllocationPlan::retained_complex_tensor_bytes(NumericDType::F64, &[n, 1])?;
        let mut bytes = 0_u64;
        for index in 0..count {
            let next = match output_kind(api, index) {
                OutputKind::Gain => {
                    HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[m, n])?
                }
                OutputKind::Solution => {
                    HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[n, n])?
                }
                OutputKind::Poles => real_poles.max(complex_poles),
            };
            bytes = bytes
                .checked_add(next)
                .ok_or(HostAllocationError::Overflow)?;
        }
        bytes
            .checked_add(output_list_bytes(count)?)
            .ok_or(HostAllocationError::Overflow)
    }

    fn real_poles(&self) -> bool {
        self.poles
            .iter()
            .all(|pole| pole.im.abs() <= crate::builtins::control::tf_model::EPS)
    }

    fn check_lengths(&self, api: RiccatiApi) -> BuiltinResult<()> {
        for (label, shape, values) in [
            ("gain", self.gain_shape, self.gain),
            ("solution", self.solution_shape, self.solution),
        ] {
            if shape[0].checked_mul(shape[1]) != Some(values.len()) {
                return Err(api.error(
                    RiccatiIssue::Internal,
                    format!("{label} shape changed before Riccati publication"),
                ));
            }
        }
        if self.poles.len() != self.solution_shape[0] {
            return Err(api.error(
                RiccatiIssue::Internal,
                "closed-loop pole count changed before Riccati publication",
            ));
        }
        Ok(())
    }

    pub(super) fn publish(
        &self,
        api: RiccatiApi,
        count: usize,
        receipt: &HostAllocationReceipt,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<OutputListStorage> {
        self.check_lengths(api)?;
        if count > 3 {
            return Err(api.error(RiccatiIssue::Internal, "too many Riccati outputs"));
        }
        let mut outputs = receipt
            .with_output_allocation(runtime, |permit| {
                OutputListStorage::try_initialize(count, permit, |index| {
                    let value = match output_kind(api, index) {
                        OutputKind::Solution => real(self.solution_shape, permit)?,
                        OutputKind::Gain => real(self.gain_shape, permit)?,
                        OutputKind::Poles if self.real_poles() => {
                            real([self.poles.len(), 1], permit)?
                        }
                        OutputKind::Poles => complex_column(self.poles.len(), permit)?,
                    };
                    Ok(value)
                })
                .map_err(|error| match error {
                    OutputListInitializationError::Allocation(error)
                    | OutputListInitializationError::Element(error) => error,
                })
            })
            .map_err(|error| api.admission(Some(runtime), error))?;
        let values = outputs.try_as_mut_slice_unique().ok_or_else(|| {
            api.error(
                RiccatiIssue::Internal,
                "Riccati output owner lost uniqueness",
            )
        })?;
        for (index, value) in values.iter_mut().enumerate() {
            match output_kind(api, index) {
                OutputKind::Solution => fill_real(value, self.solution, receipt, runtime, api)?,
                OutputKind::Gain => fill_real(value, self.gain, receipt, runtime, api)?,
                OutputKind::Poles if self.real_poles() => {
                    fill_real_poles(value, self.poles, receipt, runtime, api)?
                }
                OutputKind::Poles => fill_poles(value, self.poles, receipt, runtime, api)?,
            }
        }
        receipt
            .checkpoint(runtime)
            .map_err(|error| api.admission(Some(runtime), error))?;
        Ok(outputs)
    }
}

fn output_list_bytes(count: usize) -> Result<u64, HostAllocationError> {
    let list_raw =
        OutputListStorage::requested_bytes(count).map_err(HostAllocationError::Numeric)?;
    let list_raw = u64::try_from(list_raw).map_err(|_| HostAllocationError::Overflow)?;
    HostAllocationPlan::retained_exact_output_bytes(list_raw, usize::from(count != 0))
}

fn real(
    shape: [usize; 2],
    permit: &dyn runmat_value::ExactHostAllocationPermit,
) -> Result<Value, ExactHostNumericAllocationError> {
    let output = ExactHostTensorAllocation::new(NumericDType::F64, &shape, permit)?;
    output.into_tensor().map(Value::Tensor)
}

fn complex_column(
    len: usize,
    permit: &dyn runmat_value::ExactHostAllocationPermit,
) -> Result<Value, ExactHostNumericAllocationError> {
    let output = ExactHostComplexTensorAllocation::new(NumericDType::F64, &[len, 1], permit)?;
    output.into_tensor().map(Value::ComplexTensor)
}

fn fill_real(
    target: &mut Value,
    source: &[f64],
    receipt: &HostAllocationReceipt,
    runtime: &RuntimeContext,
    api: RiccatiApi,
) -> BuiltinResult<()> {
    let Value::Tensor(tensor) = target else {
        return Err(api.error(RiccatiIssue::Internal, "Riccati real output class changed"));
    };
    let values = tensor
        .try_as_f64_slice_mut_unique()
        .ok_or_else(|| api.error(RiccatiIssue::Internal, "Riccati real output is not unique"))?;
    for (to, from) in values
        .chunks_mut(CHECKPOINT_ELEMENTS)
        .zip(source.chunks(CHECKPOINT_ELEMENTS))
    {
        receipt
            .checkpoint(runtime)
            .map_err(|error| api.admission(Some(runtime), error))?;
        if from.iter().any(|value| !value.is_finite()) {
            return Err(api.error(
                RiccatiIssue::SolverFailed,
                "Riccati real output is nonfinite",
            ));
        }
        to.copy_from_slice(from);
    }
    Ok(())
}

fn fill_poles(
    target: &mut Value,
    source: &[Complex64],
    receipt: &HostAllocationReceipt,
    runtime: &RuntimeContext,
    api: RiccatiApi,
) -> BuiltinResult<()> {
    let Value::ComplexTensor(tensor) = target else {
        return Err(api.error(RiccatiIssue::Internal, "Riccati pole output class changed"));
    };
    let values = tensor
        .try_as_f64_slice_mut_unique()
        .ok_or_else(|| api.error(RiccatiIssue::Internal, "Riccati pole output is not unique"))?;
    for (to, from) in values
        .chunks_mut(CHECKPOINT_ELEMENTS)
        .zip(source.chunks(CHECKPOINT_ELEMENTS))
    {
        receipt
            .checkpoint(runtime)
            .map_err(|error| api.admission(Some(runtime), error))?;
        for (slot, pole) in to.iter_mut().zip(from) {
            if !pole.re.is_finite() || !pole.im.is_finite() {
                return Err(api.error(
                    RiccatiIssue::SolverFailed,
                    "Riccati pole output is nonfinite",
                ));
            }
            *slot = runmat_value::ComplexElement(pole.re, pole.im);
        }
    }
    Ok(())
}

fn fill_real_poles(
    target: &mut Value,
    source: &[Complex64],
    receipt: &HostAllocationReceipt,
    runtime: &RuntimeContext,
    api: RiccatiApi,
) -> BuiltinResult<()> {
    let Value::Tensor(tensor) = target else {
        return Err(api.error(RiccatiIssue::Internal, "Riccati pole output class changed"));
    };
    let values = tensor
        .try_as_f64_slice_mut_unique()
        .ok_or_else(|| api.error(RiccatiIssue::Internal, "Riccati pole output is not unique"))?;
    for (to, from) in values
        .chunks_mut(CHECKPOINT_ELEMENTS)
        .zip(source.chunks(CHECKPOINT_ELEMENTS))
    {
        receipt
            .checkpoint(runtime)
            .map_err(|error| api.admission(Some(runtime), error))?;
        for (slot, pole) in to.iter_mut().zip(from) {
            if !pole.re.is_finite() || !pole.im.is_finite() {
                return Err(api.error(
                    RiccatiIssue::SolverFailed,
                    "Riccati pole output is nonfinite",
                ));
            }
            *slot = pole.re;
        }
    }
    Ok(())
}
