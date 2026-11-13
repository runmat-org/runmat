//! One physical double conversion of original UTF-16 code units.
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::RuntimeError;
use runmat_value::{CharArray, NumericDType, Tensor};

#[derive(Debug)]
pub(crate) enum CharacterNumericError {
    InvalidLayout(String),
    Runtime(Box<RuntimeError>),
}

impl CharacterNumericError {
    pub(crate) fn into_runtime_error(
        self,
        invalid: impl FnOnce(String) -> RuntimeError,
    ) -> RuntimeError {
        match self {
            Self::InvalidLayout(message) => invalid(message),
            Self::Runtime(error) => *error,
        }
    }
}

/// Read MATLAB linear order directly; surrogate units remain separate numbers.
/// The managed owner quotes both final payload and shape before allocating.
pub(crate) fn to_double(
    runtime: Option<&RuntimeContext>,
    chars: &CharArray,
) -> Result<Tensor, CharacterNumericError> {
    chars
        .validate_layout()
        .map_err(CharacterNumericError::InvalidLayout)?;
    // A rank-empty character owner has zero units, whereas a numeric rank-empty
    // shape denotes one element. Do not manufacture an extra numeric value.
    if chars.shape.is_empty() {
        return Err(CharacterNumericError::InvalidLayout(
            "Rank-empty character storage has no numeric scalar element".into(),
        ));
    }
    let allocation_error = |source: HostAllocationError| {
        CharacterNumericError::Runtime(Box::new(host_allocation_runtime_error(
            runtime,
            "RunMat:HostAllocation:CharacterNumeric",
            "character numeric conversion",
            source,
        )))
    };
    let retained = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &chars.shape)
        .map_err(allocation_error)?;
    let plan = HostAllocationPlan::new(0, retained).map_err(allocation_error)?;
    let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(allocation_error)?;
    match receipt {
        Some(receipt) => {
            let runtime = runtime.expect("managed conversion has its original runtime");
            let mut tensor = receipt
                .exact_output_tensor(runtime, NumericDType::F64, &chars.shape)
                .map_err(allocation_error)?
                .into_tensor()
                .map_err(|cause| allocation_error(cause.into()))?;
            let output = tensor
                .try_as_f64_slice_mut_unique()
                .expect("new exact double conversion is unique");
            for (index, (target, unit)) in
                output.iter_mut().zip(chars.iter_column_major()).enumerate()
            {
                if index.is_multiple_of(4096) {
                    receipt.checkpoint(runtime).map_err(allocation_error)?;
                }
                *target = f64::from(unit.raw());
            }
            receipt.checkpoint(runtime).map_err(allocation_error)?;
            Ok(tensor)
        }
        None => {
            let mut data = runmat_value::exact_capacity_vec(chars.data().len())
                .map_err(|cause| allocation_error(cause.into()))?;
            for unit in chars.iter_column_major() {
                data.push(f64::from(unit.raw()));
            }
            Tensor::new(data, chars.shape.clone()).map_err(CharacterNumericError::InvalidLayout)
        }
    }
}
