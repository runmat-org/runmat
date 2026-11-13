use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, AccelProvider, GpuTensorHandle,
    ProviderPrecision,
};
use runmat_builtins::{COMPLEX_ERROR_INTERNAL, COMPLEX_ERROR_INVALID_INPUT};
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::super::{error, host};

/// Domain admission is separate from uploading: both operands and their shape
/// relationship must be accepted before allocating anything on a provider.
pub(super) enum RealOperand<'a> {
    Resident(&'a GpuTensorHandle),
    Host(Tensor),
}

impl<'a> RealOperand<'a> {
    pub(super) fn admit(value: &'a Value) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => Ok(Self::Resident(handle)),
            value => host::from_value(value.clone()).map(|input| Self::Host(input.tensor)),
        }
    }

    pub(super) fn shape(&self) -> &[usize] {
        match self {
            Self::Resident(handle) => handle.shape(),
            Self::Host(tensor) => tensor.shape(),
        }
    }
}

pub(super) struct RealGpuOperand {
    pub(super) handle: GpuTensorHandle,
}

impl RealGpuOperand {
    pub(super) fn from_admitted(
        value: RealOperand<'_>,
        provider: &'static dyn AccelProvider,
        protected: &[&GpuTensorHandle],
    ) -> BuiltinResult<Self> {
        match value {
            RealOperand::Resident(handle) => {
                // The binary root already fenced every original source before
                // uploading either operand. This later integrity check must
                // never grant a whole-call retry after a sibling upload.
                let owner = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
                    error(
                        &COMPLEX_ERROR_INVALID_INPUT,
                        "GPU input provider is unavailable",
                    )
                })?;
                if !runmat_accelerate_api::same_provider_instance(owner, provider)
                    || handle.device_id() != provider.device_id()
                {
                    return Err(error(
                        &COMPLEX_ERROR_INVALID_INPUT,
                        "GPU inputs must belong to the same provider",
                    ));
                }
                Ok(Self {
                    handle: handle.clone(),
                })
            }
            RealOperand::Host(input) => {
                let _metadata: Vec<_> = protected
                    .iter()
                    .map(|handle| gpu_helpers::ResidencyRestoreGuard::new(handle))
                    .collect();
                let handle = gpu_helpers::upload_tensor_protecting(provider, &input, protected)
                    .map_err(|source| {
                        let mut failure = error(&COMPLEX_ERROR_INTERNAL, source.to_string());
                        failure.source = Some(Box::new(source));
                        failure
                    })?;
                Ok(Self { handle })
            }
        }
    }
}

pub(super) fn requires_exact_host_path(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some() || handle_is_logical(handle))
}

pub(super) fn floating_result_precision(
    real: &Value,
    imaginary: &Value,
) -> Option<ProviderPrecision> {
    if is_single(real) || is_single(imaginary) {
        Some(ProviderPrecision::F32)
    } else {
        Some(ProviderPrecision::F64)
    }
}

fn is_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if handle_precision(handle) == Some(ProviderPrecision::F32))
}

#[cfg(test)]
mod lease_tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };

    #[test]
    fn operand_drop_preserves_borrowed_alias_and_retires_last_upload_owner() {
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let borrowed = RealGpuOperand {
            handle: allocation.handle.clone(),
        };
        drop(borrowed);
        assert_eq!(allocation.probe.retirements(), 0);
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        let owned = RealGpuOperand {
            handle: allocation.handle,
        };
        drop(owned);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
