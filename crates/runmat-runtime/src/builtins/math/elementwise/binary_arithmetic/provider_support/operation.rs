use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderPrecision};

use super::dispatch::compatible_inputs;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArithmeticPairRoute {
    Real,
    Complex(ProviderPrecision),
}

#[derive(Clone, Copy)]
pub(in crate::builtins::math::elementwise::binary_arithmetic) enum ArithmeticProviderOperation {
    Add,
    Subtract,
    Multiply,
}

impl ArithmeticProviderOperation {
    pub(super) fn pair_route(
        self,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> Option<ArithmeticPairRoute> {
        if compatible_inputs(left, right) {
            return Some(ArithmeticPairRoute::Real);
        }
        // This operation type contains only add/subtract/multiply. Division
        // and power have separate real-only admission owners.
        let storage = [left, right].map(runmat_accelerate_api::handle_storage);
        if !storage.contains(&GpuTensorStorage::ComplexInterleaved)
            || [left, right].iter().any(|handle| {
                runmat_accelerate_api::handle_integer_type(handle).is_some()
                    || runmat_accelerate_api::handle_is_logical(handle)
                    || handle.allocation_lease().validate_live().is_err()
            })
        {
            return None;
        }
        let precision = runmat_accelerate_api::handle_precision(left)?;
        (runmat_accelerate_api::handle_precision(right) == Some(precision))
            .then_some(ArithmeticPairRoute::Complex(precision))
    }

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Add => "plus",
            Self::Subtract => "minus",
            Self::Multiply => "times",
        }
    }

    pub(super) async fn apply(
        self,
        provider: &dyn AccelProvider,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let _left = crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(left);
        let _right = crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(right);
        match self {
            Self::Add => provider.elem_add(left, right).await,
            Self::Subtract => provider.elem_sub(left, right).await,
            Self::Multiply => provider.elem_mul(left, right).await,
        }
    }

    pub(super) fn scalar_left(
        self,
        provider: &dyn AccelProvider,
        right: &GpuTensorHandle,
        scalar: f64,
    ) -> anyhow::Result<GpuTensorHandle> {
        let _right = crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(right);
        match self {
            Self::Add => provider.scalar_add(right, scalar),
            Self::Subtract => provider.scalar_rsub(right, scalar),
            Self::Multiply => provider.scalar_mul(right, scalar),
        }
    }

    pub(super) fn scalar_right(
        self,
        provider: &dyn AccelProvider,
        left: &GpuTensorHandle,
        scalar: f64,
    ) -> anyhow::Result<GpuTensorHandle> {
        let _left = crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(left);
        match self {
            Self::Add => provider.scalar_add(left, scalar),
            Self::Subtract => provider.scalar_sub(left, scalar),
            Self::Multiply => provider.scalar_mul(left, scalar),
        }
    }
}
