use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::gpu_helpers::{
    BinaryGpuOutputContract, GpuOutputAliasPolicy, UnaryGpuOutputContract,
};
use crate::BuiltinResult;

pub(super) async fn try_power(base: &Value, exponent: &Value) -> BuiltinResult<Option<Value>> {
    let Value::GpuTensor(base) = base else {
        return Ok(None);
    };
    if !compatible_base(base) {
        return Ok(None);
    }
    let provider = gpu_helpers::exact_provider_for_handle(base).ok_or_else(|| {
        super::errors::internal("mpower: resident base has no exact provider owner")
    })?;
    let Some(exponent) = super::exponent::parse(exponent)? else {
        return Ok(None);
    };
    if exponent < 0 {
        return Err(super::errors::invalid_argument(
            "Negative matrix powers not supported yet",
        ));
    }
    let [rows, columns] = base.shape.as_slice() else {
        return Ok(None);
    };
    if rows != columns {
        return Err(super::errors::invalid_input(format!(
            "Matrix must be square for matrix power: {rows}x{columns}"
        )));
    }
    match exponent {
        0 => identity(provider, base).map(|result| result.map(gpu_helpers::resident_gpu_value)),
        1 => Ok(Some(Value::GpuTensor(base.clone()))),
        exponent => binary_power(provider, base, exponent as u32)
            .await
            .map(|handle| handle.map(gpu_helpers::resident_gpu_value)),
    }
}

fn identity(
    provider: &'static dyn AccelProvider,
    prototype: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let output = match provider.eye_like(prototype) {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
            let identity = crate::builtins::common::matrix::matrix_eye(prototype.shape[0]);
            return gpu_helpers::upload_tensor(provider, &identity)
                .map(Some)
                .map_err(|error| {
                    super::errors::internal(format!(
                        "mpower: failed to upload provider identity: {error}"
                    ))
                });
        }
        Err(error) => {
            return Err(super::errors::internal(format!(
                "mpower: provider identity construction failed: {error}"
            )))
        }
    };
    let contract = UnaryGpuOutputContract {
        storage: GpuTensorStorage::Real,
        precision: Some(provider.precision()),
        integer: None,
        logical: false,
        alias: GpuOutputAliasPolicy::RequireDistinct,
    };
    if gpu_helpers::unary_gpu_output_matches(&output, prototype, provider, contract) {
        return Ok(Some(output));
    }
    gpu_helpers::free_rejected_provider_output(&output, &[prototype], provider);
    Err(super::errors::internal(
        "mpower: provider returned invalid identity metadata",
    ))
}

async fn binary_power(
    provider: &'static dyn AccelProvider,
    base: &GpuTensorHandle,
    mut exponent: u32,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let mut base_state = ManagedHandle::borrowed(provider, base);
    let mut result: Option<ManagedHandle> = None;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = Some(match result {
                Some(current) => {
                    let Some(product) =
                        multiply(provider, current.handle(), base_state.handle()).await?
                    else {
                        return Ok(None);
                    };
                    product
                }
                None => base_state.transfer_clone(),
            });
        }
        exponent >>= 1;
        if exponent > 0 {
            let Some(square) = multiply(provider, base_state.handle(), base_state.handle()).await?
            else {
                return Ok(None);
            };
            base_state = square;
        }
    }
    result
        .map(|handle| Some(handle.into_handle()))
        .ok_or_else(|| super::errors::internal("mpower: empty provider power result"))
}

async fn multiply(
    provider: &'static dyn AccelProvider,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> BuiltinResult<Option<ManagedHandle>> {
    let output = match provider.matmul(lhs, rhs).await {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {
            return Ok(None);
        }
        Err(error) => {
            return Err(super::errors::internal(format!(
                "mpower: provider matrix multiplication failed: {error}"
            )))
        }
    };
    let contract = BinaryGpuOutputContract {
        shape: lhs.shape.clone(),
        storage: GpuTensorStorage::Real,
        precision: Some(provider.precision()),
        integer: None,
        logical: false,
        alias: GpuOutputAliasPolicy::RequireDistinct,
    };
    if gpu_helpers::binary_gpu_output_matches(&output, lhs, rhs, provider, &contract) {
        return Ok(Some(ManagedHandle::owned(provider, output)));
    }
    gpu_helpers::free_rejected_provider_output(&output, &[lhs, rhs], provider);
    Err(super::errors::internal(
        "mpower: provider returned invalid matrix-product metadata",
    ))
}

fn compatible_base(handle: &GpuTensorHandle) -> bool {
    let Some(provider) = gpu_helpers::exact_provider_for_handle(handle) else {
        return false;
    };
    runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle) == Some(provider.precision())
}

struct ManagedHandle {
    provider: &'static dyn AccelProvider,
    handle: GpuTensorHandle,
    owned: bool,
}

impl ManagedHandle {
    fn borrowed(provider: &'static dyn AccelProvider, handle: &GpuTensorHandle) -> Self {
        Self {
            provider,
            handle: handle.clone(),
            owned: false,
        }
    }

    fn owned(provider: &'static dyn AccelProvider, handle: GpuTensorHandle) -> Self {
        Self {
            provider,
            handle,
            owned: true,
        }
    }

    fn transfer_clone(&mut self) -> Self {
        let owned = std::mem::replace(&mut self.owned, false);
        Self {
            provider: self.provider,
            handle: self.handle.clone(),
            owned,
        }
    }

    fn handle(&self) -> &GpuTensorHandle {
        &self.handle
    }

    fn into_handle(mut self) -> GpuTensorHandle {
        self.owned = false;
        self.handle.clone()
    }
}

impl Drop for ManagedHandle {
    fn drop(&mut self) {
        if self.owned && self.provider.free(&self.handle).is_ok() {
            runmat_accelerate_api::clear_handle_metadata(&self.handle);
        }
    }
}
