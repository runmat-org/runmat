//! Polynomial evaluation owns every prepared upload and retains original inputs.
use super::{polyval_error, Mu, NumericArray};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult};
use num_complex::Complex64;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderExecutionError, ProviderPolyvalMu,
    ProviderPolyvalOptions, ProviderPrecision,
};
use runmat_value::{Tensor, Value};

#[cfg(test)]
#[path = "resident_tests.rs"]
mod tests;

pub(super) struct Request {
    owner: Option<&'static dyn AccelProvider>,
    originals: Vec<GpuTensorHandle>,
}

impl Request {
    pub(super) fn new(
        points: &Value,
        coefficients: &Value,
        options: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        gpu_helpers::require_resident_compute_values(&[points, coefficients], false, || {
            polyval_error("polyval: no exact owner for resident input")
        })?;
        let anchor = [points, coefficients]
            .into_iter()
            .find_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle),
                _ => None,
            });
        let owner = anchor
            .map(|handle| {
                gpu_helpers::require_resident_compute_owner(handle, || {
                    polyval_error("polyval: no exact owner for resident input")
                })
            })
            .transpose()?;
        let mut originals = Vec::new();
        for value in [points, coefficients].into_iter().chain(options.iter()) {
            crate::dispatcher::try_visit_gpu_handles(value, &mut |handle| {
                originals.push(handle.clone())
            })?;
        }
        Ok(Self { owner, originals })
    }

    fn native_tensor(&self, data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Tensor> {
        let tensor = match self
            .owner
            .expect("owner admitted before preparation")
            .precision()
        {
            ProviderPrecision::F32 => {
                Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
            }
            ProviderPrecision::F64 => Tensor::new(data, shape),
        };
        tensor.map_err(|error| polyval_error(format!("polyval: {error}")))
    }

    fn upload(
        &self,
        tensor: &Tensor,
        extra: &[&GpuTensorHandle],
    ) -> BuiltinResult<Option<GpuTensorHandle>> {
        let Some(owner) = self.owner else {
            return Ok(None);
        };
        let protected = self
            .originals
            .iter()
            .chain(extra.iter().copied())
            .collect::<Vec<_>>();
        match gpu_helpers::upload_tensor_protecting_typed(owner, tensor, &protected) {
            Ok(handle) => Ok(Some(handle)),
            Err(error) => {
                let error = ProviderExecutionError::from_source(error);
                if error.is_unsupported() {
                    return Ok(None);
                }
                Err(crate::runtime_error::with_provider_execution_source(
                    polyval_error(format!("polyval: GPU upload failed: {error}")),
                    error,
                ))
            }
        }
    }

    fn adopt(&self, output: GpuTensorHandle) -> BuiltinResult<Value> {
        let output = gpu_helpers::propagate_owned_output_provenance(output, self.originals.iter())?;
        gpu_helpers::resident_gpu_value(output)
    }

    pub(super) fn upload_result(&self, tensor: &Tensor) -> BuiltinResult<Option<Value>> {
        if self.owner.is_none() {
            return Ok(None);
        }
        let native =
            self.native_tensor(tensor::tensor_values_f64(tensor), tensor.shape().to_vec())?;
        self.upload(&native, &[])?
            .map(|output| self.adopt(output))
            .transpose()
    }

    pub(super) fn try_evaluate(
        &self,
        coefficients: &[Complex64],
        real: bool,
        inputs: &NumericArray,
        mu: Option<Mu>,
    ) -> BuiltinResult<Option<Value>> {
        let Some(owner) = self.owner else {
            return Ok(None);
        };
        if !real || !inputs.all_real || coefficients.is_empty() || inputs.data.is_empty() {
            return Ok(None);
        }
        let coefficients = self.native_tensor(
            coefficients.iter().map(|v| v.re).collect(),
            vec![1, coefficients.len()],
        )?;
        let points = self.native_tensor(
            inputs.data.iter().map(|v| v.re).collect(),
            inputs.shape.clone(),
        )?;
        let Some(coefficients) = self.upload(&coefficients, &[])? else {
            return Ok(None);
        };
        let Some(points) = self.upload(&points, &[&coefficients])? else {
            return Ok(None);
        };
        let _residency = self
            .originals
            .iter()
            .chain([&coefficients, &points])
            .map(gpu_helpers::ResidencyRestoreGuard::new)
            .collect::<Vec<_>>();
        let options = ProviderPolyvalOptions {
            mu: mu.map(|value| ProviderPolyvalMu {
                mean: value.mean,
                scale: value.scale,
            }),
        };
        let output = match owner.polyval(&coefficients, &points, &options) {
            Ok(output) => output,
            Err(error)
                if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(
                    &error,
                ) =>
            {
                return Ok(None)
            }
            Err(error) => {
                return Err(crate::runtime_error::with_provider_execution_source(
                    polyval_error(format!("polyval: provider execution failed: {error}")),
                    ProviderExecutionError::from(error),
                ))
            }
        };
        let protected = self.originals.iter().chain([&coefficients, &points]);
        if output.shape() != inputs.shape
            || output.physical().storage() != GpuTensorStorage::Real
            || output.physical().element_type() != points.physical().element_type()
            || runmat_accelerate_api::handle_is_logical(&output)
            || runmat_accelerate_api::handle_is_transposed(&output)
            || gpu_helpers::exact_provider_for_handle(&output)
                .is_none_or(|actual| !runmat_accelerate_api::same_provider_instance(actual, owner))
            || protected
                .into_iter()
                .any(|input| output.shares_allocation_with(input))
        {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                build_runtime_error("polyval: invalid provider output")
                    .with_builtin("polyval")
                    .build(),
            ));
        }
        self.adopt(output).map(Some)
    }
}
