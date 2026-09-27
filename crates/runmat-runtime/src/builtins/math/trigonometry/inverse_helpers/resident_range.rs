//! Range evidence for real inverse-trigonometric provider kernels.
use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(crate) enum RealDomain {
    ClosedUnitInterval,
    AtLeastOne,
}

fn error(builtin: &str, detail: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("{builtin}: {detail}"))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:Internal"))
        .build()
}

struct Reductions<'a> {
    input: &'a GpuTensorHandle,
    owned: Vec<GpuTensorHandle>,
}

impl Reductions<'_> {
    fn adopt(
        &mut self,
        provider: &'static dyn AccelProvider,
        output: GpuTensorHandle,
        builtin: &str,
    ) -> BuiltinResult<GpuTensorHandle> {
        let protected: Vec<_> = std::iter::once(self.input).chain(&self.owned).collect();
        let contract = gpu_helpers::BinaryGpuOutputContract {
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(self.input),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        };
        let valid = !protected
            .iter()
            .any(|handle| gpu_helpers::same_gpu_handle(handle, &output))
            && gpu_helpers::binary_gpu_output_matches(
                &output, self.input, self.input, provider, &contract,
            )
            && gpu_helpers::expected_handle_numeric_element_type(&output).is_ok_and(|element| {
                gpu_helpers::numeric_descriptor_matches(&output, element, GpuTensorStorage::Real)
            });
        if !valid {
            gpu_helpers::free_rejected_provider_output(&output, &protected, provider);
            return Err(error(
                builtin,
                "provider returned malformed or aliased range reduction",
            ));
        }
        self.owned.push(output.clone());
        Ok(output)
    }
}

impl Drop for Reductions<'_> {
    fn drop(&mut self) {
        for output in &self.owned {
            gpu_helpers::free_unprotected_exact_owner(output, &[self.input]);
        }
    }
}

/// None means no conclusive real-domain proof, not successful provider execution.
/// Empty inputs, NaN reductions and explicitly unsupported hooks use the existing
/// host computation route. Other failures propagate; all accepted temporaries
/// remain owned across every suspension point and are reclaimed on abandonment.
pub(crate) async fn requires_complex(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    domain: RealDomain,
    builtin: &str,
) -> BuiltinResult<Option<bool>> {
    if input.shape.contains(&0) {
        return Ok(None);
    }
    let owner = gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| error(builtin, "GPU input has no owning provider"))?;
    if !std::ptr::eq(owner, provider)
        || !gpu_helpers::expected_handle_numeric_element_type(input).is_ok_and(|element| {
            gpu_helpers::numeric_descriptor_matches(input, element, GpuTensorStorage::Real)
        })
    {
        return Err(error(
            builtin,
            "range reduction requires an owned real numeric input",
        ));
    }
    let mut outputs = Reductions {
        input,
        owned: Vec::new(),
    };
    match provider.reduce_min(input).await {
        Ok(output) => outputs.adopt(provider, output, builtin)?,
        Err(failure) if gpu_helpers::provider_hook_is_unsupported(&failure) => return Ok(None),
        Err(failure) => return Err(error(builtin, format!("reduce_min failed: {failure}"))),
    };
    if matches!(domain, RealDomain::ClosedUnitInterval) {
        match provider.reduce_max(input).await {
            Ok(output) => {
                outputs.adopt(provider, output, builtin)?;
            }
            Err(failure) if gpu_helpers::provider_hook_is_unsupported(&failure) => return Ok(None),
            Err(failure) => return Err(error(builtin, format!("reduce_max failed: {failure}"))),
        }
    }
    let mut outside = false;
    for output in &outputs.owned {
        let host = gpu_helpers::download_native_values_async(provider, output)
            .await
            .map_err(|failure| {
                crate::builtins::common::map_control_flow_with_builtin(failure, builtin)
            })?;
        if host.data.len() != 1 {
            return Err(error(
                builtin,
                "range reduction did not download one scalar",
            ));
        }
        let value = &host.data[0];
        if value.is_nan() {
            return Ok(None);
        }
        outside |= match domain {
            RealDomain::ClosedUnitInterval => value.is_outside_closed_unit_interval(0.0),
            RealDomain::AtLeastOne => value.is_less_than_one(),
        };
    }
    Ok(Some(outside))
}
