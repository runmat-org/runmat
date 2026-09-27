//! Provider reductions used by plotting, before registry mutation.
use super::{plotting_error, value_to_scalar_async};
use crate::builtins::common::gpu_helpers as gpu;
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

#[cfg(test)]
#[path = "gpu_bounds_tests.rs"]
mod tests;

struct Temporaries<'a> {
    provider: &'static dyn AccelProvider,
    protected: &'a [&'a GpuTensorHandle],
    handles: Vec<GpuTensorHandle>,
}

impl Drop for Temporaries<'_> {
    fn drop(&mut self) {
        for handle in &self.handles {
            gpu::free_rejected_provider_output(handle, self.protected, self.provider);
        }
    }
}

impl Temporaries<'_> {
    fn admit(
        &mut self,
        output: GpuTensorHandle,
        input: &GpuTensorHandle,
        context: &'static str,
    ) -> BuiltinResult<GpuTensorHandle> {
        let protected_alias = self
            .protected
            .iter()
            .any(|value| gpu::same_gpu_handle(value, &output));
        let prior_alias = self
            .handles
            .iter()
            .any(|value| gpu::same_gpu_handle(value, &output));
        let valid = !protected_alias
            && !prior_alias
            && output
                .shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                == Some(1)
            && output.device_id == self.provider.device_id()
            && gpu::exact_provider_for_handle(&output)
                .is_some_and(|owner| std::ptr::eq(owner, self.provider))
            && gpu::expected_handle_numeric_element_type(input).is_ok_and(|element| {
                gpu::numeric_descriptor_matches(&output, element, GpuTensorStorage::Real)
            })
            && gpu::gpu_class_metadata_matches(
                &output,
                runmat_accelerate_api::handle_precision(input),
                runmat_accelerate_api::handle_integer_type(input),
                false,
            );
        // Own every fresh result immediately, including malformed metadata.
        // Aliases are never adopted and never freed.
        if !protected_alias && !prior_alias {
            self.handles.push(output.clone());
        }
        if valid {
            Ok(output)
        } else {
            Err(plotting_error(
                context,
                format!("{context}: invalid provider bounds result"),
            ))
        }
    }
}

pub(super) async fn axis_bounds_protecting(
    input: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    context: &'static str,
) -> BuiltinResult<(f32, f32)> {
    let provider = gpu::exact_provider_for_handle(input).ok_or_else(|| {
        plotting_error(context, format!("{context}: no provider owns bounds input"))
    })?;
    let _metadata = protected
        .iter()
        .map(|handle| gpu::HandleMetadataRestoreGuard::new(handle))
        .collect::<Vec<_>>();
    let mut owned = Temporaries {
        provider,
        protected,
        handles: Vec::with_capacity(2),
    };
    let minimum = provider.reduce_min(input).await.map_err(|error| {
        plotting_error(context, format!("{context}: reduce_min failed: {error}"))
    })?;
    let minimum = owned.admit(minimum, input, context)?;
    let maximum = provider.reduce_max(input).await.map_err(|error| {
        plotting_error(context, format!("{context}: reduce_max failed: {error}"))
    })?;
    let maximum = owned.admit(maximum, input, context)?;
    let minimum = value_to_scalar_async(Value::GpuTensor(minimum), context).await?;
    let maximum = value_to_scalar_async(Value::GpuTensor(maximum), context).await?;
    Ok((minimum as f32, maximum as f32))
}
