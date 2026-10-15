//! Provider reductions used by plotting, before registry mutation.
use super::{plotting_error, value_to_scalar_async};

struct ExpectedOutput {
    element: runmat_accelerate_api::NumericElementType,
    precision: Option<runmat_accelerate_api::ProviderPrecision>,
    integer: Option<runmat_accelerate_api::IntegerElementType>,
}
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

impl Temporaries<'_> {
    fn admit(
        &mut self,
        output: GpuTensorHandle,
        expected: &ExpectedOutput,
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
                .shape()
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                == Some(1)
            && output.device_id() == self.provider.device_id()
            && gpu::exact_provider_for_handle(&output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, self.provider)
            })
            && gpu::numeric_descriptor_matches(&output, expected.element, GpuTensorStorage::Real)
            && gpu::gpu_class_metadata_matches(
                &output,
                expected.precision,
                expected.integer,
                false,
            );
        if valid {
            self.handles.push(output.clone());
            Ok(output)
        } else {
            Err(gpu::reject_gpu_output(
                output,
                plotting_error(
                    context,
                    format!("{context}: invalid provider bounds result"),
                ),
            ))
        }
    }
}

/// Binary64 scalar evidence decoded by the existing scalar owner: F64 is
/// retained and F32 widened exactly. Existing integer plotting conversions are
/// unchanged. This does not upgrade a provider's preceding reduction policy.
/// Rendering projection is deliberately separate from numerical bin geometry.
pub(crate) struct NumericBounds {
    pub min: f64,
    pub max: f64,
}

impl NumericBounds {
    pub(crate) fn rendering(&self) -> (f32, f32) {
        (self.min as f32, self.max as f32)
    }
}

pub(super) async fn axis_bounds_protecting(
    input: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    context: &'static str,
) -> BuiltinResult<(f32, f32)> {
    numeric_bounds_protecting(input, protected, context)
        .await
        .map(|bounds| bounds.rendering())
}

pub(super) async fn numeric_bounds_protecting(
    input: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    context: &'static str,
) -> BuiltinResult<NumericBounds> {
    let provider = gpu::exact_provider_for_handle(input).ok_or_else(|| {
        plotting_error(context, format!("{context}: no provider owns bounds input"))
    })?;
    let expected = ExpectedOutput {
        element: gpu::expected_handle_numeric_element_type(input)
            .map_err(|error| plotting_error(context, error))?,
        precision: runmat_accelerate_api::handle_precision(input),
        integer: runmat_accelerate_api::handle_integer_type(input),
    };
    let metadata = protected
        .iter()
        .map(|handle| gpu::ResidencyRestoreGuard::new(handle))
        .collect::<Vec<_>>();
    let mut owned = Temporaries {
        provider,
        protected,
        handles: Vec::with_capacity(2),
    };
    let minimum = provider.reduce_min(input).await;
    for guard in &metadata {
        guard.restore();
    }
    let minimum = minimum.map_err(|error| {
        crate::runtime_error::with_provider_execution_source(
            plotting_error(context, format!("{context}: reduce_min failed: {error}")),
            error,
        )
    })?;
    let minimum = owned.admit(minimum, &expected, context)?;
    let maximum = provider.reduce_max(input).await;
    for guard in &metadata {
        guard.restore();
    }
    let maximum = maximum.map_err(|error| {
        crate::runtime_error::with_provider_execution_source(
            plotting_error(context, format!("{context}: reduce_max failed: {error}")),
            error.after_effect(),
        )
    })?;
    let maximum = owned.admit(maximum, &expected, context)?;
    let minimum = value_to_scalar_async(Value::GpuTensor(minimum), context)
        .await
        .map_err(|error| super::preparation_after_effect(error, context))?;
    let maximum = value_to_scalar_async(Value::GpuTensor(maximum), context)
        .await
        .map_err(|error| super::preparation_after_effect(error, context))?;
    Ok(NumericBounds {
        min: minimum,
        max: maximum,
    })
}
