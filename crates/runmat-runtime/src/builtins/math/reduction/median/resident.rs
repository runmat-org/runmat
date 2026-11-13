//! Checked ownership of optional resident median hooks. Unsupported is the only
//! provider outcome that selects host fallback; owned intermediates retire by
//! ordinary scope/async cancellation without touching borrowed operands.
use super::*;
use runmat_accelerate_api::{GpuTensorStorage, ProviderExecutionError};

/// One call-local barrier before the first resident effect. Parsing pure host
/// controls still reports its ordinary errors before ownership is needed.
pub(super) struct InputAdmission<'a> {
    inputs: Vec<&'a GpuTensorHandle>,
    admitted: bool,
}

impl<'a> InputAdmission<'a> {
    pub(super) fn new(inputs: impl Iterator<Item = &'a GpuTensorHandle>) -> Self {
        Self {
            inputs: inputs.collect(),
            admitted: false,
        }
    }

    pub(super) fn before_effect(&mut self) -> BuiltinResult<()> {
        if self.admitted {
            return Ok(());
        }
        gpu_helpers::require_resident_compute_owners(&self.inputs, false, || {
            terminal("median: resident input has no exact owning provider")
        })?;
        for handle in &self.inputs {
            gpu_helpers::require_resident_compute_owner(handle, || {
                terminal("median: resident input has no exact owning provider")
            })?;
            handle
                .allocation_lease()
                .validate_live()
                .map_err(|error| provider_error(ProviderExecutionError::from_source(error)))?;
        }
        self.admitted = true;
        Ok(())
    }
}

pub(super) fn terminal(message: &str) -> RuntimeError {
    let mut error = median_internal_error(message);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

pub(super) fn provider_error(error: ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        median_internal_error(error.to_string()),
        error,
    )
}

pub(super) async fn try_reduce(
    provider: &'static dyn AccelProvider,
    original: &GpuTensorHandle,
    axes: &MedianAxes,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Option<GpuTensorHandle>> {
    if matches!(axes, MedianAxes::All) {
        if original.shape().is_empty() {
            return Ok(Some(original.clone()));
        }
        return admit(
            provider,
            original,
            original,
            protected,
            vec![1, 1],
            provider.reduce_median(original).await,
        );
    }
    let mut dims = match axes {
        MedianAxes::Default => {
            if original.shape().is_empty() {
                return Ok(Some(original.clone()));
            }
            vec![default_dimension_from_shape(original.shape())]
        }
        MedianAxes::Dim(dim) => vec![*dim],
        MedianAxes::Vec(dims) => dims.clone(),
        MedianAxes::All => unreachable!("all handled above"),
    };
    dims.sort_unstable();
    dims.dedup();
    let mut current = original.clone();
    for dim in dims {
        if dim == 0 {
            return Err(median_invalid_argument("median: dimension must be >= 1"));
        }
        let Some(shape) = reduction_shape(current.shape(), dim) else {
            continue;
        };
        // The original and current intermediate remain strongly owned through
        // the hook. Admission protects both plus any original resident controls.
        let outcome = provider.reduce_median_dim(&current, dim - 1).await;
        let Some(next) = admit(provider, &current, original, protected, shape, outcome)? else {
            return Ok(None);
        };
        current = next;
    }
    Ok(Some(current))
}

fn admit(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    original: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
    shape: Vec<usize>,
    outcome: anyhow::Result<GpuTensorHandle>,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let output = match outcome {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => return Ok(None),
        Err(error) => return Err(provider_error(error.into())),
    };
    let contract = gpu_helpers::BinaryGpuOutputContract {
        shape,
        storage: GpuTensorStorage::Real,
        // The existing empty host reducer emits Double NaNs even for an empty
        // Single input. Do not silently relabel a returned Single allocation.
        precision: if input.shape().contains(&0) {
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        } else {
            runmat_accelerate_api::handle_precision(input)
        },
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    if !gpu_helpers::binary_gpu_output_matches(&output, input, original, provider, &contract)
        || protected
            .iter()
            .any(|input| output.shares_allocation_with(input))
    {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            terminal("median: invalid provider reduction output"),
        ));
    }
    if let Err(error) = output.allocation_lease().validate_live() {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            provider_error(ProviderExecutionError::from_source(error)),
        ));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(
        output,
        std::iter::once(original).chain(protected.iter().copied()),
    )
    .map_err(|error| provider_error(ProviderExecutionError::from_source(error)))?;
    Ok(Some(output))
}
