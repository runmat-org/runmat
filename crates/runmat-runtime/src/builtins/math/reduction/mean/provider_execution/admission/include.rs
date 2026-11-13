//! Original numerical contract remains authoritative across include-NaN stages.
use super::*;

fn admit(
    provider: &dyn AccelProvider,
    original: &GpuTensorHandle,
    source: &GpuTensorHandle,
    output: GpuTensorHandle,
    shape: &[usize],
    protected: &[&GpuTensorHandle],
    lawful_alias: bool,
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    let aliases_source = output.shares_allocation_with(source);
    let valid = exact_execution_owner(provider, original).is_some()
        && gpu_helpers::unary_gpu_output_matches_shape(
            &output,
            original,
            provider,
            shape,
            UnaryGpuOutputContract {
                storage: runmat_accelerate_api::handle_storage(original),
                precision: runmat_accelerate_api::handle_precision(original),
                integer: None,
                logical: false,
                alias: GpuOutputAliasPolicy::AllowInput,
            },
        )
        && output.physical().element_type() == original.physical().element_type()
        && output.view().validate(output.physical()).is_ok()
        && (!aliases_source || lawful_alias)
        && protected.iter().all(|operand| {
            !output.shares_allocation_with(operand)
                || (aliases_source && lawful_alias && source.shares_allocation_with(operand))
        })
        && (aliases_source
            || output.view().layout() == runmat_accelerate_api::GpuTensorLayout::Contiguous);
    if valid {
        Ok(output)
    } else {
        Err(MeanProviderFailure::Execution(Box::new(gpu_helpers::reject_gpu_output(
            output, mean_internal_error("mean: malformed original-contract intermediate class, shape, owner, layout, or alias")))))
    }
}

pub(in super::super) fn admit_include_reshape(
    provider: &dyn AccelProvider,
    original: &GpuTensorHandle,
    source: &GpuTensorHandle,
    output: GpuTensorHandle,
    shape: &[usize],
    protected: &[&GpuTensorHandle],
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    let valid = gpu_helpers::reshape_output_matches(source, &output, provider, shape);
    if !valid {
        return Err(MeanProviderFailure::Execution(Box::new(
            gpu_helpers::reject_gpu_output(
                output,
                mean_internal_error("mean: malformed provider reshape intermediate"),
            ),
        )));
    }
    admit(provider, original, source, output, shape, protected, true)
}

pub(in super::super) fn admit_include_reduction(
    provider: &dyn AccelProvider,
    original: &GpuTensorHandle,
    source: &GpuTensorHandle,
    output: GpuTensorHandle,
    shape: &[usize],
    axis: usize,
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    let identity = source.shape().get(axis) == Some(&1) && output.view() == source.view();
    admit(
        provider,
        original,
        source,
        output,
        shape,
        &[original, source],
        identity,
    )
}

pub(in super::super) fn admit_include_permutation(
    provider: &dyn AccelProvider,
    original: &GpuTensorHandle,
    source: &GpuTensorHandle,
    output: GpuTensorHandle,
    order: &[usize],
    protected: &[&GpuTensorHandle],
) -> Result<GpuTensorHandle, MeanProviderFailure> {
    let shape = order
        .iter()
        .map(|&axis| source.shape().get(axis).copied())
        .collect::<Option<Vec<_>>>();
    if !gpu_helpers::permutation_output_matches(source, &output, provider, order) || shape.is_none()
    {
        return Err(MeanProviderFailure::Execution(Box::new(
            gpu_helpers::reject_gpu_output(
                output,
                mean_internal_error("mean: malformed provider permutation intermediate"),
            ),
        )));
    }
    admit(
        provider,
        original,
        source,
        output,
        &shape.expect("validated order"),
        protected,
        true,
    )
}
