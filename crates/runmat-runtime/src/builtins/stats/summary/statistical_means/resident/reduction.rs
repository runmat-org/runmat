//! Dimension selection and NaN-aware mean on one provider.
use super::{admitted_real, expected_shape, provider_failure, Failure};
use crate::builtins::stats::summary::statistical_means::options::{
    Axes, MissingMode, ReductionKind,
};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

pub(super) fn selected_axes(shape: &[usize], axes: &Axes) -> Vec<usize> {
    match axes {
        Axes::Default => vec![shape.iter().position(|extent| *extent > 1).unwrap_or(0)],
        Axes::All => (0..shape.len()).collect(),
        Axes::Dims(dimensions) => dimensions
            .iter()
            .filter_map(|axis| {
                let zero_based = axis - 1;
                (zero_based < shape.len()).then_some(zero_based)
            })
            .collect(),
    }
}

async fn sum(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    axes: &[usize],
) -> Result<GpuTensorHandle, Failure> {
    let mut current = input.clone();
    let precision = runmat_accelerate_api::handle_precision(input).ok_or(Failure::Unsupported)?;
    for axis in axes {
        let next = provider
            .reduce_sum_dim(&current, *axis)
            .await
            .map_err(|source| provider_failure(kind, source))?;
        let expected = expected_shape(current.shape(), &Axes::Dims(vec![axis + 1]));
        current = admitted_real(kind, provider, input, next, &expected, precision)?;
    }
    Ok(current)
}

pub(super) async fn mean(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    original: &GpuTensorHandle,
    transformed: &GpuTensorHandle,
    axes: &Axes,
    missing: MissingMode,
) -> Result<GpuTensorHandle, Failure> {
    let selected = selected_axes(original.shape(), axes);
    let precision =
        runmat_accelerate_api::handle_precision(original).ok_or(Failure::Unsupported)?;
    if selected.is_empty() {
        return Ok(transformed.clone());
    }
    if missing == MissingMode::Include {
        let mut current = transformed.clone();
        for axis in &selected {
            let next = provider
                .reduce_mean_dim(&current, *axis)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            let expected = expected_shape(current.shape(), &Axes::Dims(vec![axis + 1]));
            current = admitted_real(kind, provider, original, next, &expected, precision)?;
        }
        return Ok(current);
    }
    let cleaned = provider
        .map_nan_to_zero(transformed)
        .map_err(|source| provider_failure(kind, source))?;
    let cleaned = admitted_real(
        kind,
        provider,
        original,
        cleaned,
        transformed.shape(),
        precision,
    )?;
    let mask = provider
        .not_nan_mask(original)
        .map_err(|source| provider_failure(kind, source))?;
    let mask = admitted_real(kind, provider, original, mask, original.shape(), precision)?;
    let numerator = sum(kind, provider, &cleaned, &selected).await?;
    let denominator = sum(kind, provider, &mask, &selected).await?;
    let result = provider
        .elem_div(&numerator, &denominator)
        .await
        .map_err(|source| provider_failure(kind, source))?;
    admitted_real(
        kind,
        provider,
        original,
        result,
        &expected_shape(original.shape(), axes),
        precision,
    )
}
