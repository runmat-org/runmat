//! Dimension selection and NaN-aware mean on one provider.
use super::{provider_failure, Failure};
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
    for axis in axes {
        current = provider
            .reduce_sum_dim(&current, *axis)
            .await
            .map_err(|source| provider_failure(kind, source))?;
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
    if selected.is_empty() {
        return Ok(transformed.clone());
    }
    if missing == MissingMode::Include {
        let mut current = transformed.clone();
        for axis in &selected {
            current = provider
                .reduce_mean_dim(&current, *axis)
                .await
                .map_err(|source| provider_failure(kind, source))?;
        }
        return Ok(current);
    }
    let cleaned = provider
        .map_nan_to_zero(transformed)
        .map_err(|source| provider_failure(kind, source))?;
    let mask = provider
        .not_nan_mask(original)
        .map_err(|source| provider_failure(kind, source))?;
    let numerator = sum(kind, provider, &cleaned, &selected).await?;
    let denominator = sum(kind, provider, &mask, &selected).await?;
    provider
        .elem_div(&numerator, &denominator)
        .await
        .map_err(|source| provider_failure(kind, source))
}
