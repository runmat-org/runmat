//! Dimension selection and NaN-aware mean on one provider.
use super::{admitted_real, expected_shape, provider_failure, Failure};
use crate::builtins::stats::summary::statistical_means::options::{
    Axes, MissingMode, ReductionKind,
};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderPrecision};

pub(super) fn selected_axes(shape: &[usize], axes: &Axes) -> Vec<usize> {
    match axes {
        Axes::Default => vec![shape.iter().position(|extent| *extent != 1).unwrap_or(0)],
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
        let next = provider
            .reduce_sum_dim(&current, *axis)
            .await
            .map_err(|source| provider_failure(kind, source))?;
        let expected = expected_shape(current.shape(), &Axes::Dims(vec![axis + 1]));
        current = admitted_real(
            kind,
            provider,
            input,
            next,
            &expected,
            ProviderPrecision::F64,
        )?;
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
    let result = admitted_real(
        kind,
        provider,
        original,
        result,
        &expected_shape(original.shape(), axes),
        ProviderPrecision::F64,
    )?;
    if precision == ProviderPrecision::F64 {
        return Ok(result);
    }
    let output = provider
        .unary_single(&result)
        .await
        .map_err(|source| provider_failure(kind, source))?;
    admitted_real(
        kind,
        provider,
        original,
        output,
        result.shape(),
        ProviderPrecision::F32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderPrecision,
    };
    use runmat_value::Value;

    #[test]
    fn omitted_single_observations_accumulate_in_double_and_publish_single() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let samples = [16_777_216.0_f32, 1.0, 1.0, f32::NAN];
            let input = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&samples),
                    shape: &[4, 1],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            provider.reset_telemetry();
            let result = block_on(mean(
                ReductionKind::Harmonic,
                provider,
                &input,
                &input,
                &Axes::Default,
                MissingMode::Omit,
            ))
            .unwrap();
            assert_eq!(
                runmat_accelerate_api::handle_precision(&result),
                Some(ProviderPrecision::F32)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(
                crate::builtins::common::test_support::gather(Value::GpuTensor(result))
                    .unwrap()
                    .materialize_f64(),
                vec![5_592_406.0],
            );
            input.allocation_lease().validate_live().unwrap();
        });
    }
}
