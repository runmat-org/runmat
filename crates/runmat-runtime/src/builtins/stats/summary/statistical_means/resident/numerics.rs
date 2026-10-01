//! Numerical kernels for real floating resident inputs. The RMS route scales
//! each output slice before squaring, avoiding finite-input overflow.
use super::{admitted_mask, admitted_real, expected_shape, provider_failure, reduction, Failure};
use crate::builtins::stats::summary::statistical_means::options::{Options, ReductionKind};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

pub(super) async fn run(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    options: &Options,
) -> Result<GpuTensorHandle, Failure> {
    let Some(precision) = runmat_accelerate_api::handle_precision(input) else {
        return Err(Failure::Unsupported);
    };
    if runmat_accelerate_api::handle_integer_type(input).is_some()
        || runmat_accelerate_api::handle_is_logical(input)
        || input.shape().contains(&0)
    {
        return Err(Failure::Unsupported);
    }
    if kind == ReductionKind::Harmonic
        && reduction::selected_axes(input.shape(), &options.axes)
            .iter()
            .all(|axis| input.shape()[*axis] == 1)
    {
        return Ok(input.clone());
    }
    let result = match kind {
        ReductionKind::Geometric => {
            if runmat_accelerate_api::handle_storage(input) != GpuTensorStorage::Real {
                return Err(Failure::Unsupported);
            }
            if reduction::selected_axes(input.shape(), &options.axes)
                .iter()
                .all(|axis| input.shape()[*axis] == 1)
            {
                let zero = provider
                    .zeros_like(input, input.shape())
                    .map_err(|source| provider_failure(kind, source))?;
                let zero = admitted_real(kind, provider, input, zero, input.shape(), precision)?;
                let nonnegative = provider
                    .elem_ge(input, &zero)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let nonnegative = admitted_mask(kind, provider, input, nonnegative, input.shape())?;
                let nan = provider
                    .fill_like(input, f64::NAN)
                    .map_err(|source| provider_failure(kind, source))?;
                let nan = admitted_real(kind, provider, input, nan, input.shape(), precision)?;
                let output = provider
                    .select(&nonnegative, input, &nan)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                return admitted_real(kind, provider, input, output, input.shape(), precision);
            }
            let cleaned = provider
                .map_nan_to_zero(input)
                .map_err(|source| provider_failure(kind, source))?;
            let cleaned = admitted_real(kind, provider, input, cleaned, input.shape(), precision)?;
            let mut minimum = cleaned;
            for axis in reduction::selected_axes(input.shape(), &options.axes) {
                let next = provider
                    .reduce_min_dim(&minimum, axis)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
                    .values;
                let expected = expected_shape(
                    minimum.shape(),
                    &super::super::options::Axes::Dims(vec![axis + 1]),
                );
                minimum = admitted_real(kind, provider, input, next, &expected, precision)?;
            }
            let zero = provider
                .zeros_like(&minimum, minimum.shape())
                .map_err(|source| provider_failure(kind, source))?;
            let zero = admitted_real(kind, provider, input, zero, minimum.shape(), precision)?;
            let nonnegative = provider
                .elem_ge(&minimum, &zero)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            let nonnegative = admitted_mask(kind, provider, input, nonnegative, minimum.shape())?;
            let logarithms = provider
                .unary_log(input)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            let logarithms =
                admitted_real(kind, provider, input, logarithms, input.shape(), precision)?;
            let average = reduction::mean(
                kind,
                provider,
                input,
                &logarithms,
                &options.axes,
                options.missing,
            )
            .await?;
            let average = admitted_real(
                kind,
                provider,
                input,
                average,
                &expected_shape(input.shape(), &options.axes),
                precision,
            )?;
            let result = provider
                .unary_exp(&average)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            let result = admitted_real(kind, provider, input, result, minimum.shape(), precision)?;
            let nan = provider
                .fill_like(&result, f64::NAN)
                .map_err(|source| provider_failure(kind, source))?;
            let nan = admitted_real(kind, provider, input, nan, minimum.shape(), precision)?;
            provider
                .select(&nonnegative, &result, &nan)
                .await
                .map_err(|source| provider_failure(kind, source))?
        }
        ReductionKind::Harmonic => {
            if runmat_accelerate_api::handle_storage(input) != GpuTensorStorage::Real {
                return Err(Failure::Unsupported);
            }
            let absolute = provider
                .unary_abs(input)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            let absolute =
                admitted_real(kind, provider, input, absolute, input.shape(), precision)?;
            let mut scale = if options.missing == super::super::options::MissingMode::Omit {
                let cleaned = provider
                    .map_nan_to_zero(&absolute)
                    .map_err(|source| provider_failure(kind, source))?;
                let cleaned =
                    admitted_real(kind, provider, input, cleaned, input.shape(), precision)?;
                let valid = provider
                    .not_nan_mask(input)
                    .map_err(|source| provider_failure(kind, source))?;
                let valid = admitted_real(kind, provider, input, valid, input.shape(), precision)?;
                let missing = provider
                    .scalar_rsub(&valid, 1.0)
                    .map_err(|source| provider_failure(kind, source))?;
                let missing =
                    admitted_real(kind, provider, input, missing, input.shape(), precision)?;
                let maximum = match precision {
                    runmat_accelerate_api::ProviderPrecision::F32 => f64::from(f32::MAX),
                    runmat_accelerate_api::ProviderPrecision::F64 => f64::MAX,
                };
                let replacement = provider
                    .scalar_mul(&missing, maximum)
                    .map_err(|source| provider_failure(kind, source))?;
                let replacement =
                    admitted_real(kind, provider, input, replacement, input.shape(), precision)?;
                let filled = provider
                    .elem_add(&cleaned, &replacement)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                admitted_real(kind, provider, input, filled, input.shape(), precision)?
            } else {
                absolute
            };
            for axis in reduction::selected_axes(input.shape(), &options.axes) {
                let next = provider
                    .reduce_min_dim(&scale, axis)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
                    .values;
                let expected = expected_shape(
                    scale.shape(),
                    &super::super::options::Axes::Dims(vec![axis + 1]),
                );
                scale = admitted_real(kind, provider, input, next, &expected, precision)?;
            }
            let scale = super::usable_scale(kind, provider, input, scale).await?;
            let scale = admitted_real(
                kind,
                provider,
                input,
                scale,
                &expected_shape(input.shape(), &options.axes),
                precision,
            )?;
            let normalized_reciprocals = provider
                .elem_div(&scale, input)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            let normalized_reciprocals = admitted_real(
                kind,
                provider,
                input,
                normalized_reciprocals,
                input.shape(),
                precision,
            )?;
            let average = reduction::mean(
                kind,
                provider,
                input,
                &normalized_reciprocals,
                &options.axes,
                options.missing,
            )
            .await?;
            let average = admitted_real(
                kind,
                provider,
                input,
                average,
                &expected_shape(input.shape(), &options.axes),
                precision,
            )?;
            provider
                .elem_div(&scale, &average)
                .await
                .map_err(|source| provider_failure(kind, source))?
        }
        ReductionKind::RootMeanSquare | ReductionKind::RootMeanSquaredError => {
            let components = if runmat_accelerate_api::handle_storage(input)
                == GpuTensorStorage::ComplexInterleaved
            {
                let real = provider
                    .unary_real(input)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let imaginary = provider
                    .unary_imag(input)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let real = admitted_real(kind, provider, input, real, input.shape(), precision)?;
                let imaginary =
                    admitted_real(kind, provider, input, imaginary, input.shape(), precision)?;
                Some((real, imaginary))
            } else {
                None
            };
            let absolute = if let Some((real, imaginary)) = &components {
                let abs_real = provider
                    .unary_abs(real)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let abs_imaginary = provider
                    .unary_abs(imaginary)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let abs_real =
                    admitted_real(kind, provider, input, abs_real, input.shape(), precision)?;
                let abs_imaginary = admitted_real(
                    kind,
                    provider,
                    input,
                    abs_imaginary,
                    input.shape(),
                    precision,
                )?;
                provider
                    .elem_max(&abs_real, &abs_imaginary)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
            } else {
                provider
                    .unary_abs(input)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
            };
            let absolute =
                admitted_real(kind, provider, input, absolute, input.shape(), precision)?;
            let selected = reduction::selected_axes(input.shape(), &options.axes);
            let mut scale = if options.missing == super::super::options::MissingMode::Omit {
                let cleaned = provider
                    .map_nan_to_zero(&absolute)
                    .map_err(|source| provider_failure(kind, source))?;
                admitted_real(kind, provider, input, cleaned, input.shape(), precision)?
            } else {
                absolute.clone()
            };
            for axis in selected {
                let result = provider
                    .reduce_max_dim(&scale, axis)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let expected = expected_shape(
                    scale.shape(),
                    &super::super::options::Axes::Dims(vec![axis + 1]),
                );
                scale = admitted_real(kind, provider, input, result.values, &expected, precision)?;
            }
            let scale = super::usable_scale(kind, provider, input, scale).await?;
            let scale = admitted_real(
                kind,
                provider,
                input,
                scale,
                &expected_shape(input.shape(), &options.axes),
                precision,
            )?;
            let squared = if let Some((real, imaginary)) = &components {
                let normalized_real = provider
                    .elem_div(real, &scale)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let normalized_imaginary = provider
                    .elem_div(imaginary, &scale)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let normalized_real = admitted_real(
                    kind,
                    provider,
                    input,
                    normalized_real,
                    input.shape(),
                    precision,
                )?;
                let normalized_imaginary = admitted_real(
                    kind,
                    provider,
                    input,
                    normalized_imaginary,
                    input.shape(),
                    precision,
                )?;
                let real_squared = provider
                    .elem_mul(&normalized_real, &normalized_real)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let imaginary_squared = provider
                    .elem_mul(&normalized_imaginary, &normalized_imaginary)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let real_squared = admitted_real(
                    kind,
                    provider,
                    input,
                    real_squared,
                    input.shape(),
                    precision,
                )?;
                let imaginary_squared = admitted_real(
                    kind,
                    provider,
                    input,
                    imaginary_squared,
                    input.shape(),
                    precision,
                )?;
                provider
                    .elem_add(&real_squared, &imaginary_squared)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
            } else {
                let normalized = provider
                    .elem_div(&absolute, &scale)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                let normalized =
                    admitted_real(kind, provider, input, normalized, input.shape(), precision)?;
                provider
                    .elem_mul(&normalized, &normalized)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
            };
            let squared = admitted_real(kind, provider, input, squared, input.shape(), precision)?;
            let average = reduction::mean(
                kind,
                provider,
                &squared,
                &squared,
                &options.axes,
                options.missing,
            )
            .await?;
            let average = admitted_real(
                kind,
                provider,
                input,
                average,
                &expected_shape(input.shape(), &options.axes),
                precision,
            )?;
            let rooted = provider
                .unary_sqrt(&average)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            let rooted = admitted_real(
                kind,
                provider,
                input,
                rooted,
                &expected_shape(input.shape(), &options.axes),
                precision,
            )?;
            provider
                .elem_mul(&rooted, &scale)
                .await
                .map_err(|source| provider_failure(kind, source))?
        }
    };
    Ok(result)
}
