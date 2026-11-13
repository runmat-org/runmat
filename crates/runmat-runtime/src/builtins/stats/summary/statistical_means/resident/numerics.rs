//! Numerical kernels for real floating resident inputs. The RMS route scales
//! each output slice before squaring, avoiding finite-input overflow.
use super::{
    admitted_mask, admitted_real, expected_shape, reduction, Failure, ObserveResidentHook,
    ResidentStageProgress,
};
use crate::builtins::stats::summary::statistical_means::options::{Options, ReductionKind};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};

pub(super) async fn run(
    kind: ReductionKind,
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    options: &Options,
    progress: &mut ResidentStageProgress,
) -> Result<GpuTensorHandle, Failure> {
    let Some(precision) = runmat_accelerate_api::handle_precision(input) else {
        return Err(progress.unavailable(kind, "resident input has no floating precision"));
    };
    if runmat_accelerate_api::handle_integer_type(input).is_some()
        || runmat_accelerate_api::handle_is_logical(input)
        || input.shape().contains(&0)
    {
        return Err(progress.unavailable(
            kind,
            "resident input is not a nonempty real floating tensor",
        ));
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
                return Err(progress.unavailable(kind, "geometric resident input is not real"));
            }
            if reduction::selected_axes(input.shape(), &options.axes)
                .iter()
                .all(|axis| input.shape()[*axis] == 1)
            {
                let zero = progress.observe_optional(
                    kind,
                    provider.zeros_like(input, input.shape()),
                    "resident constant constructor unavailable",
                )?;
                let zero = admitted_real(kind, provider, input, zero, input.shape(), precision)?;
                let zero = reduction::protect_live_operands(
                    kind,
                    zero,
                    &[input],
                    "resident zero constructor aliased its input",
                )?;
                let nonnegative = provider
                    .elem_ge(input, &zero)
                    .await
                    .observe(progress, kind)?;
                let nonnegative = admitted_mask(kind, provider, input, nonnegative, input.shape())?;
                let nan = progress.observe_optional(
                    kind,
                    provider.fill_like(input, f64::NAN),
                    "resident constant constructor unavailable",
                )?;
                let nan = admitted_real(kind, provider, input, nan, input.shape(), precision)?;
                let nan = reduction::protect_live_operands(
                    kind,
                    nan,
                    &[input, &zero, &nonnegative],
                    "resident NaN constructor aliased a live operand",
                )?;
                let output = provider
                    .select(&nonnegative, input, &nan)
                    .await
                    .observe(progress, kind)?;
                return admitted_real(kind, provider, input, output, input.shape(), precision);
            }
            let cleaned = provider.map_nan_to_zero(input).observe(progress, kind)?;
            let cleaned = admitted_real(kind, provider, input, cleaned, input.shape(), precision)?;
            let mut minimum = cleaned;
            for axis in reduction::selected_axes(input.shape(), &options.axes) {
                let next = provider
                    .reduce_min_dim(&minimum, axis)
                    .await
                    .observe(progress, kind)?
                    .values;
                let expected = expected_shape(
                    minimum.shape(),
                    &super::super::options::Axes::Dims(vec![axis + 1]),
                );
                minimum = admitted_real(kind, provider, input, next, &expected, precision)?;
            }
            let zero = progress.observe_optional(
                kind,
                provider.zeros_like(&minimum, minimum.shape()),
                "resident constant constructor unavailable",
            )?;
            let zero = admitted_real(kind, provider, input, zero, minimum.shape(), precision)?;
            let zero = reduction::protect_live_operands(
                kind,
                zero,
                &[input, &minimum],
                "resident zero constructor aliased a live operand",
            )?;
            let nonnegative = provider
                .elem_ge(&minimum, &zero)
                .await
                .observe(progress, kind)?;
            let nonnegative = admitted_mask(kind, provider, input, nonnegative, minimum.shape())?;
            let logarithms = provider.unary_log(input).await.observe(progress, kind)?;
            let logarithms =
                admitted_real(kind, provider, input, logarithms, input.shape(), precision)?;
            let average = reduction::mean(
                kind,
                provider,
                input,
                &logarithms,
                &options.axes,
                options.missing,
                progress,
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
            let result = provider.unary_exp(&average).await.observe(progress, kind)?;
            let result = admitted_real(kind, provider, input, result, minimum.shape(), precision)?;
            let nan = progress.observe_optional(
                kind,
                provider.fill_like(&result, f64::NAN),
                "resident constant constructor unavailable",
            )?;
            let nan = admitted_real(kind, provider, input, nan, minimum.shape(), precision)?;
            let nan = reduction::protect_live_operands(
                kind,
                nan,
                &[
                    input,
                    &minimum,
                    &zero,
                    &nonnegative,
                    &logarithms,
                    &average,
                    &result,
                ],
                "resident NaN constructor aliased a live operand",
            )?;
            provider
                .select(&nonnegative, &result, &nan)
                .await
                .observe(progress, kind)?
        }
        ReductionKind::Harmonic => {
            if runmat_accelerate_api::handle_storage(input) != GpuTensorStorage::Real {
                return Err(progress.unavailable(kind, "harmonic resident input is not real"));
            }
            let absolute = provider.unary_abs(input).await.observe(progress, kind)?;
            let absolute =
                admitted_real(kind, provider, input, absolute, input.shape(), precision)?;
            let mut scale = if options.missing == super::super::options::MissingMode::Omit {
                let cleaned = provider
                    .map_nan_to_zero(&absolute)
                    .observe(progress, kind)?;
                let cleaned =
                    admitted_real(kind, provider, input, cleaned, input.shape(), precision)?;
                let valid = provider.not_nan_mask(input).observe(progress, kind)?;
                let valid = admitted_real(kind, provider, input, valid, input.shape(), precision)?;
                let missing = provider.scalar_rsub(&valid, 1.0).observe(progress, kind)?;
                let missing =
                    admitted_real(kind, provider, input, missing, input.shape(), precision)?;
                let maximum = match precision {
                    runmat_accelerate_api::ProviderPrecision::F32 => f64::from(f32::MAX),
                    runmat_accelerate_api::ProviderPrecision::F64 => f64::MAX,
                };
                let replacement = provider
                    .scalar_mul(&missing, maximum)
                    .observe(progress, kind)?;
                let replacement =
                    admitted_real(kind, provider, input, replacement, input.shape(), precision)?;
                let filled = provider
                    .elem_add(&cleaned, &replacement)
                    .await
                    .observe(progress, kind)?;
                admitted_real(kind, provider, input, filled, input.shape(), precision)?
            } else {
                absolute.clone()
            };
            for axis in reduction::selected_axes(input.shape(), &options.axes) {
                let next = provider
                    .reduce_min_dim(&scale, axis)
                    .await
                    .observe(progress, kind)?
                    .values;
                let expected = expected_shape(
                    scale.shape(),
                    &super::super::options::Axes::Dims(vec![axis + 1]),
                );
                scale = admitted_real(kind, provider, input, next, &expected, precision)?;
            }
            let scale =
                super::usable_scale(kind, provider, input, scale, &[&absolute], progress).await?;
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
                .observe(progress, kind)?;
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
                progress,
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
                .observe(progress, kind)?
        }
        ReductionKind::RootMeanSquare | ReductionKind::RootMeanSquaredError => {
            let components = if runmat_accelerate_api::handle_storage(input)
                == GpuTensorStorage::ComplexInterleaved
            {
                let real = provider.unary_real(input).await.observe(progress, kind)?;
                let imaginary = provider.unary_imag(input).await.observe(progress, kind)?;
                let real = admitted_real(kind, provider, input, real, input.shape(), precision)?;
                let imaginary =
                    admitted_real(kind, provider, input, imaginary, input.shape(), precision)?;
                Some((real, imaginary))
            } else {
                None
            };
            let absolute = if let Some((real, imaginary)) = &components {
                let abs_real = provider.unary_abs(real).await.observe(progress, kind)?;
                let abs_imaginary = provider
                    .unary_abs(imaginary)
                    .await
                    .observe(progress, kind)?;
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
                    .observe(progress, kind)?
            } else {
                provider.unary_abs(input).await.observe(progress, kind)?
            };
            let absolute =
                admitted_real(kind, provider, input, absolute, input.shape(), precision)?;
            let selected = reduction::selected_axes(input.shape(), &options.axes);
            let mut scale = if options.missing == super::super::options::MissingMode::Omit {
                let cleaned = provider
                    .map_nan_to_zero(&absolute)
                    .observe(progress, kind)?;
                admitted_real(kind, provider, input, cleaned, input.shape(), precision)?
            } else {
                absolute.clone()
            };
            for axis in selected {
                let result = provider
                    .reduce_max_dim(&scale, axis)
                    .await
                    .observe(progress, kind)?;
                let expected = expected_shape(
                    scale.shape(),
                    &super::super::options::Axes::Dims(vec![axis + 1]),
                );
                scale = admitted_real(kind, provider, input, result.values, &expected, precision)?;
            }
            let protected = components
                .as_ref()
                .map_or([input, input, &absolute], |(real, imaginary)| {
                    [real, imaginary, &absolute]
                });
            let scale =
                super::usable_scale(kind, provider, input, scale, &protected, progress).await?;
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
                    .observe(progress, kind)?;
                let normalized_imaginary = provider
                    .elem_div(imaginary, &scale)
                    .await
                    .observe(progress, kind)?;
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
                    .observe(progress, kind)?;
                let imaginary_squared = provider
                    .elem_mul(&normalized_imaginary, &normalized_imaginary)
                    .await
                    .observe(progress, kind)?;
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
                    .observe(progress, kind)?
            } else {
                let normalized = provider
                    .elem_div(&absolute, &scale)
                    .await
                    .observe(progress, kind)?;
                let normalized =
                    admitted_real(kind, provider, input, normalized, input.shape(), precision)?;
                provider
                    .elem_mul(&normalized, &normalized)
                    .await
                    .observe(progress, kind)?
            };
            let squared = admitted_real(kind, provider, input, squared, input.shape(), precision)?;
            let average = reduction::mean(
                kind,
                provider,
                &squared,
                &squared,
                &options.axes,
                options.missing,
                progress,
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
                .observe(progress, kind)?;
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
                .observe(progress, kind)?
        }
    };
    Ok(result)
}
