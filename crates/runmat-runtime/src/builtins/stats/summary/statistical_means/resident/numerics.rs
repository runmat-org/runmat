//! Numerical kernels for real floating resident inputs. The RMS route scales
//! each output slice before squaring, avoiding finite-input overflow.
use super::{
    contract_failure, expected_shape, provider_failure, reduction, scalar_summary, validate_output,
    Failure,
};
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
    let result = match kind {
        ReductionKind::Geometric => {
            if runmat_accelerate_api::handle_storage(input) != GpuTensorStorage::Real {
                return Err(Failure::Unsupported);
            }
            let cleaned = provider
                .map_nan_to_zero(input)
                .map_err(|source| provider_failure(kind, source))?;
            let mut minimum = cleaned;
            for axis in 0..input.shape().len() {
                minimum = provider
                    .reduce_min_dim(&minimum, axis)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
                    .values;
            }
            let minimum = scalar_summary(kind, provider, &minimum).await?;
            if minimum < 0.0 {
                return Err(contract_failure(
                    kind,
                    "geomean input values must be nonnegative",
                ));
            }
            let logarithms = provider
                .unary_log(input)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            validate_output(kind, provider, input, &logarithms, input.shape(), precision)?;
            let average = reduction::mean(
                kind,
                provider,
                input,
                &logarithms,
                &options.axes,
                options.missing,
            )
            .await?;
            provider
                .unary_exp(&average)
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
            validate_output(kind, provider, input, &absolute, input.shape(), precision)?;
            let mut scale = if options.missing == super::super::options::MissingMode::Omit {
                let cleaned = provider
                    .map_nan_to_zero(&absolute)
                    .map_err(|source| provider_failure(kind, source))?;
                let valid = provider
                    .not_nan_mask(input)
                    .map_err(|source| provider_failure(kind, source))?;
                let missing = provider
                    .scalar_rsub(&valid, 1.0)
                    .map_err(|source| provider_failure(kind, source))?;
                let maximum = match precision {
                    runmat_accelerate_api::ProviderPrecision::F32 => f64::from(f32::MAX),
                    runmat_accelerate_api::ProviderPrecision::F64 => f64::MAX,
                };
                let replacement = provider
                    .scalar_mul(&missing, maximum)
                    .map_err(|source| provider_failure(kind, source))?;
                provider
                    .elem_add(&cleaned, &replacement)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
            } else {
                absolute
            };
            for axis in reduction::selected_axes(input.shape(), &options.axes) {
                scale = provider
                    .reduce_min_dim(&scale, axis)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
                    .values;
            }
            let scale = super::usable_scale(kind, provider, input, scale).await?;
            validate_output(
                kind,
                provider,
                input,
                &scale,
                &expected_shape(input.shape(), &options.axes),
                precision,
            )?;
            let normalized_reciprocals = provider
                .elem_div(&scale, input)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            validate_output(
                kind,
                provider,
                input,
                &normalized_reciprocals,
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
                validate_output(kind, provider, input, &real, input.shape(), precision)?;
                validate_output(kind, provider, input, &imaginary, input.shape(), precision)?;
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
            validate_output(kind, provider, input, &absolute, input.shape(), precision)?;
            let selected = reduction::selected_axes(input.shape(), &options.axes);
            let mut scale = if options.missing == super::super::options::MissingMode::Omit {
                provider
                    .map_nan_to_zero(&absolute)
                    .map_err(|source| provider_failure(kind, source))?
            } else {
                absolute.clone()
            };
            for axis in selected {
                let result = provider
                    .reduce_max_dim(&scale, axis)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                scale = result.values;
            }
            let scale = super::usable_scale(kind, provider, input, scale).await?;
            validate_output(
                kind,
                provider,
                input,
                &scale,
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
                validate_output(
                    kind,
                    provider,
                    input,
                    &normalized_real,
                    input.shape(),
                    precision,
                )?;
                validate_output(
                    kind,
                    provider,
                    input,
                    &normalized_imaginary,
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
                provider
                    .elem_add(&real_squared, &imaginary_squared)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
            } else {
                let normalized = provider
                    .elem_div(&absolute, &scale)
                    .await
                    .map_err(|source| provider_failure(kind, source))?;
                validate_output(kind, provider, input, &normalized, input.shape(), precision)?;
                provider
                    .elem_mul(&normalized, &normalized)
                    .await
                    .map_err(|source| provider_failure(kind, source))?
            };
            validate_output(kind, provider, input, &squared, input.shape(), precision)?;
            let average = reduction::mean(
                kind,
                provider,
                input,
                &squared,
                &options.axes,
                options.missing,
            )
            .await?;
            let rooted = provider
                .unary_sqrt(&average)
                .await
                .map_err(|source| provider_failure(kind, source))?;
            provider
                .elem_mul(&rooted, &scale)
                .await
                .map_err(|source| provider_failure(kind, source))?
        }
    };
    Ok(result)
}
