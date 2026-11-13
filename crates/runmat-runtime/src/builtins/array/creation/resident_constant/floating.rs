//! Floating constant adaptation only after certified constructor absence.
use super::*;
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::{ComplexTensor, NumericDType, Tensor};

pub(in super::super) fn floating_constant(
    provider: &'static dyn AccelProvider,
    name: &str,
    shape: &[usize],
    value: f64,
    dtype: NumericDType,
    prototype: Option<&GpuTensorHandle>,
) -> BuiltinResult<GpuTensorHandle> {
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
        _ => unreachable!("floating constructor requires floating dtype"),
    };
    let storage = prototype.map_or(GpuTensorStorage::Real, |input| input.physical().storage());
    let protected = prototype.into_iter().collect::<Vec<_>>();
    let attempt = if storage == GpuTensorStorage::ComplexInterleaved {
        Ok(None)
    } else if let Some(input) = prototype.filter(|input| input.shape() == shape) {
        provider.fill_like(input, value)
    } else if provider.precision() == precision {
        provider.fill(shape, value)
    } else {
        Ok(None)
    };
    let output = match attempt {
        Ok(Some(output)) => output,
        Err(cause) => return Err(provider_failure(name, "provider fill failed", cause)),
        Ok(None) => {
            let elements = count(shape)?;
            if storage == GpuTensorStorage::ComplexInterleaved {
                let host = ComplexTensor::from_f64_values_with_dtype(
                    vec![(value, 0.0); elements],
                    shape.to_vec(),
                    dtype,
                )
                .map_err(|source| {
                    build_runtime_error(format!("{name}: {source}"))
                        .with_builtin(name)
                        .build()
                })?;
                gpu_helpers::upload_complex_tensor_protecting(provider, &host, &protected).map_err(
                    |cause| {
                        provider_failure(
                            name,
                            "complex upload failed",
                            runmat_accelerate_api::ProviderExecutionError::from_source(cause),
                        )
                    },
                )?
            } else {
                let host = Tensor::new_with_dtype(vec![value; elements], shape.to_vec(), dtype)
                    .map_err(|source| {
                        build_runtime_error(format!("{name}: {source}"))
                            .with_builtin(name)
                            .build()
                    })?;
                gpu_helpers::upload_tensor_protecting(provider, &host, &protected).map_err(
                    |cause| {
                        provider_failure(
                            name,
                            "upload failed",
                            runmat_accelerate_api::ProviderExecutionError::from_source(cause),
                        )
                    },
                )?
            }
        }
    };
    let output = validate_constructor_gpu_output_protecting(
        name,
        provider,
        output,
        shape,
        storage,
        Some(precision),
        None,
        false,
        &protected,
    )?;
    let output = if let Some(prototype) = prototype {
        gpu_helpers::propagate_owned_output_provenance(output, [prototype])?
    } else {
        // This branch is reached only by the explicit `gpuArray` constructor
        // form. A `like` form always carries its prototype above and inherits
        // that handle's intent instead.
        match output.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit) {
            Ok(explicit) => explicit,
            Err(source) => {
                let error = build_runtime_error(format!(
                    "{name}: cannot preserve explicit gpuArray output intent: {source}"
                ))
                .with_builtin(name)
                .with_source(source)
                .build();
                return Err(gpu_helpers::reject_gpu_output(output, error));
            }
        }
    };
    Ok(output)
}
