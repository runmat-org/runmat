//! Admission of one forward FFT pass before publication or another provider call.
use super::trim_transform_output_shape;
use crate::builtins::common::{gpu_helpers, shape::normalize_scalar_shape};
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderExecutionError, ProviderPrecision,
};

pub(in crate::builtins::math::fft) fn validate_forward_input(
    input: &GpuTensorHandle,
    error: impl FnOnce() -> RuntimeError,
) -> BuiltinResult<()> {
    input.allocation_lease().validate_live().map_err(|source| {
        crate::runtime_error::with_provider_execution_source(
            error(),
            ProviderExecutionError::from_source(source),
        )
    })
}

pub(in crate::builtins::math::fft) fn admit_forward_output(
    provider: &'static dyn AccelProvider,
    output: GpuTensorHandle,
    current: &GpuTensorHandle,
    original: &GpuTensorHandle,
    length: Option<usize>,
    axis: usize,
    error: impl FnOnce() -> RuntimeError,
) -> BuiltinResult<GpuTensorHandle> {
    if let Err(source) = output.allocation_lease().validate_live() {
        let error = crate::runtime_error::with_provider_execution_source(
            error(),
            ProviderExecutionError::from_source(source),
        );
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    let origin_rank = current.shape().len();
    let mut shape = normalize_scalar_shape(current.shape());
    while shape.len() <= axis {
        shape.push(1);
    }
    if let Some(length) = length {
        shape[axis] = length;
    }
    let minimum_rank = if shape[axis] == 0 {
        origin_rank
    } else {
        origin_rank.max(axis + 1)
    };
    trim_transform_output_shape(&mut shape, minimum_rank);
    // The language FFT preserves Single, but promotes logical/native integer
    // inputs to Double. Preferred provider precision is not output authority.
    let precision = if !runmat_accelerate_api::handle_is_logical(original)
        && runmat_accelerate_api::handle_precision(original) == Some(ProviderPrecision::F32)
    {
        ProviderPrecision::F32
    } else {
        ProviderPrecision::F64
    };
    let contract = gpu_helpers::BinaryGpuOutputContract {
        shape,
        storage: GpuTensorStorage::ComplexInterleaved,
        precision: Some(precision),
        integer: None,
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
    };
    // A unit-length complex transform is exactly identity, without arithmetic
    // or mutation. An immutable view of that CURRENT value is safe. It cannot
    // resurrect the original allocation after an earlier nontrivial pass.
    let aliases_current = output.shares_allocation_with(current);
    let aliases_original = output.shares_allocation_with(original);
    let unchanged_axis = current.shape().get(axis).copied().unwrap_or(1) == 1
        && length.unwrap_or(1) == 1
        && runmat_accelerate_api::handle_storage(current) == GpuTensorStorage::ComplexInterleaved;
    let safe_alias = (!aliases_current && !aliases_original)
        || (aliases_current
            && unchanged_axis
            && (!aliases_original || current.shares_allocation_with(original)));
    if !safe_alias
        || !gpu_helpers::binary_gpu_output_matches(&output, current, original, provider, &contract)
    {
        let mut error = error();
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [current, original])
        .map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::super::{transform_complex_tensor, TransformDirection};
    use runmat_value::{ComplexStorage, ComplexTensor};

    #[test]
    fn forward_and_inverse_host_scalar_shapes_are_canonical() {
        for direction in [TransformDirection::Forward, TransformDirection::Inverse] {
            for shape in [vec![], vec![1], vec![1, 1]] {
                let input = ComplexTensor::new(vec![(3.25, -1.5)], shape).unwrap();
                let output =
                    transform_complex_tensor(input, None, Some(1), direction, "fft-shape-test")
                        .unwrap();
                assert_eq!(output.shape(), [1, 1]);
                let ComplexStorage::F64(data) = output.complex_storage() else {
                    panic!("native double")
                };
                assert_eq!((data[0].0, data[0].1), (3.25, -1.5));
            }
            for shape in [vec![2], vec![1, 2], vec![2, 1], vec![1, 1, 1]] {
                let count: usize = shape.iter().product();
                let input = ComplexTensor::new(vec![(0.0, 0.0); count], shape.clone()).unwrap();
                let output =
                    transform_complex_tensor(input, None, Some(1), direction, "fft-shape-test")
                        .unwrap();
                assert_eq!(output.shape(), shape);
            }
        }
    }
}
