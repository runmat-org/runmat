//! Shared resident admission and result ownership for signal convolutions.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderExecutionError};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers::{self, BinaryGpuOutputContract};
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};

#[cfg(test)]
pub(crate) mod test_support;

mod coefficients;
pub(crate) use coefficients::materialize_host_coefficients;
pub(crate) mod placement;
mod selector;
pub(crate) use selector::{convolution_selector_text, ConvolutionSelectorText};

fn terminal(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

pub(crate) fn checked_vector_elements(shape: &[usize]) -> Option<usize> {
    let elements = shape
        .iter()
        .try_fold(1_usize, |product, dim| product.checked_mul(*dim))?;
    if shape.len() > 2 && shape[2..].iter().any(|dim| *dim != 1) {
        return None;
    }
    let (rows, cols) = match shape {
        [] => (1, 1),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    };
    (rows <= 1 || cols <= 1 || elements == 0).then_some(elements)
}

pub(crate) fn checked_matrix_dims(shape: &[usize]) -> Option<(usize, usize)> {
    let (rows, cols) = match shape {
        [] => (1, 1),
        [rows] => (*rows, 1),
        [rows, cols, ..] => (*rows, *cols),
    };
    if shape.len() > 2 && shape[2..].iter().any(|dim| *dim != 1) {
        return None;
    }
    rows.checked_mul(cols).map(|_| (rows, cols))
}

/// Authenticate every borrowed operand before the first provider operation.
pub(crate) fn admit_resident_operand(
    value: &Value,
    error: impl Fn() -> RuntimeError,
) -> BuiltinResult<()> {
    let Value::GpuTensor(handle) = value else {
        return Ok(());
    };
    crate::builtins::common::validation::validate_resident_metadata(value).map_err(|cause| {
        let mut failure = terminal(error());
        failure.span = cause.span;
        failure.source = Some(Box::new(cause));
        failure
    })?;
    handle.allocation_lease().validate_live().map_err(|cause| {
        crate::runtime_error::with_provider_execution_source(
            terminal(error()),
            ProviderExecutionError::from_source(cause),
        )
    })?;
    if handle
        .shape()
        .iter()
        .try_fold(1_usize, |product, dim| product.checked_mul(*dim))
        .is_none()
    {
        return Err(terminal(error()));
    }
    Ok(())
}

/// Publish only a physically valid, distinct provider result. A rejected result
/// consumes its owning reference while borrowed operands remain live.
pub(crate) fn admit_binary_output(
    mut output: GpuTensorHandle,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    contract: &BinaryGpuOutputContract,
    error: impl Fn() -> RuntimeError,
) -> BuiltinResult<Value> {
    if !gpu_helpers::binary_gpu_output_matches(&output, left, right, provider, contract) {
        return Err(gpu_helpers::reject_gpu_output(output, terminal(error())));
    }
    if let Err(cause) = gpu_helpers::propagate_output_provenance(&mut output, [left, right]) {
        let mut failure = terminal(error());
        failure.source = Some(Box::new(cause));
        return Err(gpu_helpers::reject_gpu_output(output, failure));
    }
    gpu_helpers::resident_gpu_value(output).map_err(|cause| {
        let mut failure = terminal(error());
        failure.source = Some(Box::new(cause));
        failure
    })
}

#[cfg(test)]
mod tests {
    use super::{checked_matrix_dims, checked_vector_elements};

    #[test]
    fn structural_admission_is_checked_and_does_not_flatten_nd_inputs() {
        assert_eq!(checked_vector_elements(&[]), Some(1));
        assert_eq!(checked_vector_elements(&[1, 4]), Some(4));
        assert_eq!(checked_vector_elements(&[4, 1, 1]), Some(4));
        assert_eq!(checked_vector_elements(&[1, 1, 4]), None);
        assert_eq!(checked_vector_elements(&[2, 2]), None);
        assert_eq!(checked_vector_elements(&[0, 4]), Some(0));
        assert_eq!(checked_vector_elements(&[usize::MAX, 2]), None);
        assert_eq!(checked_matrix_dims(&[2, 3, 1]), Some((2, 3)));
        assert_eq!(checked_matrix_dims(&[2, 3, 2]), None);
        assert_eq!(checked_matrix_dims(&[usize::MAX, 2]), None);
    }
}
