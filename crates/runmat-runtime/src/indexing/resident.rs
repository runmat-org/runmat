//! Owning resident indexing boundaries shared by reads and mutations.
use crate::builtins::common::gpu_helpers;
use crate::{runtime_error::semantic_error, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderExecutionError};

pub(super) fn owner(handle: &GpuTensorHandle) -> Result<&'static dyn AccelProvider, RuntimeError> {
    gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
        semantic_error(
            "AccelerationProviderUnavailable",
            "No acceleration provider owns the indexed gpuArray",
        )
    })
}

pub(super) fn failure(context: &str, source: ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        semantic_error(
            "AccelerationOperationFailed",
            format!("{context}: {source}"),
        ),
        source,
    )
}

pub(super) fn admit_read(
    mut output: GpuTensorHandle,
    input: &GpuTensorHandle,
    shape: &[usize],
) -> Result<runmat_value::Value, RuntimeError> {
    let provider = owner(input)?;
    let logical = runmat_accelerate_api::handle_is_logical(input);
    let valid = output.shape() == shape
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|actual| std::ptr::eq(actual, provider))
        && !output.shares_allocation_with(input)
        && runmat_accelerate_api::handle_storage(&output)
            == runmat_accelerate_api::handle_storage(input)
        && runmat_accelerate_api::handle_is_logical(&output) == logical
        && (logical || output.physical().element_type() == input.physical().element_type());
    if !valid {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            semantic_error(
                "AccelerationOperationFailed",
                "Indexing provider returned an invalid class, shape, owner or alias",
            ),
        ));
    }
    gpu_helpers::propagate_output_provenance(&mut output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}

pub(super) fn empty_read(
    input: &GpuTensorHandle,
    shape: &[usize],
) -> Result<runmat_value::Value, RuntimeError> {
    use runmat_accelerate_api::{HostNumericDataView as Data, NumericElementType as Element};
    let provider = owner(input)?;
    let result = if runmat_accelerate_api::handle_is_logical(input) {
        runmat_accelerate_api::upload_logical_protecting(
            provider,
            &runmat_accelerate_api::HostLogicalTensorView { data: &[], shape },
            &[input],
        )
    } else {
        let data = match input.physical().element_type() {
            Element::F64 => Data::F64(&[]),
            Element::F32 => Data::F32(&[]),
            Element::I8 => Data::I8(&[]),
            Element::I16 => Data::I16(&[]),
            Element::I32 => Data::I32(&[]),
            Element::I64 => Data::I64(&[]),
            Element::U8 => Data::U8(&[]),
            Element::U16 => Data::U16(&[]),
            Element::U32 => Data::U32(&[]),
            Element::U64 => Data::U64(&[]),
        };
        runmat_accelerate_api::upload_numeric_protecting(
            provider,
            &runmat_accelerate_api::HostNumericTensorView {
                data,
                shape,
                storage: runmat_accelerate_api::handle_storage(input),
            },
            &[input],
        )
    }
    .map_err(|cause| {
        failure(
            "empty indexed result",
            ProviderExecutionError::from_source(cause),
        )
    })?;
    admit_read(result, input, shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{GpuTensorStorage, HostNumericDataView, HostNumericTensorView};
    #[test]
    fn empty_reads_preserve_single_complex_and_logical_producer_classes() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            for storage in [GpuTensorStorage::Real, GpuTensorStorage::ComplexInterleaved] {
                let data = if storage == GpuTensorStorage::Real {
                    &[3.0_f32][..]
                } else {
                    &[3.0_f32, 4.0][..]
                };
                let input = runmat_accelerate_api::upload_numeric_protecting(
                    provider,
                    &HostNumericTensorView {
                        data: HostNumericDataView::F32(data),
                        shape: &[1, 1],
                        storage,
                    },
                    &[],
                )
                .unwrap();
                let runmat_value::Value::GpuTensor(output) = empty_read(&input, &[1, 0]).unwrap()
                else {
                    panic!("resident result");
                };
                assert_eq!(output.shape(), &[1, 0]);
                assert_eq!(
                    output.physical().element_type(),
                    input.physical().element_type()
                );
                assert_eq!(runmat_accelerate_api::handle_storage(&output), storage);
                assert!(!output.shares_allocation_with(&input));
            }
            let input = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1],
                    shape: &[1, 1],
                })
                .unwrap();
            let runmat_value::Value::GpuTensor(output) = empty_read(&input, &[0, 1]).unwrap()
            else {
                panic!("resident logical");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&output));
            assert_eq!(output.shape(), &[0, 1]);
            assert!(!output.shares_allocation_with(&input));
        });
    }
}
