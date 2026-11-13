//! Existing native upload owner, borrowed factors and bounded protected refs.
use super::*;
use crate::builtins::common::gpu_helpers::host_upload::GpuHostUpload;
use runmat_accelerate_api::{
    GpuTensorStorage, HostNumericDataView, HostNumericTensorView, NumericElementType,
};
use runmat_value::NumericDType;
use std::error::Error;

pub(super) fn is_pre_effect_unsupported(error: &crate::RuntimeError) -> bool {
    // This is the upload owner's immediate typed cause, not permission found
    // inside arbitrary language or cleanup chains.
    error
        .source()
        .and_then(|source| source.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>())
        .is_some_and(|cause| cause.is_unsupported())
}

pub(super) fn publish_factor(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
    output_source: &GpuTensorHandle,
    source: &GpuTensorHandle,
    operands: &[GpuTensorHandle],
    outputs: &bundle::PublishedFactors,
) -> BuiltinResult<GpuTensorHandle> {
    // At most two coefficient operands and four roles, plus prototype. Only
    // stack references, never cloned handles or a payload mirror.
    if operands.len() > 2 {
        return Err(eig_internal_error("eig: coefficient owner extent changed"));
    }
    let mut protected = [source; 7];
    let mut count = 0;
    for handle in std::iter::once(source)
        .chain(operands)
        .chain(outputs.handles())
    {
        protected[count] = handle;
        count += 1;
    }
    let protected = &protected[..count];
    let output = match value {
        Value::Tensor(tensor) => {
            let data = match tensor.numeric_dtype() {
                NumericDType::F32 => {
                    HostNumericDataView::F32(tensor.as_f32_slice().expect("single factor"))
                }
                NumericDType::F64 => {
                    HostNumericDataView::F64(tensor.as_f64_slice().expect("double factor"))
                }
                _ => return Err(eig_internal_error("eig: nonfloating host factor")),
            };
            publish_real(
                runtime,
                output_source,
                &HostNumericTensorView {
                    data,
                    shape: tensor.shape(),
                    storage: GpuTensorStorage::Real,
                },
                protected,
            )?
        }
        Value::ComplexTensor(tensor) => {
            let element = match tensor.numeric_dtype() {
                NumericDType::F32 => NumericElementType::F32,
                NumericDType::F64 => NumericElementType::F64,
                _ => return Err(eig_internal_error("eig: nonfloating complex factor")),
            };
            let lanes = tensor
                .len()
                .checked_mul(2)
                .ok_or_else(|| eig_internal_error("eig: complex transfer extent overflow"))?;
            let scratch =
                crate::context::host_allocation::HostAllocationPlan::numeric_workspace_bytes(
                    tensor.numeric_dtype(),
                    lanes,
                )
                .map_err(|cause| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        runtime,
                        "RunMat:eig:Allocation",
                        "eig complex transfer workspace",
                        cause,
                    )
                })?;
            let mut upload = GpuHostUpload::prepare_borrowed_native_output_in_optional_context(
                runtime,
                BUILTIN_NAME,
                output_source,
                tensor.shape(),
                element,
                GpuTensorStorage::ComplexInterleaved,
                scratch,
            )?;
            upload.publish_complex_protecting(tensor, tensor.shape(), protected)?
        }
        Value::Num(number) => publish_real(
            runtime,
            output_source,
            &HostNumericTensorView {
                data: HostNumericDataView::F64(std::slice::from_ref(number)),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            },
            protected,
        )?,
        Value::Complex(real, imaginary) => publish_real(
            runtime,
            output_source,
            &HostNumericTensorView {
                data: HostNumericDataView::F64(&[*real, *imaginary]),
                shape: &[1, 1],
                storage: GpuTensorStorage::ComplexInterleaved,
            },
            protected,
        )?,
        _ => return Err(eig_internal_error("eig: invalid host factor class")),
    };
    // The upload owner's prototype already has the exact combined intent.
    // Original source and every operand remain protected above.
    Ok(output)
}

fn publish_real(
    runtime: Option<&crate::context::RuntimeContext>,
    source: &GpuTensorHandle,
    view: &HostNumericTensorView<'_>,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<GpuTensorHandle> {
    let mut upload = GpuHostUpload::prepare_borrowed_native_output_in_optional_context(
        runtime,
        BUILTIN_NAME,
        source,
        view.shape,
        view.data.element_type(),
        view.storage,
        0,
    )?;
    upload.publish_protecting(view, protected)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_bundle_retry_requires_immediate_typed_pre_effect_unsupported() {
        for mode in 0..3 {
            let cause = runmat_accelerate_api::ProviderExecutionError::from(
                runmat_accelerate_api::unsupported_provider_operation("not implemented"),
            );
            let cause = match mode {
                1 => cause.with_unproven_effects(),
                2 => cause.after_effect(),
                _ => cause,
            };
            let error = crate::runtime_error::with_provider_execution_source(
                eig_internal_error("transfer failed"),
                cause,
            );
            assert_eq!(is_pre_effect_unsupported(&error), mode == 0);
        }
        let spoof = build_runtime_error("unsupported")
            .with_identifier("RunMat:Unsupported")
            .build();
        assert!(!is_pre_effect_unsupported(&spoof));
        let resource = crate::context::host_allocation::host_allocation_runtime_error(
            None,
            "RunMat:eig:Allocation",
            "transfer denied",
            crate::context::host_allocation::HostAllocationError::Cancelled,
        );
        assert!(!is_pre_effect_unsupported(&resource));
    }
}
