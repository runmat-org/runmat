use super::super::{
    error_with_detail, gpu_helpers, tensor, BuiltinResult, IntegerClass, Tensor,
    ERROR_INVALID_INPUT,
};
use super::BitBuffer;

pub(in crate::builtins::math::bitwise::engine) enum AdmittedBuffer<T> {
    Host(T),
    Resident(runmat_accelerate_api::GpuTensorHandle),
}

impl<T> AdmittedBuffer<T> {
    pub(in crate::builtins::math::bitwise::engine) fn host(&self) -> Option<&T> {
        match self {
            Self::Host(value) => Some(value),
            Self::Resident(_) => None,
        }
    }
    pub(in crate::builtins::math::bitwise::engine) fn shape(
        &self,
        host: fn(&T) -> &[usize],
    ) -> &[usize] {
        match self {
            Self::Host(value) => host(value),
            Self::Resident(handle) => handle.shape(),
        }
    }
    pub(in crate::builtins::math::bitwise::engine) async fn materialize(
        self,
        name: &'static str,
        convert: impl FnOnce(Tensor) -> BuiltinResult<T>,
    ) -> BuiltinResult<T> {
        match self {
            Self::Host(value) => Ok(value),
            Self::Resident(handle) => {
                gpu_helpers::require_resident_compute_owner(&handle, || {
                    error_with_detail(
                        name,
                        &ERROR_INVALID_INPUT,
                        "no provider owns the resident operand",
                    )
                })?;
                let value = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error))?;
                convert(value)
            }
        }
    }
}

impl AdmittedBuffer<BitBuffer> {
    pub(in crate::builtins::math::bitwise::engine) fn result_metadata(
        &self,
    ) -> (Option<IntegerClass>, bool) {
        match self {
            Self::Host(value) => (value.output_class, value.is_scalar),
            Self::Resident(handle) => (
                runmat_accelerate_api::handle_integer_class(handle),
                tensor::element_count(handle.shape()) == 1,
            ),
        }
    }
    pub(in crate::builtins::math::bitwise::engine) fn bit_width(
        &self,
        assumed: Option<IntegerClass>,
    ) -> u32 {
        match self {
            Self::Host(value) => value.compute_class,
            Self::Resident(handle) => {
                runmat_accelerate_api::handle_integer_class(handle).or(assumed)
            }
        }
        .map_or(64, IntegerClass::bit_width)
    }
}

pub(super) fn admit_resident(
    name: &'static str,
    handle: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<()> {
    let element_type = gpu_helpers::expected_handle_numeric_element_type(handle)
        .map_err(|error| error_with_detail(name, &ERROR_INVALID_INPUT, error))?;
    if !gpu_helpers::numeric_descriptor_matches(
        handle,
        element_type,
        runmat_accelerate_api::GpuTensorStorage::Real,
    ) {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "resident bitwise operands require explicit real storage",
        ));
    }
    if runmat_accelerate_api::numeric_readback_for_handle(handle).is_err() {
        return Err(error_with_detail(
            name,
            &ERROR_INVALID_INPUT,
            "no provider owns the resident operand",
        ));
    }
    Ok(())
}
