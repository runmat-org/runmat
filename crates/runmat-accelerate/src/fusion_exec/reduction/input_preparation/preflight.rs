//! Query-free whole-set source carrier, owner, lease and class preflight.
use super::{invalid, PreflightedInputs};
use crate::fusion_exec::ResolvedReductionProgram;
use crate::placement::tensor_representation;
use runmat_accelerate_api::{
    same_provider_instance, AccelProvider, GpuTensorStorage, NumericElementType,
    ProviderElementType, ProviderExecutionResult, ProviderLayout, ProviderPrecision,
    ProviderResidency, ProviderStorage,
};
use runmat_runtime::builtins::common::gpu_helpers::{
    exact_provider_for_handle, gpu_class_metadata_matches,
};
use runmat_value::{NumericDType, Value};

fn host_class_is_exact(value: &Value, precision: runmat_accelerate_api::ProviderPrecision) -> bool {
    match (value, precision) {
        (Value::Num(_), runmat_accelerate_api::ProviderPrecision::F64) => true,
        (Value::Tensor(t), runmat_accelerate_api::ProviderPrecision::F32) => {
            t.numeric_dtype() == NumericDType::F32 && t.as_f32_slice().is_some()
        }
        (Value::Tensor(t), runmat_accelerate_api::ProviderPrecision::F64) => {
            t.numeric_dtype() == NumericDType::F64 && t.as_f64_slice().is_some()
        }
        _ => false,
    }
}

/// Resolve the entire ordered data set before any provider config, candidate,
/// upload, or cast proposal query. This holds the exact program borrow and
/// provider instance; it is not a serializable physical authorization token.
pub(in crate::fusion_exec::reduction) fn preflight_sources<'p, 'a>(
    provider: &'static dyn AccelProvider,
    program: &'p ResolvedReductionProgram<'a>,
) -> ProviderExecutionResult<PreflightedInputs<'p, 'a>> {
    let mut sources = Vec::with_capacity(program.data_inputs().len());
    for input in program.data_inputs() {
        let source = tensor_representation(input.value, provider.precision())
            .ok_or_else(|| invalid("typed reduction source has no numeric representation"))?;
        let precision = match source.element_type {
            ProviderElementType::F32 => ProviderPrecision::F32,
            ProviderElementType::F64 => ProviderPrecision::F64,
            _ => return Err(invalid("typed reduction source is not real floating")),
        };
        if source.storage != ProviderStorage::DenseReal
            || source.checked_byte_len().is_none()
            || !matches!(
                source.residency,
                ProviderResidency::Host | ProviderResidency::Device
            )
        {
            return Err(invalid(
                "typed reduction source class or extent unavailable",
            ));
        }
        let shape: &[usize] = match input.value {
            Value::GpuTensor(handle) => handle.shape(),
            Value::Tensor(tensor) => tensor.shape(),
            Value::Num(_) => &[1, 1],
            _ => return Err(invalid("typed reduction source carrier unavailable")),
        };
        let exact_shape: Vec<u64> = shape
            .iter()
            .map(|&extent| {
                u64::try_from(extent).map_err(|_| invalid("typed source extent exceeds u64"))
            })
            .collect::<ProviderExecutionResult<_>>()?;
        if source.shape != exact_shape {
            return Err(invalid("typed reduction source shape is not exact"));
        }
        match input.value {
            Value::GpuTensor(handle) => {
                let element_type = match precision {
                    ProviderPrecision::F32 => NumericElementType::F32,
                    ProviderPrecision::F64 => NumericElementType::F64,
                };
                if source.residency != ProviderResidency::Device
                    || handle.device_id() != provider.device_id()
                    || !exact_provider_for_handle(handle)
                        .is_some_and(|owner| same_provider_instance(owner, provider))
                    || handle.view().validate(handle.physical()).is_err()
                    || handle.allocation_lease().validate_live().is_err()
                    || !gpu_class_metadata_matches(handle, Some(precision), None, false)
                    || handle.physical().element_type() != element_type
                    || handle.physical().storage() != GpuTensorStorage::Real
                    || handle.physical().element_capacity()
                        < source
                            .checked_element_count()
                            .expect("checked representation")
                    || handle.physical().payload_bytes()
                        < source.checked_byte_len().expect("checked representation")
                {
                    return Err(invalid(
                        "typed reduction resident source lacks exact owner or view",
                    ));
                }
            }
            value => {
                if source.residency != ProviderResidency::Host
                    || source.layout != ProviderLayout::ColumnMajorContiguous
                    || !host_class_is_exact(value, precision)
                {
                    return Err(invalid(
                        "typed reduction host source class or carrier changed",
                    ));
                }
            }
        }
        sources.push(source);
    }
    Ok(PreflightedInputs {
        program,
        owner: provider,
        sources,
    })
}
