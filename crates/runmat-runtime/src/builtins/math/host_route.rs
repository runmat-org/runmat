//! Shared representation admission for actual canonical numeric host branches.
//! This is not inferred from output residency; owners call it before choosing
//! their host-only implementation and confirm only after that implementation.
use crate::context::{RuntimeContext, VmFusionHostInstruction, VmFusionHostRoute};
use runmat_value::{NumericDType, Value};

pub(crate) enum NumericHostRoute {
    Vm(VmFusionHostRoute),
    Prepared(crate::context::preparation::PreparedHostExecution),
}

impl NumericHostRoute {
    pub(crate) fn confirm_host_branch(
        &self,
        runtime: &RuntimeContext,
        instruction: VmFusionHostInstruction,
    ) {
        match self {
            Self::Vm(route) => route.confirm_host_branch(runtime, instruction),
            Self::Prepared(route) => route.confirm(runtime, instruction),
        }
    }
}

pub(crate) fn real_floating_host(value: &Value) -> bool {
    matches!(value, Value::Num(_))
        || matches!(value, Value::Tensor(tensor)
        if matches!(tensor.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
}

pub(crate) fn real_floating_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Num(_) => Some(&[1, 1]),
        Value::Tensor(tensor)
            if matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) =>
        {
            Some(tensor.shape())
        }
        Value::GpuTensor(handle) => {
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)?;
            let element =
                crate::builtins::common::gpu_helpers::expected_handle_numeric_element_type(handle)
                    .ok()?;
            (matches!(
                element,
                runmat_accelerate_api::NumericElementType::F32
                    | runmat_accelerate_api::NumericElementType::F64
            ) && crate::builtins::common::gpu_helpers::numeric_descriptor_matches(
                handle,
                element,
                runmat_accelerate_api::GpuTensorStorage::Real,
            ))
            .then_some(handle.shape())
        }
        _ => None,
    }
}

pub(crate) fn canonical<'a>(
    runtime: Option<&RuntimeContext>,
    instruction: VmFusionHostInstruction,
    mut values: impl Iterator<Item = &'a Value>,
) -> Option<NumericHostRoute> {
    let runtime = runtime?;
    if !values.all(real_floating_host) {
        return None;
    }
    selected(runtime, instruction)
}

/// The caller is an actual canonical host branch after its own class/domain
/// admission. This does not classify arbitrary values or outputs as host work.
pub(crate) fn selected(
    runtime: &RuntimeContext,
    instruction: VmFusionHostInstruction,
) -> Option<NumericHostRoute> {
    if let Some(route) = runtime
        .vm_fusion_host_route()
        .filter(|route| route.allows_host_preparation(runtime, instruction))
    {
        return Some(NumericHostRoute::Vm(route.clone()));
    }
    runtime
        .prepared_host_execution()
        .filter(|route| route.allows(runtime, instruction))
        .map(|route| NumericHostRoute::Prepared(route.clone()))
}
