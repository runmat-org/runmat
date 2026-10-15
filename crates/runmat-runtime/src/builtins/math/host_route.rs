//! Shared representation admission for actual canonical numeric host branches.
//! This is not inferred from output residency; owners call it before choosing
//! their host-only implementation and confirm only after that implementation.
use crate::context::{RuntimeContext, VmFusionHostInstruction, VmFusionHostRoute};
use runmat_value::{NumericDType, Value};

pub(crate) fn real_floating_host(value: &Value) -> bool {
    matches!(value, Value::Num(_))
        || matches!(value, Value::Tensor(tensor)
        if matches!(tensor.numeric_dtype(), NumericDType::F32 | NumericDType::F64))
}

pub(crate) fn canonical<'a>(
    runtime: Option<&RuntimeContext>,
    instruction: VmFusionHostInstruction,
    mut values: impl Iterator<Item = &'a Value>,
) -> Option<VmFusionHostRoute> {
    let runtime = runtime?;
    let route = runtime.vm_fusion_host_route()?;
    if !values.all(real_floating_host) || !route.allows_host_preparation(runtime, instruction) {
        return None;
    }
    Some(route.clone())
}
