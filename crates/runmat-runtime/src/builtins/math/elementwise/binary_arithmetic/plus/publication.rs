//! Own the computed sum before prototype adaptation and host-route confirmation.

use runmat_value::{NumericDType, Value};

use super::super::output_prototype::{apply_output_template, ResolvedOutputTemplate};
use super::OUTPUT_PROTOTYPE_CONTEXT;
use crate::context::{RuntimeContext, VmFusionHostInstruction};
use crate::execution::RootedValue;
use crate::BuiltinResult;

pub(super) async fn finish(
    base: Value,
    template: &ResolvedOutputTemplate,
    host_branch: bool,
    unqualified_template: bool,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<RootedValue> {
    let base = RootedValue::new(base).map_err(|cause| cause.into_runtime_error())?;
    let value =
        apply_output_template(OUTPUT_PROTOTYPE_CONTEXT, base.value().clone(), template).await?;
    let value = RootedValue::new(value).map_err(|cause| cause.into_runtime_error())?;
    if unqualified_template
        && host_branch
        && matches!(value.value(), Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
    {
        if let Some(runtime) = runtime {
            if let Some(route) = runtime.vm_fusion_host_route() {
                route.confirm_host_branch(runtime, VmFusionHostInstruction::ElementwiseBinary);
            }
        }
    }
    Ok(value)
}
