use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;

use super::OutputPrototypeContext;

#[derive(Clone, Copy)]
pub(super) enum PrototypeClass {
    Real,
    Complex,
}

#[derive(Clone)]
pub(super) enum DevicePreference {
    Host,
    LikeGpu(runmat_accelerate_api::GpuTensorHandle),
}

pub(super) struct LikeAnalysis {
    pub(super) device: DevicePreference,
    pub(super) class: PrototypeClass,
}

#[async_recursion::async_recursion(?Send)]
pub(super) async fn analyse(
    context: OutputPrototypeContext,
    prototype: &Value,
) -> BuiltinResult<LikeAnalysis> {
    match prototype {
        Value::GpuTensor(handle) => Ok(LikeAnalysis {
            device: DevicePreference::LikeGpu(handle.clone()),
            class: PrototypeClass::Real,
        }),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Real,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Complex,
        }),
        other => {
            let gathered = gather(context, other).await?;
            analyse(context, &gathered).await
        }
    }
}

async fn gather(context: OutputPrototypeContext, value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(value)
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, context.identity.name)),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_) => Ok(value.clone()),
        _ => Err(context.described_error(
            context.invalid_argument,
            format!("unsupported prototype for 'like' ({value:?})"),
        )),
    }
}
