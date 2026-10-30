//! RunMat `gpuInfo` status string from shared active-provider metadata.

use super::device_metadata::active_device_metadata;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use runmat_builtins::GPU_INFO_ERROR_TOO_MANY_INPUTS;
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, StructValue, Value};

use crate::build_runtime_error;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::gpuinfo")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gpuInfo",
    op_kind: GpuOpKind::Custom("device-summary"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Formats metadata reported by the active provider; no GPU kernels are launched.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::acceleration::gpu::gpuinfo")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gpuInfo",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion—returns a host-resident string.",
};

#[runtime_builtin(
    name = "gpuInfo",
    binding_variant = "default",
    builtin_path = "crate::builtins::acceleration::gpu::gpuinfo"
)]
async fn gpu_info_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(build_runtime_error(GPU_INFO_ERROR_TOO_MANY_INPUTS.message)
            .with_builtin("gpuInfo")
            .with_identifier(
                GPU_INFO_ERROR_TOO_MANY_INPUTS
                    .identifier
                    .expect("gpuInfo arity error has a public identifier"),
            )
            .build()
            .into());
    }
    let info = active_device_metadata().map(|metadata| metadata.into_struct());
    Ok(Value::string(format_summary(info.as_ref())))
}

fn format_summary(info: Option<&StructValue>) -> String {
    match info {
        Some(struct_value) => {
            let mut parts = Vec::new();
            for (key, value) in struct_value.fields.iter() {
                if let Some(fragment) = format_field(key, value) {
                    parts.push(fragment);
                }
            }
            format!("GPU[{}]", parts.join(", "))
        }
        None => "GPU[no provider]".to_string(),
    }
}

fn format_field(key: &str, value: &Value) -> Option<String> {
    match value {
        Value::Int(intv) => Some(format!("{key}={}", int_to_string(intv))),
        Value::Num(n) => Some(format!("{key}={}", Value::Num(*n))),
        Value::Bool(flag) => Some(format!("{key}={}", if *flag { "true" } else { "false" })),
        Value::String(runmat_value::StringElement::Text(text)) => {
            Some(format!("{key}='{}'", escape_single_quotes(text)))
        }
        _ => None,
    }
}

fn int_to_string(value: &IntValue) -> String {
    match value {
        IntValue::I8(v) => v.to_string(),
        IntValue::I16(v) => v.to_string(),
        IntValue::I32(v) => v.to_string(),
        IntValue::I64(v) => v.to_string(),
        IntValue::U8(v) => v.to_string(),
        IntValue::U16(v) => v.to_string(),
        IntValue::U32(v) => v.to_string(),
        IntValue::U64(v) => v.to_string(),
    }
}

fn escape_single_quotes(text: &str) -> String {
    text.replace('\'', "''")
}

#[cfg(test)]
mod tests;
