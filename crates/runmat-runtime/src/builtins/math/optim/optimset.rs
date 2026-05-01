//! Minimal MATLAB-compatible `optimset` options struct builder.

use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::{field_name, optim_error};
use crate::builtins::math::optim::type_resolvers::optim_options_type;
use crate::BuiltinResult;

const NAME: &str = "optimset";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::optimset")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "optimset",
    op_kind: GpuOpKind::Custom("options"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host metadata construction. GPU values used as option payloads are preserved without gathering.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::optimset")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "optimset",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Option struct construction is host metadata work and does not fuse.",
};

#[runtime_builtin(
    name = "optimset",
    category = "math/optim",
    summary = "Create or update an optimization options structure for fzero and fsolve.",
    keywords = "optimset,options,TolX,TolFun,MaxIter,Display",
    type_resolver(optim_options_type),
    builtin_path = "crate::builtins::math::optim::optimset"
)]
async fn optimset_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let mut fields = StructValue::new();
    let mut args = rest.into_iter();

    if let Some(first) = args.next() {
        match first {
            Value::Struct(existing) => fields = existing,
            other => {
                let second = args.next().ok_or_else(|| {
                    optim_error(NAME, "optimset: expected option name/value pairs")
                })?;
                let name = field_name(&other)?;
                fields.insert(canonical_option_name(&name), second);
            }
        }
    }

    let remaining = args.collect::<Vec<_>>();
    if remaining.len() % 2 != 0 {
        return Err(optim_error(
            NAME,
            "optimset: expected option name/value pairs",
        ));
    }
    for pair in remaining.chunks(2) {
        let name = field_name(&pair[0])?;
        fields.insert(canonical_option_name(&name), pair[1].clone());
    }

    Ok(Value::Struct(fields))
}

fn canonical_option_name(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "tolx" => "TolX".to_string(),
        "tolfun" => "TolFun".to_string(),
        "maxiter" => "MaxIter".to_string(),
        "maxfunevals" => "MaxFunEvals".to_string(),
        "display" => "Display".to_string(),
        _ => name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn optimset_builds_struct_from_pairs() {
        let value = block_on(optimset_builtin(vec![
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("Display"),
            Value::from("off"),
        ]))
        .unwrap();
        match value {
            Value::Struct(options) => {
                assert!(matches!(options.fields.get("TolX"), Some(Value::Num(_))));
                assert!(matches!(
                    options.fields.get("Display"),
                    Some(Value::String(_))
                ));
            }
            other => panic!("unexpected value {other:?}"),
        }
    }
}
