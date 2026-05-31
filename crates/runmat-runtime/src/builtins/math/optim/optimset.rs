//! Minimal MATLAB-compatible `optimset` options struct builder.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::common::field_name;
use crate::builtins::math::optim::type_resolvers::optim_options_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "optimset";

const OPTIMSET_OUTPUT_OPTIONS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Options struct for optimization solvers.",
}];

const OPTIMSET_INPUTS_PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option value.",
    },
];

const OPTIMSET_INPUTS_EXISTING_AND_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing options struct to update.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value(s) and additional name/value pairs.",
    },
];

const OPTIMSET_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "options = optimset()",
        inputs: &[],
        outputs: &OPTIMSET_OUTPUT_OPTIONS,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimset(name, value, ...)",
        inputs: &OPTIMSET_INPUTS_PAIRS,
        outputs: &OPTIMSET_OUTPUT_OPTIONS,
    },
    BuiltinSignatureDescriptor {
        label: "options = optimset(oldopts, name, value, ...)",
        inputs: &OPTIMSET_INPUTS_EXISTING_AND_PAIRS,
        outputs: &OPTIMSET_OUTPUT_OPTIONS,
    },
];

const OPTIMSET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMSET.INVALID_ARGUMENT",
    identifier: Some("RunMat:optimset:InvalidArgument"),
    when: "Name/value argument grammar is invalid.",
    message: "optimset: invalid argument",
};

const OPTIMSET_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMSET.INVALID_INPUT",
    identifier: Some("RunMat:optimset:InvalidInput"),
    when: "Option field names are not valid string scalars.",
    message: "optimset: invalid input",
};

const OPTIMSET_ERRORS: [BuiltinErrorDescriptor; 2] = [
    OPTIMSET_ERROR_INVALID_ARGUMENT,
    OPTIMSET_ERROR_INVALID_INPUT,
];

pub const OPTIMSET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OPTIMSET_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &OPTIMSET_ERRORS,
};

fn optimset_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("optimset:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

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
    summary = "Create or update optimization options structures.",
    keywords = "optimset,options,TolX,TolFun,MaxIter,Display",
    type_resolver(optim_options_type),
    descriptor(crate::builtins::math::optim::optimset::OPTIMSET_DESCRIPTOR),
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
                    optimset_error_with_detail(
                        &OPTIMSET_ERROR_INVALID_ARGUMENT,
                        "expected option name/value pairs",
                    )
                })?;
                let name = field_name(&other).map_err(|err| {
                    optimset_error_with_detail(&OPTIMSET_ERROR_INVALID_INPUT, err.message())
                })?;
                fields.insert(canonical_option_name(&name), second);
            }
        }
    }

    let remaining = args.collect::<Vec<_>>();
    if remaining.len() % 2 != 0 {
        return Err(optimset_error_with_detail(
            &OPTIMSET_ERROR_INVALID_ARGUMENT,
            "expected option name/value pairs",
        ));
    }
    for pair in remaining.chunks(2) {
        let name = field_name(&pair[0]).map_err(|err| {
            optimset_error_with_detail(&OPTIMSET_ERROR_INVALID_INPUT, err.message())
        })?;
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

    #[test]
    fn optimset_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = OPTIMSET_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "options = optimset()",
                "options = optimset(name, value, ...)",
                "options = optimset(oldopts, name, value, ...)",
            ]
        );

        let codes: Vec<&str> = OPTIMSET_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec!["RM.OPTIMSET.INVALID_ARGUMENT", "RM.OPTIMSET.INVALID_INPUT"]
        );
    }

    #[test]
    fn optimset_odd_name_value_pairs_use_stable_identifier() {
        let err = block_on(optimset_builtin(vec![Value::from("TolX")])).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:optimset:InvalidArgument"));
    }
}
