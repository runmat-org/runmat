//! MATLAB-compatible `spline` builtin backed by piecewise-polynomial structs.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{
    build_spline_pp, evaluate_pp, pp_to_value, query_points, series_from_values, Extrapolation,
};
use crate::{build_runtime_error, RuntimeError};

const NAME: &str = "spline";

const SPLINE_OUTPUT_PP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pp",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Piecewise-polynomial struct representation.",
}];

const SPLINE_OUTPUT_VALUES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Spline-evaluated values at query points.",
}];

const SPLINE_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Strictly increasing sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
];

const SPLINE_INPUTS_X_Y_XQ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Strictly increasing sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
];

const SPLINE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pp = spline(X, Y)",
        inputs: &SPLINE_INPUTS_X_Y,
        outputs: &SPLINE_OUTPUT_PP,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = spline(X, Y, Xq)",
        inputs: &SPLINE_INPUTS_X_Y_XQ,
        outputs: &SPLINE_OUTPUT_VALUES,
    },
];

const SPLINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:spline:InvalidArgument"),
    when: "Argument count or interpolation option grammar is invalid.",
    message: "spline: invalid argument",
};

const SPLINE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLINE.INVALID_INPUT",
    identifier: Some("RunMat:spline:InvalidInput"),
    when: "Sample/query inputs are invalid for spline construction or evaluation.",
    message: "spline: invalid input",
};

const SPLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLINE.INTERNAL",
    identifier: Some("RunMat:spline:Internal"),
    when: "Spline coefficient generation or tensor assembly fails internally.",
    message: "spline: internal interpolation failure",
};

const SPLINE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SPLINE_ERROR_INVALID_ARGUMENT,
    SPLINE_ERROR_INVALID_INPUT,
    SPLINE_ERROR_INTERNAL,
];

pub const SPLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPLINE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPLINE_ERRORS,
};

fn spline_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn spline_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    spline_error_with_message(
        format!(
            "{}: {}",
            SPLINE_ERROR_INVALID_ARGUMENT.message,
            detail.as_ref()
        ),
        &SPLINE_ERROR_INVALID_ARGUMENT,
    )
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::spline")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("cubic-spline"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Spline coefficient construction and evaluation currently run on the CPU reference path after gathering GPU inputs.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::spline"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Spline builds a piecewise-polynomial representation and is not fused.",
};

fn spline_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.len() {
        0 | 1 => Type::Unknown,
        2 => Type::Struct { known_fields: None },
        _ => match args.get(2) {
            Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
            Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
                shape: shape.clone(),
            },
            _ => Type::tensor(),
        },
    }
}

#[runtime_builtin(
    name = "spline",
    category = "math/interpolation",
    summary = "Cubic spline interpolation and piecewise-polynomial construction.",
    keywords = "spline,cubic interpolation,pp,ppval",
    accel = "sink",
    sink = true,
    type_resolver(spline_type),
    descriptor(crate::builtins::math::interpolation::spline::SPLINE_DESCRIPTOR),
    builtin_path = "crate::builtins::math::interpolation::spline"
)]
async fn spline_builtin(x: Value, y: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let series = series_from_values(x, y, NAME).await.map_err(|err| {
        spline_error_with_message(err.message().to_string(), &SPLINE_ERROR_INVALID_INPUT)
    })?;
    let pp = build_spline_pp(&series, NAME).map_err(|err| {
        spline_error_with_message(err.message().to_string(), &SPLINE_ERROR_INTERNAL)
    })?;
    match rest.len() {
        0 => pp_to_value(pp, NAME).map_err(|err| {
            spline_error_with_message(err.message().to_string(), &SPLINE_ERROR_INTERNAL)
        }),
        1 => {
            let query = query_points(rest.into_iter().next().expect("query"), NAME)
                .await
                .map_err(|err| {
                    spline_error_with_message(
                        err.message().to_string(),
                        &SPLINE_ERROR_INVALID_INPUT,
                    )
                })?;
            evaluate_pp(&pp, &query, &Extrapolation::Extrapolate, NAME).map_err(|err| {
                spline_error_with_message(err.message().to_string(), &SPLINE_ERROR_INTERNAL)
            })
        }
        _ => Err(spline_invalid_argument("too many input arguments")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{StructValue, Tensor};

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    #[test]
    fn spline_two_arg_returns_pp_struct() {
        let value = block_on(spline_builtin(
            row(&[1.0, 2.0, 3.0]),
            row(&[1.0, 4.0, 9.0]),
            vec![],
        ))
        .expect("spline");
        let Value::Struct(StructValue { fields }) = value else {
            panic!("expected struct");
        };
        assert!(fields.contains_key("breaks"));
        assert!(fields.contains_key("coefs"));
    }

    #[test]
    fn spline_three_arg_evaluates() {
        let value = block_on(spline_builtin(
            row(&[1.0, 2.0, 3.0]),
            row(&[1.0, 4.0, 9.0]),
            vec![Value::Num(1.5)],
        ))
        .expect("spline");
        let Value::Num(result) = value else {
            panic!("expected scalar");
        };
        assert!((result - 2.25).abs() < 1e-10);
    }

    #[test]
    fn spline_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = SPLINE_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"pp = spline(X, Y)"));
        assert!(labels.contains(&"Vq = spline(X, Y, Xq)"));
    }

    #[test]
    fn spline_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = SPLINE_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.SPLINE.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.SPLINE.INVALID_INPUT"));
        assert!(codes.contains(&"RM.SPLINE.INTERNAL"));
    }

    #[test]
    fn spline_too_many_args_uses_stable_identifier() {
        let err = block_on(spline_builtin(
            row(&[1.0, 2.0]),
            row(&[1.0, 4.0]),
            vec![Value::Num(1.5), Value::Num(2.5)],
        ))
        .expect_err("expected spline argument error");
        assert_eq!(err.identifier(), SPLINE_ERROR_INVALID_ARGUMENT.identifier);
    }
}
