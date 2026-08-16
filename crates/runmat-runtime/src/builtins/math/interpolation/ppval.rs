//! Evaluate piecewise-polynomial structs produced by `spline` and `pchip`.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{evaluate_pp, pp_from_value, query_points, Extrapolation};
use crate::{build_runtime_error, RuntimeError};

const NAME: &str = "ppval";

const PPVAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Evaluated piecewise-polynomial values at query points.",
}];

const PPVAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "pp",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Piecewise-polynomial struct with form/breaks/coefs fields.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
];

const PPVAL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Vq = ppval(pp, Xq)",
    inputs: &PPVAL_INPUTS,
    outputs: &PPVAL_OUTPUT,
}];

const PPVAL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PPVAL.INVALID_ARGUMENT",
    identifier: Some("RunMat:ppval:InvalidArgument"),
    when: "The piecewise-polynomial struct shape or field grammar is invalid.",
    message: "ppval: invalid argument",
};

const PPVAL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PPVAL.INVALID_INPUT",
    identifier: Some("RunMat:ppval:InvalidInput"),
    when: "Query points or polynomial coefficients are invalid for evaluation.",
    message: "ppval: invalid input",
};

const PPVAL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PPVAL.INTERNAL",
    identifier: Some("RunMat:ppval:Internal"),
    when: "Piecewise-polynomial output assembly fails due to internal tensor construction paths.",
    message: "ppval: internal interpolation failure",
};

const PPVAL_ERRORS: [BuiltinErrorDescriptor; 3] = [
    PPVAL_ERROR_INVALID_ARGUMENT,
    PPVAL_ERROR_INVALID_INPUT,
    PPVAL_ERROR_INTERNAL,
];

pub const PPVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PPVAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PPVAL_ERRORS,
};

const PPVAL_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ppval-integer-query-points",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ppval accepts typed-integer query points as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PpvalIntegerQueryPointsExtension"),
};
pub const PPVAL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [PPVAL_INTEGER_QUERY_EXTENSION];
const PPVAL_INTEGER_QUERY_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xq",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double query points. RunMat admits typed integers only when every value is exactly representable at the floating piecewise-polynomial boundary.",
    }];
pub const PPVAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Vq = ppval(pp,integer_Xq)",
        inputs: &PPVAL_INTEGER_QUERY_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer query points are gated before gather and cross once into the polynomial evaluation domain.",
    }];

fn ppval_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::ppval")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("piecewise-polynomial-eval"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Initial implementation evaluates pp structs on the CPU after gathering GPU query points.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::interpolation::ppval")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "ppval is currently a runtime sink.",
};

fn ppval_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.get(1) {
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        _ => Type::tensor(),
    }
}

#[runtime_builtin(
    name = "ppval",
    category = "math/interpolation",
    summary = "Evaluate a piecewise-polynomial structure at query points.",
    keywords = "ppval,piecewise polynomial,spline,pchip",
    accel = "sink",
    sink = true,
    type_resolver(ppval_type),
    descriptor(crate::builtins::math::interpolation::ppval::PPVAL_DESCRIPTOR),
    extensions(crate::builtins::math::interpolation::ppval::PPVAL_EXTENSIONS),
    integer_capabilities(crate::builtins::math::interpolation::ppval::PPVAL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::interpolation::ppval"
)]
async fn ppval_builtin(pp: Value, xq: Value) -> crate::BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&xq, NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &xq,
        &PPVAL_INTEGER_QUERY_EXTENSION,
        NAME,
        "query point",
    )
    .await?;
    let parsed = pp_from_value(pp, NAME).await.map_err(|err| {
        ppval_error_with_message(err.message().to_string(), &PPVAL_ERROR_INVALID_ARGUMENT)
    })?;
    let query = query_points(xq, NAME).await.map_err(|err| {
        ppval_error_with_message(err.message().to_string(), &PPVAL_ERROR_INVALID_INPUT)
    })?;
    evaluate_pp(&parsed, &query, &Extrapolation::Extrapolate, NAME)
        .map_err(|err| ppval_error_with_message(err.message().to_string(), &PPVAL_ERROR_INTERNAL))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::interpolation::pp::{build_spline_pp, pp_to_value, NumericSeries};
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    #[test]
    fn ppval_evaluates_spline_struct() {
        let series = NumericSeries {
            x: vec![1.0, 2.0, 3.0],
            y: vec![1.0, 4.0, 9.0],
            series: 1,
            trailing_shape: Vec::new(),
        };
        let pp = pp_to_value(
            build_spline_pp(&series, "spline").expect("spline"),
            "spline",
        )
        .expect("pp");
        let query = Value::Tensor(Tensor::new(vec![1.5, 2.5], vec![1, 2]).expect("tensor"));
        let value = block_on(ppval_builtin(pp, query)).expect("ppval");
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert!((tensor.materialize_f64()[0] - 2.25).abs() < 1e-10);
        assert!((tensor.materialize_f64()[1] - 6.25).abs() < 1e-10);
    }

    #[test]
    fn ppval_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = PPVAL_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(labels, vec!["Vq = ppval(pp, Xq)"]);
    }

    #[test]
    fn ppval_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = PPVAL_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.PPVAL.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.PPVAL.INVALID_INPUT"));
        assert!(codes.contains(&"RM.PPVAL.INTERNAL"));
    }

    #[test]
    fn ppval_invalid_first_arg_uses_stable_identifier() {
        let err = block_on(ppval_builtin(
            Value::Num(42.0),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("tensor")),
        ))
        .expect_err("expected ppval argument error");
        assert_eq!(err.identifier(), PPVAL_ERROR_INVALID_ARGUMENT.identifier);
    }
}
