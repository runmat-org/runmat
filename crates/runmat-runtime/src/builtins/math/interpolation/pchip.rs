//! MATLAB-compatible `pchip` builtin for shape-preserving cubic interpolation.

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

use super::pp::{
    build_pchip_pp, evaluate_pp, pp_to_value, query_points, series_from_values, Extrapolation,
};
use crate::{build_runtime_error, RuntimeError};

const NAME: &str = "pchip";

const PCHIP_OUTPUT_PP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pp",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Piecewise-polynomial struct representation.",
}];

const PCHIP_OUTPUT_VALUES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "PCHIP-evaluated values at query points.",
}];

const PCHIP_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
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

const PCHIP_INPUTS_X_Y_XQ: [BuiltinParamDescriptor; 3] = [
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

const PCHIP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pp = pchip(X, Y)",
        inputs: &PCHIP_INPUTS_X_Y,
        outputs: &PCHIP_OUTPUT_PP,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = pchip(X, Y, Xq)",
        inputs: &PCHIP_INPUTS_X_Y_XQ,
        outputs: &PCHIP_OUTPUT_VALUES,
    },
];

const PCHIP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCHIP.INVALID_ARGUMENT",
    identifier: Some("RunMat:pchip:InvalidArgument"),
    when: "Argument count or interpolation option grammar is invalid.",
    message: "pchip: invalid argument",
};

const PCHIP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCHIP.INVALID_INPUT",
    identifier: Some("RunMat:pchip:InvalidInput"),
    when: "Sample/query inputs are invalid for PCHIP construction or evaluation.",
    message: "pchip: invalid input",
};

const PCHIP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCHIP.INTERNAL",
    identifier: Some("RunMat:pchip:Internal"),
    when: "PCHIP coefficient generation or tensor assembly fails internally.",
    message: "pchip: internal interpolation failure",
};

const PCHIP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    PCHIP_ERROR_INVALID_ARGUMENT,
    PCHIP_ERROR_INVALID_INPUT,
    PCHIP_ERROR_INTERNAL,
];

pub const PCHIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PCHIP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PCHIP_ERRORS,
};

const PCHIP_INTEGER_SAMPLE_LOCATIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pchip-integer-sample-locations",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pchip accepts typed-integer sample locations as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PchipIntegerSampleLocationsExtension"),
    };
const PCHIP_INTEGER_SAMPLE_VALUES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pchip-integer-sample-values",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pchip accepts typed-integer sample values as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PchipIntegerSampleValuesExtension"),
    };
const PCHIP_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pchip-integer-query-points",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pchip accepts typed-integer query points as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PchipIntegerQueryPointsExtension"),
};
pub const PCHIP_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    PCHIP_INTEGER_SAMPLE_LOCATIONS_EXTENSION,
    PCHIP_INTEGER_SAMPLE_VALUES_EXTENSION,
    PCHIP_INTEGER_QUERY_EXTENSION,
];

const PCHIP_INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double sample locations. RunMat admits typed integers only when each value is exactly representable at the binary64 interpolation boundary.",
    }];
const PCHIP_INTEGER_Y_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double sample values. RunMat admits typed integers only through the explicit floating interpolation boundary.",
    }];
const PCHIP_INTEGER_XQ_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xq",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double query points. RunMat admits typed integers only when exact binary64 conversion is possible.",
    }];
pub const PCHIP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "pp = pchip(integer_X, Y) or Vq = pchip(integer_X, Y, Xq)",
        inputs: &PCHIP_INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed sample locations are independently gated before gather and cross once into the floating PCHIP coefficient domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "pp = pchip(X, integer_Y) or Vq = pchip(X, integer_Y, Xq)",
        inputs: &PCHIP_INTEGER_Y_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed sample values are independently gated before gather and produce floating coefficients or interpolated values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = pchip(X, Y, integer_Xq)",
        inputs: &PCHIP_INTEGER_XQ_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed query points are independently gated and converted only after exactness is proved.",
    },
];

fn pchip_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn pchip_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    pchip_error_with_message(
        format!(
            "{}: {}",
            PCHIP_ERROR_INVALID_ARGUMENT.message,
            detail.as_ref()
        ),
        &PCHIP_ERROR_INVALID_ARGUMENT,
    )
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::pchip")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("pchip"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "PCHIP coefficient construction and evaluation currently run on the CPU reference path after gathering GPU inputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::interpolation::pchip")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "PCHIP builds a piecewise-polynomial representation and is not fused.",
};

fn pchip_type(args: &[Type], _ctx: &ResolveContext) -> Type {
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
    name = "pchip",
    category = "math/interpolation",
    summary = "Construct or evaluate shape-preserving piecewise cubic Hermite interpolants.",
    keywords = "pchip,shape preserving,cubic hermite,interpolation,ppval",
    accel = "sink",
    sink = true,
    type_resolver(pchip_type),
    descriptor(crate::builtins::math::interpolation::pchip::PCHIP_DESCRIPTOR),
    extensions(crate::builtins::math::interpolation::pchip::PCHIP_EXTENSIONS),
    integer_capabilities(crate::builtins::math::interpolation::pchip::PCHIP_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::interpolation::pchip"
)]
async fn pchip_builtin(x: Value, y: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_pchip_integer_boundary(&x, &PCHIP_INTEGER_SAMPLE_LOCATIONS_EXTENSION, "X").await?;
    ensure_pchip_integer_boundary(&y, &PCHIP_INTEGER_SAMPLE_VALUES_EXTENSION, "Y").await?;
    if let Some(query) = rest.first() {
        ensure_pchip_integer_boundary(query, &PCHIP_INTEGER_QUERY_EXTENSION, "Xq").await?;
    }
    let series = series_from_values(x, y, NAME).await.map_err(|err| {
        pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INVALID_INPUT)
    })?;
    let pp = build_pchip_pp(&series, NAME).map_err(|err| {
        pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INTERNAL)
    })?;
    match rest.len() {
        0 => pp_to_value(pp, NAME).map_err(|err| {
            pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INTERNAL)
        }),
        1 => {
            let query = query_points(rest.into_iter().next().expect("query"), NAME)
                .await
                .map_err(|err| {
                    pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INVALID_INPUT)
                })?;
            evaluate_pp(&pp, &query, &Extrapolation::Extrapolate, NAME).map_err(|err| {
                pchip_error_with_message(err.message().to_string(), &PCHIP_ERROR_INTERNAL)
            })
        }
        _ => Err(pchip_invalid_argument("too many input arguments")),
    }
}

async fn ensure_pchip_integer_boundary(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
    role: &str,
) -> crate::BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64_async, value_has_native_integer_class,
    };
    if !value_has_native_integer_class(value) {
        return Ok(());
    }
    crate::compatibility::ensure_builtin_extension_enabled(extension, NAME)?;
    if !native_integer_value_is_exact_f64_async(value).await? {
        return Err(pchip_error_with_message(
            format!("pchip: integer {role} values must be exactly representable as double"),
            &PCHIP_ERROR_INVALID_INPUT,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    #[test]
    fn pchip_three_arg_evaluates_monotone_data() {
        let value = block_on(pchip_builtin(
            row(&[1.0, 2.0, 3.0, 4.0]),
            row(&[0.0, 1.0, 1.5, 1.75]),
            vec![row(&[1.5, 2.5, 3.5])],
        ))
        .expect("pchip");
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert!(tensor
            .materialize_f64()
            .windows(2)
            .all(|pair| pair[1] >= pair[0]));
    }

    #[test]
    fn pchip_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = PCHIP_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"pp = pchip(X, Y)"));
        assert!(labels.contains(&"Vq = pchip(X, Y, Xq)"));
    }

    #[test]
    fn pchip_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = PCHIP_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.PCHIP.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.PCHIP.INVALID_INPUT"));
        assert!(codes.contains(&"RM.PCHIP.INTERNAL"));
    }

    #[test]
    fn pchip_too_many_args_uses_stable_identifier() {
        let err = block_on(pchip_builtin(
            row(&[1.0, 2.0]),
            row(&[1.0, 4.0]),
            vec![Value::Num(1.5), Value::Num(2.5)],
        ))
        .expect_err("expected pchip argument error");
        assert_eq!(err.identifier(), PCHIP_ERROR_INVALID_ARGUMENT.identifier);
    }
}
