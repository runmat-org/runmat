//! Additional distribution-specific random-number generators.

use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array.",
}];

const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First distribution parameter.",
};

const INPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second distribution parameter.",
};

const INPUT_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output size arguments.",
};

const INPUTS_AB: [BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_B];
const INPUTS_AB_SZ: [BuiltinParamDescriptor; 3] = [INPUT_A, INPUT_B, INPUT_SZ];
const WBLRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = wblrnd(a, b)",
        inputs: &INPUTS_AB,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = wblrnd(a, b, sz)",
        inputs: &INPUTS_AB_SZ,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = wblrnd(a, b, sz1, sz2, ...)",
        inputs: &INPUTS_AB_SZ,
        outputs: &OUTPUT_R,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISTRIBUTION_RANDOM.INVALID_ARGUMENT",
    identifier: None,
    when: "Input parameters or size arguments are missing, malformed, or incompatible.",
    message: "distribution random: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISTRIBUTION_RANDOM.INTERNAL",
    identifier: None,
    when: "Internal tensor conversion, allocation, or RNG state access fails.",
    message: "distribution random: internal error",
};

macro_rules! random_descriptor {
    ($name:literal, $signatures:expr) => {
        const ERRORS: [BuiltinErrorDescriptor; 2] = [
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
                when: ERROR_INVALID_ARGUMENT.when,
                message: ERROR_INVALID_ARGUMENT.message,
            },
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: ERROR_INTERNAL.when,
                message: ERROR_INTERNAL.message,
            },
        ];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

fn random_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() <= 2 {
        Type::Unknown
    } else {
        Type::Tensor { shape: None }
    }
}

fn random_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:InvalidArgument"))
        .build()
}

fn random_internal(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:Internal"))
        .build()
}

async fn value_to_tensor(name: &'static str, value: &Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(value)
        .await
        .map_err(|err| random_error(name, format!("{name}: {err}")))?;
    tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| random_error(name, format!("{name}: {err}")))
}

struct RandomArgs {
    first: Vec<f64>,
    second: Vec<f64>,
    shape: Vec<usize>,
}

async fn parse_two_parameter_args(
    name: &'static str,
    args: Vec<Value>,
) -> BuiltinResult<RandomArgs> {
    if args.len() < 2 {
        return Err(random_error(
            name,
            format!("{name}: expected two parameters"),
        ));
    }
    let first = value_to_tensor(name, &args[0]).await?;
    let second = value_to_tensor(name, &args[1]).await?;
    let (first_data, second_data, parameter_shape) =
        tensor::binary_numeric_tensors(&first, &second, name, name)
            .map_err(|err| random_error(name, err.message().to_string()))?;

    let explicit_shape = if args.len() > 2 {
        Some(parse_shape_args(name, &args[2..]).await?)
    } else {
        None
    };
    let shape = explicit_shape.unwrap_or_else(|| normalize_shape(parameter_shape.clone()));
    if tensor::element_count(&shape) == 0 {
        return Ok(RandomArgs {
            first: vec![0.0],
            second: vec![0.0],
            shape,
        });
    }
    if first_data.len() != 1 && normalize_shape(parameter_shape) != shape {
        return Err(random_error(
            name,
            format!("{name}: requested size must match nonscalar parameters"),
        ));
    }
    Ok(RandomArgs {
        first: first_data,
        second: second_data,
        shape,
    })
}

async fn parse_shape_args(name: &'static str, rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    let mut dims = Vec::new();
    for arg in rest {
        match extract_dims(arg, name).await {
            Ok(Some(values)) => dims.extend(values),
            Ok(None) => {
                return Err(random_error(
                    name,
                    format!("{name}: invalid size argument: {arg:?}"),
                ));
            }
            Err(err) => return Err(random_error(name, err)),
        }
    }
    Ok(normalize_dims(dims))
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape = vec![1, 1];
    } else if shape.len() == 1 {
        shape.push(1);
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        normalize_shape(dims)
    }
}

fn validate_weibull(name: &'static str, args: &RandomArgs) -> BuiltinResult<()> {
    for value in args.first.iter().chain(args.second.iter()) {
        if value.is_nan() || *value <= 0.0 {
            return Err(random_error(
                name,
                format!("{name}: scale and shape parameters must be positive"),
            ));
        }
    }
    Ok(())
}

#[path = "distribution_random/gamrnd.rs"]
pub mod gamrnd;

#[path = "distribution_random/binornd.rs"]
pub mod binornd;

pub mod wblrnd {
    use super::*;
    random_descriptor!("wblrnd", WBLRND_SIGNATURES);

    const INTEGER_SCALE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "wblrnd-integer-scale",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "wblrnd with a typed-integer scale parameter is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WblrndIntegerScaleExtension"),
    };

    const INTEGER_SHAPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "wblrnd-integer-shape",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "wblrnd with a typed-integer shape parameter is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WblrndIntegerShapeExtension"),
    };

    const INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "wblrnd-integer-size",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "wblrnd with typed-integer size arguments is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WblrndIntegerSizeExtension"),
    };

    const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "wblrnd-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "wblrnd with logical parameters or size arguments is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WblrndLogicalInputExtension"),
    };

    pub const EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
        INTEGER_SCALE_EXTENSION,
        INTEGER_SHAPE_EXTENSION,
        INTEGER_SIZE_EXTENSION,
        LOGICAL_INPUT_EXTENSION,
    ];

    const INTEGER_SCALE_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "a",
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Typed-integer scale values are gated before gather and must remain exact at the binary64 sampling boundary.",
        }];

    const INTEGER_SHAPE_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "b",
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Typed-integer shape values are gated before gather and must remain exact at the binary64 sampling boundary.",
        }];

    const INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "sz",
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: "Typed-integer size values are decoded from authoritative storage as bounded structural dimensions.",
        }];

    pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
        BuiltinIntegerCapabilityDescriptor {
            form: "r = wblrnd(integer_a, b, ___)",
            inputs: &INTEGER_SCALE_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::HostAndGpu,
            overload: BuiltinIntegerOverloadKind::Multiple,
            notes: "The public parameter classes are single and double. RunMat mode accepts every integer class after exact conversion validation; a documented single parameter still selects single output, and resident parameter inputs preserve provider ownership.",
        },
        BuiltinIntegerCapabilityDescriptor {
            form: "r = wblrnd(a, integer_b, ___)",
            inputs: &INTEGER_SHAPE_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::HostAndGpu,
            overload: BuiltinIntegerOverloadKind::Multiple,
            notes: "The RunMat-only integer shape parameter crosses one checked binary64 sampling boundary without changing the documented floating output-class rule.",
        },
        BuiltinIntegerCapabilityDescriptor {
            form: "r = wblrnd(a, b, integer_sz)",
            inputs: &INTEGER_SIZE_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::Structural,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::GatherFallback,
            overload: BuiltinIntegerOverloadKind::StructuralParameter,
            notes: "The public size arguments are integer-valued single or double values. RunMat-only typed-integer sizes are exact structural controls and do not select precision or output residency.",
        },
    ];

    #[runtime_builtin(
        name = "wblrnd",
        category = "stats/random",
        summary = "Generate Weibull-distributed random samples.",
        keywords = "wblrnd,weibull,random,distribution,statistics",
        type_resolver(super::random_type),
        descriptor(self::DESCRIPTOR),
        extensions(self::EXTENSIONS),
        integer_capabilities(self::INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::random::distribution_random::wblrnd"
    )]
    pub(crate) async fn wblrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        ensure_extensions(&args)?;
        ensure_exact_integer_values(&args).await?;
        let output_single = args.iter().take(2).any(is_single_value);
        let gpu_source = gpu_helpers::select_resident_output_source(
            args.iter().take(2).filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            "wblrnd",
        )?;
        let parsed = parse_two_parameter_args("wblrnd", args).await?;
        validate_weibull("wblrnd", &parsed)?;
        let len = tensor::element_count(&parsed.shape);
        let data = random::generate_weibull(&parsed.first, &parsed.second, len, "wblrnd")
            .map_err(|err| random_internal("wblrnd", err.message().to_string()))?;
        let host = if output_single {
            Value::Tensor(
                Tensor::from_f32(
                    data.into_iter().map(|value| value as f32).collect(),
                    parsed.shape,
                )
                .map_err(|err| random_internal("wblrnd", format!("wblrnd: {err}")))?,
            )
        } else {
            Tensor::new(data, parsed.shape)
                .map(tensor::tensor_into_value)
                .map_err(|err| random_internal("wblrnd", format!("wblrnd: {err}")))?
        };
        match gpu_source {
            Some(source) => gpu_helpers::restore_class_preserving_value(&source, host, "wblrnd"),
            None => Ok(host),
        }
    }

    fn ensure_extensions(args: &[Value]) -> BuiltinResult<()> {
        for (value, extension) in args
            .iter()
            .take(2)
            .zip([&INTEGER_SCALE_EXTENSION, &INTEGER_SHAPE_EXTENSION])
        {
            if is_typed_integer_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(extension, "wblrnd")?;
            }
        }
        if args.iter().skip(2).any(is_typed_integer_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_SIZE_EXTENSION,
                "wblrnd",
            )?;
        }
        if args.iter().any(is_logical_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_INPUT_EXTENSION,
                "wblrnd",
            )?;
        }
        Ok(())
    }

    async fn ensure_exact_integer_values(args: &[Value]) -> BuiltinResult<()> {
        for value in args
            .iter()
            .take(2)
            .filter(|value| is_typed_integer_value(value))
        {
            let tensor = value_to_tensor("wblrnd", value).await?;
            let inexact = tensor.integer_storage().is_some_and(|storage| {
                storage
                    .exact_values()
                    .iter()
                    .any(|value| !crate::builtins::common::validation::integer_is_exact_f64(value))
            });
            if inexact {
                return Err(random_error(
                    "wblrnd",
                    "wblrnd: integer parameters must be exactly representable as double",
                ));
            }
        }
        Ok(())
    }

    fn is_typed_integer_value(value: &Value) -> bool {
        matches!(value, Value::Int(_))
            || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
            || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
    }

    fn is_logical_value(value: &Value) -> bool {
        matches!(value, Value::Bool(_) | Value::LogicalArray(_))
            || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    }

    fn is_single_value(value: &Value) -> bool {
        matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
            || matches!(value, Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_none()
                    && !runmat_accelerate_api::handle_is_logical(handle)
                    && runmat_accelerate_api::handle_precision(handle) == Some(ProviderPrecision::F32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn reset() -> impl Drop {
        let guard = random::test_guard();
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
        guard
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    #[test]
    fn wblrnd_accepts_size_and_positive_parameters() {
        let _guard = random::test_guard();
        let _guard = reset();
        let out = block_on(wblrnd::wblrnd_builtin(vec![
            Value::Num(4.0),
            Value::Num(3.0),
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        ]))
        .expect("wblrnd");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 3]);
                assert!(tensor.materialize_f64().iter().all(|value| *value >= 0.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn wblrnd_typed_integer_roles_are_independently_gated() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let cases = [
            (
                vec![
                    integer_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]),
                    Value::Num(3.0),
                ],
                "RunMat:compatibility:WblrndIntegerScaleExtension",
            ),
            (
                vec![
                    Value::Num(4.0),
                    integer_tensor(IntegerStorage::U16(vec![3]), vec![1, 1]),
                ],
                "RunMat:compatibility:WblrndIntegerShapeExtension",
            ),
            (
                vec![
                    Value::Num(4.0),
                    Value::Num(3.0),
                    integer_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2]),
                ],
                "RunMat:compatibility:WblrndIntegerSizeExtension",
            ),
        ];
        for (args, identifier) in cases {
            let error = block_on(wblrnd::wblrnd_builtin(args)).expect_err("integer role gate");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn wblrnd_rejects_lossy_wide_parameter_and_preserves_single_output() {
        let _guard = random::test_lock().lock().unwrap();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        reset();
        let error = block_on(wblrnd::wblrnd_builtin(vec![
            integer_tensor(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]),
            Value::Num(3.0),
        ]))
        .expect_err("lossy parameter must reject");
        assert!(
            error.message().contains("exactly representable as double"),
            "unexpected error: {}",
            error.message()
        );

        let output = block_on(wblrnd::wblrnd_builtin(vec![
            Value::Tensor(Tensor::from_f32(vec![4.0], vec![1, 1]).unwrap()),
            Value::Num(3.0),
        ]))
        .expect("single wblrnd");
        let Value::Tensor(output) = output else {
            panic!("expected tensor output");
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }
}
