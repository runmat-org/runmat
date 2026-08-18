//! Additional distribution-specific random-number generators.

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, NumericScalar, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

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

const INPUT_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of Bernoulli trials.",
};

const INPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Success probability for each trial.",
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
const INPUTS_NP: [BuiltinParamDescriptor; 2] = [INPUT_N, INPUT_P];
const INPUTS_NP_SZ: [BuiltinParamDescriptor; 3] = [INPUT_N, INPUT_P, INPUT_SZ];

const GAMRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = gamrnd(a, b)",
        inputs: &INPUTS_AB,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = gamrnd(a, b, sz)",
        inputs: &INPUTS_AB_SZ,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = gamrnd(a, b, sz1, sz2, ...)",
        inputs: &INPUTS_AB_SZ,
        outputs: &OUTPUT_R,
    },
];

const BINORND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = binornd(n, p)",
        inputs: &INPUTS_NP,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = binornd(n, p, sz)",
        inputs: &INPUTS_NP_SZ,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = binornd(n, p, sz1, sz2, ...)",
        inputs: &INPUTS_NP_SZ,
        outputs: &OUTPUT_R,
    },
];

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

fn validate_gamma(name: &'static str, args: &RandomArgs) -> BuiltinResult<()> {
    for value in &args.first {
        if value.is_nan() || *value < 0.0 {
            return Err(random_error(
                name,
                format!("{name}: shape parameter must be nonnegative"),
            ));
        }
    }
    for value in &args.second {
        if value.is_nan() || *value <= 0.0 {
            return Err(random_error(
                name,
                format!("{name}: scale parameter must be positive"),
            ));
        }
    }
    Ok(())
}

fn validate_binomial(name: &'static str, args: &RandomArgs) -> BuiltinResult<()> {
    for value in &args.first {
        if !value.is_finite() || *value <= 0.0 || value.fract() != 0.0 {
            return Err(random_error(
                name,
                format!("{name}: number of trials must be a positive integer"),
            ));
        }
    }
    for value in &args.second {
        if value.is_nan() || !(0.0..=1.0).contains(value) {
            return Err(random_error(
                name,
                format!("{name}: probability must be between 0 and 1"),
            ));
        }
    }
    Ok(())
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

pub mod gamrnd {
    use super::*;
    random_descriptor!("gamrnd", GAMRND_SIGNATURES);

    const INTEGER_SHAPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "gamrnd-integer-shape-parameter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gamrnd with a typed-integer shape parameter is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GamrndIntegerShapeParameterExtension"),
    };

    const INTEGER_SCALE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "gamrnd-integer-scale-parameter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gamrnd with a typed-integer scale parameter is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GamrndIntegerScaleParameterExtension"),
    };

    const INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "gamrnd-integer-size",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "gamrnd with typed-integer size arguments is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GamrndIntegerSizeExtension"),
    };

    pub const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
        INTEGER_SHAPE_EXTENSION,
        INTEGER_SCALE_EXTENSION,
        INTEGER_SIZE_EXTENSION,
    ];

    const INTEGER_SHAPE_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "a",
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Typed-integer shape parameters are gated before gather and must be exactly representable at the binary64 sampling boundary.",
        }];

    const INTEGER_SCALE_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "b",
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Typed-integer scale parameters are gated before gather and must be exactly representable at the binary64 sampling boundary.",
        }];

    const INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "sz",
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
            notes: "Typed-integer size values are decoded exactly from authoritative storage into bounded structural dimensions.",
        }];

    pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
        BuiltinIntegerCapabilityDescriptor {
            form: "r = gamrnd(integer_a, b, ___)",
            inputs: &INTEGER_SHAPE_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::GatherFallback,
            overload: BuiltinIntegerOverloadKind::Multiple,
            notes: "RunMat-only integer shape parameters cross a checked binary64 boundary. Current public documentation does not resolve gamrnd single/double output-class propagation, so this capability does not claim one.",
        },
        BuiltinIntegerCapabilityDescriptor {
            form: "r = gamrnd(a, integer_b, ___)",
            inputs: &INTEGER_SCALE_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::GatherFallback,
            overload: BuiltinIntegerOverloadKind::Multiple,
            notes: "RunMat-only integer scale parameters cross a checked binary64 boundary. Documented floating gpuArray inputs remain ungated and restore output residency.",
        },
        BuiltinIntegerCapabilityDescriptor {
            form: "r = gamrnd(a, b, integer_sz)",
            inputs: &INTEGER_SIZE_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::Structural,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::GatherFallback,
            overload: BuiltinIntegerOverloadKind::StructuralParameter,
            notes: "RunMat-only integer sizes are exact structural controls; they do not select output class or GPU execution residency.",
        },
    ];

    #[runtime_builtin(
        name = "gamrnd",
        category = "stats/random",
        summary = "Generate gamma-distributed random samples.",
        keywords = "gamrnd,gamma,random,distribution,statistics",
        type_resolver(super::random_type),
        descriptor(self::DESCRIPTOR),
        extensions(self::EXTENSIONS),
        integer_capabilities(self::INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::random::distribution_random::gamrnd"
    )]
    pub(crate) async fn gamrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let args = parse_args(args).await?;
        validate_gamma("gamrnd", &args.random)?;
        let len = checked_element_count(&args.random.shape)?;
        let data = random::generate_gamma_shape_scale(
            &args.random.first,
            &args.random.second,
            len,
            "gamrnd",
        )
        .map_err(|err| random_internal("gamrnd", err.message().to_string()))?;
        build_output(data, args.random.shape, args.gpu_source)
    }

    struct GamrndArgs {
        random: RandomArgs,
        gpu_source: Option<GpuTensorHandle>,
    }

    async fn parse_args(args: Vec<Value>) -> BuiltinResult<GamrndArgs> {
        if args.len() < 2 {
            return Err(random_error("gamrnd", "gamrnd: expected a and b"));
        }
        ensure_extensions(&args)?;
        let gpu_source = gpu_helpers::select_resident_output_source(
            args.iter().take(2).filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            "gamrnd",
        )?;
        let first = value_to_tensor("gamrnd", &args[0]).await?;
        let second = value_to_tensor("gamrnd", &args[1]).await?;
        ensure_exact_integer_boundary(&first, "shape parameter")?;
        ensure_exact_integer_boundary(&second, "scale parameter")?;
        let (first_data, second_data, parameter_shape) =
            tensor::binary_numeric_tensors(&first, &second, "gamrnd", "gamrnd")
                .map_err(|err| random_error("gamrnd", err.message().to_string()))?;
        let shape = if args.len() > 2 {
            parse_shape_args(&args[2..]).await?
        } else {
            normalize_shape(parameter_shape.clone())
        };
        if first_data.len() != 1 && normalize_shape(parameter_shape) != shape {
            return Err(random_error(
                "gamrnd",
                "gamrnd: requested size must match nonscalar parameters",
            ));
        }
        Ok(GamrndArgs {
            random: RandomArgs {
                first: first_data,
                second: second_data,
                shape,
            },
            gpu_source,
        })
    }

    fn ensure_extensions(args: &[Value]) -> BuiltinResult<()> {
        for (value, extension) in args
            .iter()
            .take(2)
            .zip([&INTEGER_SHAPE_EXTENSION, &INTEGER_SCALE_EXTENSION])
        {
            if is_typed_integer_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(extension, "gamrnd")?;
            }
        }
        if args.iter().skip(2).any(is_typed_integer_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_SIZE_EXTENSION,
                "gamrnd",
            )?;
        }
        Ok(())
    }

    fn is_typed_integer_value(value: &Value) -> bool {
        matches!(value, Value::Int(_))
            || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
            || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
    }

    pub(super) fn ensure_exact_integer_boundary(tensor: &Tensor, role: &str) -> BuiltinResult<()> {
        let Some(storage) = tensor.integer_storage() else {
            return Ok(());
        };
        if storage
            .exact_values()
            .iter()
            .any(|integer| !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(integer))
        {
            return Err(random_error(
                "gamrnd",
                format!("gamrnd: integer {role} values must be exactly representable as double"),
            ));
        }
        Ok(())
    }

    async fn parse_shape_args(rest: &[Value]) -> BuiltinResult<Vec<usize>> {
        let mut dims = Vec::new();
        for value in rest {
            let parsed = parse_shape_value(value).await?;
            if rest.len() > 1 && parsed.len() != 1 {
                return Err(random_error(
                    "gamrnd",
                    "gamrnd: separate size arguments must be scalars",
                ));
            }
            dims.extend(parsed);
        }
        Ok(normalize_dims(dims))
    }

    async fn parse_shape_value(value: &Value) -> BuiltinResult<Vec<usize>> {
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(|err| random_error("gamrnd", format!("gamrnd: {err}")))?;
        let tensor = tensor::value_into_tensor_for("gamrnd", gathered)
            .map_err(|err| random_error("gamrnd", format!("gamrnd: {err}")))?;
        if tensor.len() > 1 && !(tensor.shape.len() == 1 || tensor.shape.first() == Some(&1)) {
            return Err(random_error(
                "gamrnd",
                "gamrnd: size vector must be a row vector",
            ));
        }
        (0..tensor.len())
            .map(|index| {
                parse_size_scalar(
                    tensor
                        .numeric_value_at(index)
                        .expect("size tensor index must exist"),
                )
            })
            .collect()
    }

    fn parse_size_scalar(value: NumericScalar) -> BuiltinResult<usize> {
        let dimension = match value {
            NumericScalar::I8(value) => signed_size(i128::from(value)),
            NumericScalar::I16(value) => signed_size(i128::from(value)),
            NumericScalar::I32(value) => signed_size(i128::from(value)),
            NumericScalar::I64(value) => signed_size(i128::from(value)),
            NumericScalar::U8(value) => unsigned_size(u128::from(value)),
            NumericScalar::U16(value) => unsigned_size(u128::from(value)),
            NumericScalar::U32(value) => unsigned_size(u128::from(value)),
            NumericScalar::U64(value) => unsigned_size(u128::from(value)),
            NumericScalar::F32(value) => floating_size(f64::from(value)),
            NumericScalar::F64(value) => floating_size(value),
        };
        dimension.ok_or_else(|| {
            random_error(
                "gamrnd",
                "gamrnd: size values must be finite integers in the supported dimension range",
            )
        })
    }

    fn signed_size(value: i128) -> Option<usize> {
        if value <= 0 {
            Some(0)
        } else {
            usize::try_from(value).ok()
        }
    }

    fn unsigned_size(value: u128) -> Option<usize> {
        usize::try_from(value).ok()
    }

    fn floating_size(value: f64) -> Option<usize> {
        if !value.is_finite() || value.fract() != 0.0 {
            return None;
        }
        if value <= 0.0 {
            return Some(0);
        }
        if value >= usize::MAX as f64 {
            return None;
        }
        Some(value as usize)
    }

    fn checked_element_count(shape: &[usize]) -> BuiltinResult<usize> {
        shape.iter().try_fold(1usize, |count, dimension| {
            count.checked_mul(*dimension).ok_or_else(|| {
                random_error(
                    "gamrnd",
                    "gamrnd: requested size exceeds the supported array bounds",
                )
            })
        })
    }

    fn build_output(
        data: Vec<f64>,
        shape: Vec<usize>,
        gpu_source: Option<GpuTensorHandle>,
    ) -> BuiltinResult<Value> {
        let tensor = Tensor::new(data, shape)
            .map_err(|err| random_internal("gamrnd", format!("gamrnd: {err}")))?;
        let Some(source) = gpu_source else {
            return Ok(tensor::tensor_into_value(tensor));
        };
        let restored = gpu_helpers::restore_class_preserving_value(
            &source,
            tensor::tensor_into_value(tensor),
            "gamrnd",
        )?;
        if runmat_accelerate_api::handle_is_explicit(&source)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(random_internal(
                "gamrnd",
                "gamrnd: provider cannot preserve explicit gpuArray output residency",
            ));
        }
        Ok(restored)
    }
}

pub mod binornd {
    use super::*;
    random_descriptor!("binornd", BINORND_SIGNATURES);

    const INTEGER_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "binornd-integer-trials",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "binornd with typed-integer trial counts is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BinorndIntegerTrialsExtension"),
    };

    const INTEGER_P_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "binornd-integer-probability",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "binornd with typed-integer probabilities is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BinorndIntegerProbabilityExtension"),
    };

    const INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "binornd-integer-size",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "binornd with typed-integer size arguments is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BinorndIntegerSizeExtension"),
    };

    const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
        id: "binornd-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "binornd with logical numeric inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BinorndLogicalInputExtension"),
    };

    pub const EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
        INTEGER_N_EXTENSION,
        INTEGER_P_EXTENSION,
        INTEGER_SIZE_EXTENSION,
        LOGICAL_INPUT_EXTENSION,
    ];

    const INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "n",
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Typed-integer trial counts are gated by binornd-integer-trials and must be exactly representable at the floating binomial-sampling boundary.",
        }];

    const INTEGER_P_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "p",
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Typed-integer probabilities are gated by binornd-integer-probability and remain authoritative until the floating binomial-sampling boundary.",
        }];

    const INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
        [BuiltinIntegerInputCapability {
            name: "sz",
            classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
            availability: BuiltinIntegerInputAvailability::RunMatOnly,
            scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
            notes: "Typed-integer size arguments are gated by binornd-integer-size and parsed exactly into structural dimensions.",
        }];

    pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
        BuiltinIntegerCapabilityDescriptor {
            form: "r = binornd(n, p, sz1, sz2, ...) with integer n",
            inputs: &INTEGER_N_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::GatherFallback,
            overload: BuiltinIntegerOverloadKind::Multiple,
            notes: "RunMat-only typed-integer trial counts produce double samples unless a documented single probability makes the result single; resident fallback restores the result to the first resident input's owning provider.",
        },
        BuiltinIntegerCapabilityDescriptor {
            form: "r = binornd(n, p, sz1, sz2, ...) with integer p",
            inputs: &INTEGER_P_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::GatherFallback,
            overload: BuiltinIntegerOverloadKind::Multiple,
            notes: "RunMat-only typed-integer probabilities produce double samples unless a documented single trial-count input makes the result single.",
        },
        BuiltinIntegerCapabilityDescriptor {
            form: "r = binornd(n, p, sz1, sz2, ...) with integer size",
            inputs: &INTEGER_SIZE_INPUT,
            computation_domain: BuiltinIntegerComputationDomain::Structural,
            output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::GatherFallback,
            overload: BuiltinIntegerOverloadKind::Multiple,
            notes: "RunMat-only typed-integer sizes are exact structural controls and do not select output precision; nonpositive dimensions produce a documented empty result.",
        },
    ];

    #[runtime_builtin(
        name = "binornd",
        category = "stats/random",
        summary = "Generate binomially-distributed random samples.",
        keywords = "binornd,binomial,random,distribution,statistics",
        type_resolver(super::random_type),
        descriptor(self::DESCRIPTOR),
        extensions(self::EXTENSIONS),
        integer_capabilities(self::INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::random::distribution_random::binornd"
    )]
    pub(crate) async fn binornd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let args = parse_args(args).await?;
        validate_binomial("binornd", &args.random)?;
        let len = tensor::element_count(&args.shape);
        let data =
            random::generate_binomial(&args.random.first, &args.random.second, len, "binornd")
                .map_err(|err| random_internal("binornd", err.message().to_string()))?;
        build_output(data, args.shape, args.output_precision, args.gpu_source)
    }

    #[derive(Clone, Copy)]
    enum OutputPrecision {
        Double,
        Single,
    }

    struct BinorndArgs {
        random: RandomArgs,
        shape: Vec<usize>,
        output_precision: OutputPrecision,
        gpu_source: Option<GpuTensorHandle>,
    }

    async fn parse_args(args: Vec<Value>) -> BuiltinResult<BinorndArgs> {
        if args.len() < 2 {
            return Err(random_error("binornd", "binornd: expected n and p"));
        }
        ensure_extensions(&args)?;
        let output_precision = if args[..2].iter().any(is_single_value) {
            OutputPrecision::Single
        } else {
            OutputPrecision::Double
        };
        let gpu_source = args.iter().find_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        });
        let first = value_to_tensor("binornd", &args[0]).await?;
        let second = value_to_tensor("binornd", &args[1]).await?;
        ensure_exact_integer_boundary(&first, "n")?;
        ensure_exact_integer_boundary(&second, "p")?;
        let (first_data, second_data, parameter_shape) =
            tensor::binary_numeric_tensors(&first, &second, "binornd", "binornd")
                .map_err(|err| random_error("binornd", err.message().to_string()))?;
        let shape = if args.len() > 2 {
            parse_shape_args(&args[2..]).await?
        } else {
            normalize_shape(parameter_shape.clone())
        };
        if first_data.len() != 1 && normalize_shape(parameter_shape) != shape {
            return Err(random_error(
                "binornd",
                "binornd: requested size must match nonscalar parameters",
            ));
        }
        Ok(BinorndArgs {
            random: RandomArgs {
                first: first_data,
                second: second_data,
                shape: shape.clone(),
            },
            shape,
            output_precision,
            gpu_source,
        })
    }

    fn ensure_extensions(args: &[Value]) -> BuiltinResult<()> {
        for (value, extension) in args
            .iter()
            .take(2)
            .zip([&INTEGER_N_EXTENSION, &INTEGER_P_EXTENSION])
        {
            if is_typed_integer_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(extension, "binornd")?;
            }
        }
        if args.iter().skip(2).any(is_typed_integer_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_SIZE_EXTENSION,
                "binornd",
            )?;
        }
        if args.iter().any(is_logical_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_INPUT_EXTENSION,
                "binornd",
            )?;
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

    fn ensure_exact_integer_boundary(tensor: &Tensor, name: &str) -> BuiltinResult<()> {
        if tensor.integer_storage().is_none() {
            return Ok(());
        }
        const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
        for index in 0..tensor.len() {
            let exact = match tensor.numeric_value_at(index) {
                Some(NumericScalar::I8(value)) => i128::from(value),
                Some(NumericScalar::I16(value)) => i128::from(value),
                Some(NumericScalar::I32(value)) => i128::from(value),
                Some(NumericScalar::I64(value)) => i128::from(value),
                Some(NumericScalar::U8(value)) => i128::from(value),
                Some(NumericScalar::U16(value)) => i128::from(value),
                Some(NumericScalar::U32(value)) => i128::from(value),
                Some(NumericScalar::U64(value)) => i128::from(value),
                _ => continue,
            };
            if !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
                return Err(random_error(
                    "binornd",
                    format!(
                        "binornd: integer {name} values must be exactly representable as double"
                    ),
                ));
            }
        }
        Ok(())
    }

    async fn parse_shape_args(rest: &[Value]) -> BuiltinResult<Vec<usize>> {
        let mut dims = Vec::new();
        for value in rest {
            let parsed = parse_shape_value(value).await?;
            if rest.len() > 1 && parsed.len() != 1 {
                return Err(random_error(
                    "binornd",
                    "binornd: separate size arguments must be scalars",
                ));
            }
            dims.extend(parsed);
        }
        Ok(normalize_dims(dims))
    }

    async fn parse_shape_value(value: &Value) -> BuiltinResult<Vec<usize>> {
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(|err| random_error("binornd", format!("binornd: {err}")))?;
        let tensor = tensor::value_into_tensor_for("binornd", gathered)
            .map_err(|err| random_error("binornd", format!("binornd: {err}")))?;
        if tensor.len() > 1 && !(tensor.shape.len() == 1 || tensor.shape.first() == Some(&1)) {
            return Err(random_error(
                "binornd",
                "binornd: size vector must be a row vector",
            ));
        }
        (0..tensor.len())
            .map(|index| {
                parse_size_scalar(
                    tensor
                        .numeric_value_at(index)
                        .expect("size tensor index must exist"),
                )
            })
            .collect()
    }

    fn parse_size_scalar(value: NumericScalar) -> BuiltinResult<usize> {
        let dimension = match value {
            NumericScalar::I8(value) => signed_size(i128::from(value)),
            NumericScalar::I16(value) => signed_size(i128::from(value)),
            NumericScalar::I32(value) => signed_size(i128::from(value)),
            NumericScalar::I64(value) => signed_size(i128::from(value)),
            NumericScalar::U8(value) => unsigned_size(u128::from(value)),
            NumericScalar::U16(value) => unsigned_size(u128::from(value)),
            NumericScalar::U32(value) => unsigned_size(u128::from(value)),
            NumericScalar::U64(value) => unsigned_size(u128::from(value)),
            NumericScalar::F32(value) => floating_size(f64::from(value)),
            NumericScalar::F64(value) => floating_size(value),
        };
        dimension.ok_or_else(|| {
            random_error(
                "binornd",
                "binornd: size values must be finite integers in the supported dimension range",
            )
        })
    }

    fn signed_size(value: i128) -> Option<usize> {
        if value <= 0 {
            Some(0)
        } else {
            usize::try_from(value).ok()
        }
    }

    fn unsigned_size(value: u128) -> Option<usize> {
        usize::try_from(value).ok()
    }

    fn floating_size(value: f64) -> Option<usize> {
        if !value.is_finite() || value.fract() != 0.0 {
            return None;
        }
        if value <= 0.0 {
            return Some(0);
        }
        if value >= usize::MAX as f64 {
            return None;
        }
        Some(value as usize)
    }

    fn build_output(
        data: Vec<f64>,
        shape: Vec<usize>,
        precision: OutputPrecision,
        gpu_source: Option<GpuTensorHandle>,
    ) -> BuiltinResult<Value> {
        let tensor = match precision {
            OutputPrecision::Double => Tensor::new(data, shape),
            OutputPrecision::Single => {
                Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
            }
        }
        .map_err(|err| random_internal("binornd", format!("binornd: {err}")))?;
        if let Some(source) = gpu_source {
            let provider = runmat_accelerate_api::provider_for_handle(&source)
                .or_else(runmat_accelerate_api::provider)
                .ok_or_else(|| {
                    random_internal(
                        "binornd",
                        "binornd: no acceleration provider registered for GPU output",
                    )
                })?;
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|err| random_internal("binornd", format!("binornd: {err}")))?;
            return Ok(gpu_helpers::resident_gpu_value(handle));
        }
        match precision {
            OutputPrecision::Double => Ok(tensor::tensor_into_value(tensor)),
            OutputPrecision::Single => Ok(Value::Tensor(tensor)),
        }
    }
}

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
                storage.exact_values().iter().any(|value| {
                    !crate::builtins::math::trigonometry::cos::integer_is_exact_f64(value)
                })
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
    use runmat_builtins::IntegerStorage;

    fn reset() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    fn all_integer_storages(value: u64) -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![value as i8]),
            IntegerStorage::I16(vec![value as i16]),
            IntegerStorage::I32(vec![value as i32]),
            IntegerStorage::I64(vec![value as i64]),
            IntegerStorage::U8(vec![value as u8]),
            IntegerStorage::U16(vec![value as u16]),
            IntegerStorage::U32(vec![value as u32]),
            IntegerStorage::U64(vec![value]),
        ]
    }

    #[test]
    fn gamrnd_accepts_broadcast_and_size_forms() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let out = block_on(gamrnd::gamrnd_builtin(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
            Value::Num(2.0),
        ]))
        .expect("gamrnd");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert!(tensor.materialize_f64().iter().all(|value| *value >= 0.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let out = block_on(gamrnd::gamrnd_builtin(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(2.0),
            Value::Num(4.0),
        ]))
        .expect("gamrnd size");
        match out {
            Value::Tensor(tensor) => assert_eq!(tensor.shape, vec![2, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gamrnd_typed_integer_roles_are_independently_gated() {
        let _guard = random::test_lock().lock().unwrap();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        reset();
        let cases = [
            (
                vec![
                    integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                    Value::Num(3.0),
                ],
                "RunMat:compatibility:GamrndIntegerShapeParameterExtension",
            ),
            (
                vec![
                    Value::Num(2.0),
                    integer_tensor(IntegerStorage::U16(vec![3]), vec![1, 1]),
                ],
                "RunMat:compatibility:GamrndIntegerScaleParameterExtension",
            ),
            (
                vec![
                    Value::Num(2.0),
                    Value::Num(3.0),
                    integer_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2]),
                ],
                "RunMat:compatibility:GamrndIntegerSizeExtension",
            ),
        ];
        for (args, identifier) in cases {
            let error = block_on(gamrnd::gamrnd_builtin(args)).unwrap_err();
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn gamrnd_integer_sizes_are_exact_and_floating_boundaries_reject_loss() {
        let _guard = random::test_lock().lock().unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset();
        let output = block_on(gamrnd::gamrnd_builtin(vec![
            integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
            integer_tensor(IntegerStorage::U16(vec![3]), vec![1, 1]),
            integer_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]),
        ]))
        .expect("checked integer gamrnd");
        let Value::Tensor(output) = output else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![2, 3]);

        let error = block_on(gamrnd::gamrnd_builtin(vec![
            integer_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
            Value::Num(1.0),
        ]))
        .unwrap_err();
        assert!(error.message().contains("exactly representable as double"));

        let exact_wide = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54]), vec![1, 1])
            .expect("integer tensor");
        gamrnd::ensure_exact_integer_boundary(&exact_wide, "test")
            .expect("exact powers of two above 2^53 remain valid");

        let oversized = block_on(gamrnd::gamrnd_builtin(vec![
            Value::Num(2.0),
            Value::Num(1.0),
            integer_tensor(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]),
            Value::Num(2.0),
        ]))
        .unwrap_err();
        assert!(oversized.message().contains("supported array bounds"));
    }

    #[test]
    fn gamrnd_documented_floating_gpu_input_is_ungated_and_restores_residency() {
        use crate::builtins::common::test_support;

        let _guard = random::test_lock().lock().unwrap();
        reset();
        test_support::with_test_provider(|provider| {
            let parameter = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let output = block_on(gamrnd::gamrnd_builtin(vec![
                Value::GpuTensor(handle),
                Value::Num(1.0),
            ]))
            .expect("documented floating gpuArray form");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(ProviderPrecision::F64)
            );
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gamrnd_wgpu_fallback_enforces_explicit_residency_contract() {
        use crate::builtins::common::test_support;

        let _accel_guard = test_support::accel_test_lock();
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("actual WGPU provider");
        let parameter = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let result = block_on(gamrnd::gamrnd_builtin(vec![
            Value::GpuTensor(handle.clone()),
            Value::Num(1.0),
        ]));
        match runmat_accelerate_api::AccelProvider::precision(provider) {
            ProviderPrecision::F64 => {
                let Value::GpuTensor(output) = result.expect("documented WGPU gamrnd") else {
                    panic!("expected resident output");
                };
                assert!(runmat_accelerate_api::handle_is_explicit(&output));
                assert_eq!(
                    output.device_id,
                    runmat_accelerate_api::AccelProvider::device_id(provider)
                );
                assert_eq!(output.shape, vec![1, 2]);
                assert_eq!(
                    runmat_accelerate_api::handle_precision(&output),
                    Some(ProviderPrecision::F64)
                );
            }
            ProviderPrecision::F32 => {
                let error = result.expect_err("double output cannot use an f32-only provider");
                assert!(error
                    .message()
                    .contains("cannot preserve explicit gpuArray output residency"));
                assert!(runmat_accelerate_api::handle_is_explicit(&handle));
            }
        }
    }

    #[test]
    fn binornd_accepts_array_parameters_and_rejects_fractional_trials() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let out = block_on(binornd::binornd_builtin(vec![
            Value::Tensor(Tensor::new(vec![5.0, 10.0], vec![1, 2]).unwrap()),
            Value::Num(0.5),
        ]))
        .expect("binornd");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert!(tensor.materialize_f64()[0] >= 0.0 && tensor.materialize_f64()[0] <= 5.0);
                assert!(tensor.materialize_f64()[1] >= 0.0 && tensor.materialize_f64()[1] <= 10.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let err = block_on(binornd::binornd_builtin(vec![
            Value::Num(2.5),
            Value::Num(0.5),
        ]))
        .expect_err("fractional trials should fail");
        assert_eq!(err.identifier(), Some("RunMat:binornd:InvalidArgument"));
    }

    #[test]
    fn binornd_large_trial_counts_use_bounded_sampler() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let out = block_on(binornd::binornd_builtin(vec![
            Value::Num(1.0e12),
            Value::Num(0.5),
        ]))
        .expect("large binornd");
        match out {
            Value::Num(value) => {
                assert!(value.is_finite());
                assert!((0.0..=1.0e12).contains(&value));
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn binornd_classifies_every_integer_input_position_and_preserves_exact_sizes() {
        let _guard = random::test_lock().lock().unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset();
        for storage in all_integer_storages(1) {
            let n = block_on(binornd::binornd_builtin(vec![
                integer_tensor(storage.clone(), vec![1, 1]),
                Value::Num(1.0),
            ]))
            .expect("typed-integer n");
            assert_eq!(n, Value::Num(1.0));

            let p = block_on(binornd::binornd_builtin(vec![
                Value::Num(1.0),
                integer_tensor(storage.clone(), vec![1, 1]),
            ]))
            .expect("typed-integer p");
            assert_eq!(p, Value::Num(1.0));

            let size = block_on(binornd::binornd_builtin(vec![
                Value::Num(1.0),
                Value::Num(1.0),
                integer_tensor(storage, vec![1, 1]),
            ]))
            .expect("typed-integer size");
            let Value::Num(value) = size else {
                panic!("one-by-one output should collapse to a scalar");
            };
            assert_eq!(value, 1.0);
        }

        let wide_size = integer_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
        let sized = block_on(binornd::binornd_builtin(vec![
            Value::Num(1.0),
            Value::Num(1.0),
            wide_size,
        ]))
        .expect("exact typed size vector");
        let Value::Tensor(sized) = sized else {
            panic!("expected sized tensor");
        };
        assert_eq!(sized.shape, vec![2, 3]);
    }

    #[test]
    fn binornd_compatibility_guards_run_before_resident_access() {
        let _guard = random::test_lock().lock().unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        reset();
        let cases = [
            (
                vec![
                    integer_tensor(IntegerStorage::I8(vec![1]), vec![1, 1]),
                    Value::Num(0.5),
                ],
                "RunMat:compatibility:BinorndIntegerTrialsExtension",
            ),
            (
                vec![
                    Value::Num(1.0),
                    integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
                ],
                "RunMat:compatibility:BinorndIntegerProbabilityExtension",
            ),
            (
                vec![
                    Value::Num(1.0),
                    Value::Num(0.5),
                    integer_tensor(IntegerStorage::U32(vec![2]), vec![1, 1]),
                ],
                "RunMat:compatibility:BinorndIntegerSizeExtension",
            ),
            (
                vec![Value::Bool(true), Value::Num(0.5)],
                "RunMat:compatibility:BinorndLogicalInputExtension",
            ),
        ];
        for (args, identifier) in cases {
            let error = block_on(binornd::binornd_builtin(args)).unwrap_err();
            assert_eq!(error.identifier(), Some(identifier));
        }

        let resident = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_306_001,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            runmat_accelerate_api::NumericElementType::I16,
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        let error = block_on(binornd::binornd_builtin(vec![
            Value::GpuTensor(resident.clone()),
            Value::Num(0.5),
        ]))
        .expect_err("resident integer guard must precede gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:BinorndIntegerTrialsExtension")
        );
    }

    #[test]
    fn binornd_preserves_single_and_rejects_inexact_integer_boundary() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let single = block_on(binornd::binornd_builtin(vec![
            Value::Tensor(Tensor::from_f32(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
        ]))
        .expect("single binornd");
        let Value::Tensor(single) = single else {
            panic!("expected native-single tensor");
        };
        assert_eq!(single.numeric_dtype(), NumericDType::F32);
        assert_eq!(single.materialize_f64(), vec![1.0, 1.0]);

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = block_on(binornd::binornd_builtin(vec![
            integer_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]),
            Value::Num(0.5),
        ]))
        .unwrap_err();
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn binornd_nonpositive_sizes_are_empty_and_trailing_ones_are_ignored() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let negative = block_on(binornd::binornd_builtin(vec![
            Value::Num(1.0),
            Value::Num(0.5),
            Value::Num(-2.0),
            Value::Num(3.0),
        ]))
        .expect("negative size");
        let Value::Tensor(negative) = negative else {
            panic!("expected empty tensor");
        };
        assert_eq!(negative.shape, vec![0, 3]);
        assert!(negative.is_empty());

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let typed_negative = block_on(binornd::binornd_builtin(vec![
            Value::Num(1.0),
            Value::Num(0.5),
            integer_tensor(IntegerStorage::I64(vec![-2]), vec![1, 1]),
            integer_tensor(IntegerStorage::U64(vec![3]), vec![1, 1]),
        ]))
        .expect("typed negative size");
        let Value::Tensor(typed_negative) = typed_negative else {
            panic!("expected typed empty tensor");
        };
        assert_eq!(typed_negative.shape, vec![0, 3]);
        assert!(typed_negative.is_empty());

        let trailing = block_on(binornd::binornd_builtin(vec![
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![3.0, 1.0, 1.0], vec![1, 3]).unwrap()),
        ]))
        .expect("trailing singleton dimensions");
        let Value::Tensor(trailing) = trailing else {
            panic!("expected tensor");
        };
        assert_eq!(trailing.shape, vec![3, 1]);
    }

    #[test]
    fn binornd_gpu_fallback_preserves_residency_and_precision() {
        use crate::builtins::common::test_support;

        let _guard = random::test_lock().lock().unwrap();
        reset();
        test_support::with_test_provider(|provider| {
            let parameter = Tensor::from_f32(vec![1.0, 1.0], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
            let output = block_on(binornd::binornd_builtin(vec![
                Value::GpuTensor(handle),
                Value::Num(1.0),
            ]))
            .expect("resident binornd");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(output_handle),
                Some(ProviderPrecision::F32)
            );
            let gathered = test_support::gather(output).expect("gather");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 1.0]);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn binornd_wgpu_fallback_preserves_residency_for_all_integer_classes() {
        use crate::builtins::common::test_support;

        let _accel_guard = test_support::accel_test_lock();
        let _guard = random::test_lock().lock().unwrap();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_integer_storages(1) {
            let parameter = Tensor::new_integer(storage, vec![1, 1]).expect("integer n");
            let handle = gpu_helpers::upload_tensor(provider, &parameter).expect("upload");
            let output = block_on(binornd::binornd_builtin(vec![
                Value::GpuTensor(handle),
                Value::Num(1.0),
            ]))
            .expect("resident integer binornd");
            assert!(matches!(output, Value::GpuTensor(_)));
            let gathered = test_support::gather(output).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![1.0]);
        }
    }

    #[test]
    fn nonscalar_parameters_must_match_explicit_size() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let err = block_on(gamrnd::gamrnd_builtin(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ]))
        .expect_err("mismatched explicit size should fail");
        assert_eq!(err.identifier(), Some("RunMat:gamrnd:InvalidArgument"));
    }

    #[test]
    fn wblrnd_accepts_size_and_positive_parameters() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
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
