//! MATLAB-compatible `rng` builtin for seeding and querying RunMat's global random generator.

use crate::builtins::common::random::{
    self, set_default, set_seed, RngAlgorithm, RngSnapshot, DEFAULT_USER_SEED,
};
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;

use log::debug;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use runmat_time::unix_timestamp_ns;

use crate::builtins::stats::type_resolvers::rng_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rng";
const MATLAB_SEED_UPPER_BOUND: u64 = 1_u64 << 32;

const RNG_WIDE_SEED_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rng-wide-seed",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rng seeds at or above 2^32 are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RngWideSeedExtension"),
};

const RNG_TYPED_STATE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rng-typed-state-fields",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rng state structures with native integer fields are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RngTypedStateFieldsExtension"),
};

pub const RNG_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [RNG_WIDE_SEED_EXTENSION, RNG_TYPED_STATE_EXTENSION];

const RNG_INTEGER_SEED_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "seed",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented seed is a nonnegative integer below 2^32; native integer scalars are read exactly before the range check.",
    }];

const RNG_WIDE_INTEGER_SEED_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "seed",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode extends the seed domain through u64; native typed values remain exact even above flintmax.",
    }];

const RNG_INTEGER_STATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Seed/State fields",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "MATLAB-compatible restoration consumes a structure returned by rng; accepting independently constructed native-integer state fields is a separately gated RunMat extension.",
    }];

pub const RNG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "s = rng(integer_seed[, generator]) where seed < 2^32",
        inputs: &RNG_INTEGER_SEED_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact seed configures the shared host stream and provider state hook; generator sequence/state parity is a general RNG-engine conformance gap, not an implicit integer conversion boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "s = rng(wide_integer_seed[, generator])",
        inputs: &RNG_WIDE_INTEGER_SEED_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This RunMat-only form is gated before stream mutation and retains an exact typed Seed field when a binary64 field would round.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "s = rng(state_with_integer_fields)",
        inputs: &RNG_INTEGER_STATE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The independently gated extension validates native Seed and State words directly from authoritative storage without an f64 mirror.",
    },
];

const RNG_OUTPUT_S: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RNG state snapshot struct with fields Type, Seed, and State.",
}];

const RNG_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const RNG_INPUTS_SEED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "seed",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Non-negative integer seed.",
}];

const RNG_INPUTS_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Option token: 'default' or 'shuffle'.",
}];

const RNG_INPUTS_STATE_STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "state",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State struct containing Type, optional Seed, and State fields.",
}];

const RNG_INPUTS_SEED_GENERATOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "seed",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Non-negative integer seed.",
    },
    BuiltinParamDescriptor {
        name: "generator",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"twister\""),
        description: "Generator token (currently only 'twister'/'default'/'runmat-lcg').",
    },
];

const RNG_INPUTS_OPTION_GENERATOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option token: 'default' or 'shuffle'.",
    },
    BuiltinParamDescriptor {
        name: "generator",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"twister\""),
        description: "Generator token (currently only 'twister'/'default'/'runmat-lcg').",
    },
];

const RNG_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "s = rng()",
        inputs: &RNG_INPUTS_NONE,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(seed)",
        inputs: &RNG_INPUTS_SEED,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(option)",
        inputs: &RNG_INPUTS_OPTION,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(state)",
        inputs: &RNG_INPUTS_STATE_STRUCT,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(seed, generator)",
        inputs: &RNG_INPUTS_SEED_GENERATOR,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(option, generator)",
        inputs: &RNG_INPUTS_OPTION_GENERATOR,
        outputs: &RNG_OUTPUT_S,
    },
];

const RNG_ERROR_SEED_NONNEGATIVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.SEED_NONNEGATIVE",
    identifier: Some("RunMat:rng:SeedMustBeNonnegative"),
    when: "Seed value is negative.",
    message: "rng: seed must be non-negative",
};

const RNG_ERROR_GENERATOR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.GENERATOR_UNSUPPORTED",
    identifier: Some("RunMat:rng:GeneratorUnsupported"),
    when: "Generator token is unsupported.",
    message: "rng: generator is not supported",
};

const RNG_ERROR_STATE_TYPE_FIELD_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.STATE_TYPE_FIELD_MISSING",
    identifier: Some("RunMat:rng:StateTypeFieldMissing"),
    when: "State struct is missing the Type field.",
    message: "rng: state struct is missing the 'Type' field",
};

const RNG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.INVALID_ARGUMENT",
    identifier: Some("RunMat:rng:InvalidArgument"),
    when: "Arguments are missing, malformed, or incompatible with supported forms.",
    message: "rng: invalid argument",
};

const RNG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.INTERNAL",
    identifier: Some("RunMat:rng:Internal"),
    when: "Internal snapshot conversion/allocation/apply fails.",
    message: "rng: internal operation failed",
};

const RNG_ERRORS: [BuiltinErrorDescriptor; 5] = [
    RNG_ERROR_SEED_NONNEGATIVE,
    RNG_ERROR_GENERATOR_UNSUPPORTED,
    RNG_ERROR_STATE_TYPE_FIELD_MISSING,
    RNG_ERROR_INVALID_ARGUMENT,
    RNG_ERROR_INTERNAL,
];

pub const RNG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RNG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RNG_ERRORS,
};

fn rng_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn rng_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    rng_error_with(error, error.message)
}

fn rng_internal_error(message: impl Into<String>) -> RuntimeError {
    rng_error_with(&RNG_ERROR_INTERNAL, message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::random::rng")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rng",
    op_kind: GpuOpKind::Custom("state-control"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("set_rng_state")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Not a numeric kernel; synchronises provider RNG state via set_rng_state when available.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::random::rng")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rng",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Control builtin; fusion planner never embeds rng in generated kernels.",
};

#[runtime_builtin(
    name = "rng",
    category = "stats/random",
    summary = "Configure, query, and restore the global pseudorandom number generator state.",
    keywords = "rng,seed,twister,shuffle,state",
    type_resolver(rng_type),
    descriptor(crate::builtins::stats::random::rng::RNG_DESCRIPTOR),
    extensions(crate::builtins::stats::random::rng::RNG_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::random::rng::RNG_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::random::rng"
)]
async fn rng_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        let current = random::snapshot()?;
        return snapshot_to_value(current);
    }

    ensure_rng_state_extensions(&args)?;

    let previous = random::snapshot()?;
    let command = parse_command(&args)?;
    apply_command(command)?;
    let current = random::snapshot()?;
    sync_provider_state(current.state);
    snapshot_to_value(previous)
}

#[derive(Debug, Clone)]
enum ParsedCommand {
    Default,
    Seed(u64),
    Shuffle,
    Restore(RngSnapshot),
}

fn parse_command(args: &[Value]) -> BuiltinResult<ParsedCommand> {
    match args.len() {
        1 => parse_single_arg(&args[0]),
        2 => parse_double_args(&args[0], &args[1]),
        _ => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: invalid number of arguments",
        )),
    }
}

fn parse_single_arg(arg: &Value) -> BuiltinResult<ParsedCommand> {
    if let Some(keyword) = keyword_of(arg) {
        return parse_keyword(&keyword, None);
    }
    match arg {
        Value::Struct(_) => Ok(ParsedCommand::Restore(snapshot_from_value(arg)?)),
        _ => {
            let seed = parse_seed_scalar(arg, "rng: seed")?;
            ensure_direct_seed_range(seed)?;
            Ok(ParsedCommand::Seed(seed))
        }
    }
}

fn parse_double_args(first: &Value, second: &Value) -> BuiltinResult<ParsedCommand> {
    if let Some(keyword) = keyword_of(first) {
        let generator = Some(parse_generator(second)?);
        return parse_keyword(&keyword, generator);
    }
    let seed = parse_seed_scalar(first, "rng: seed")?;
    ensure_direct_seed_range(seed)?;
    let _ = parse_generator(second)?;
    Ok(ParsedCommand::Seed(seed))
}

fn parse_keyword(keyword: &str, generator: Option<RngAlgorithm>) -> BuiltinResult<ParsedCommand> {
    let algo = generator.unwrap_or(RngAlgorithm::RunMatLcg);
    if algo != RngAlgorithm::RunMatLcg {
        return Err(rng_error_with(
            &RNG_ERROR_GENERATOR_UNSUPPORTED,
            format!(
                "rng: generator '{}' is not supported in RunMat",
                algo.as_str()
            ),
        ));
    }
    match keyword {
        "default" | "twister" | "runmat-lcg" => Ok(ParsedCommand::Default),
        "shuffle" => Ok(ParsedCommand::Shuffle),
        other => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("rng: unknown option '{other}'"),
        )),
    }
}

fn apply_command(command: ParsedCommand) -> BuiltinResult<()> {
    match command {
        ParsedCommand::Default => {
            set_default()?;
            Ok(())
        }
        ParsedCommand::Seed(seed) => {
            set_seed(seed)?;
            Ok(())
        }
        ParsedCommand::Shuffle => {
            let seed = shuffle_seed();
            set_seed(seed)?;
            Ok(())
        }
        ParsedCommand::Restore(snapshot) => {
            random::apply_snapshot(snapshot)?;
            Ok(())
        }
    }
}

fn snapshot_to_value(snapshot: RngSnapshot) -> BuiltinResult<Value> {
    let mut struct_value = StructValue::new();
    let seed = snapshot.seed.unwrap_or(DEFAULT_USER_SEED);
    let seed_value =
        if crate::builtins::math::trigonometry::cos::integer_is_exact_f64(&IntValue::U64(seed)) {
            Value::Num(seed as f64)
        } else {
            Value::Int(IntValue::U64(seed))
        };
    struct_value.fields.insert(
        "Type".to_string(),
        Value::String(snapshot.algorithm.as_str().to_string()),
    );
    struct_value.fields.insert("Seed".to_string(), seed_value);
    let lo = (snapshot.state & 0xFFFF_FFFF) as f64;
    let hi = (snapshot.state >> 32) as f64;
    let tensor = Tensor::new(vec![lo, hi], vec![1, 2])
        .map_err(|e| rng_internal_error(format!("rng: {e}")))?;
    struct_value
        .fields
        .insert("State".to_string(), Value::Tensor(tensor));
    Ok(Value::Struct(struct_value))
}

fn snapshot_from_value(value: &Value) -> BuiltinResult<RngSnapshot> {
    let Value::Struct(struct_value) = value else {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: expected a structure with fields Type, Seed, and State",
        ));
    };
    let type_value = struct_value
        .fields
        .get("Type")
        .or_else(|| struct_value.fields.get("type"))
        .ok_or_else(|| rng_error(&RNG_ERROR_STATE_TYPE_FIELD_MISSING))?;
    let generator = match keyword_of(type_value) {
        Some(ref kw) => parse_generator_keyword(kw)?,
        None => {
            return Err(rng_error_with(
                &RNG_ERROR_INVALID_ARGUMENT,
                "rng: Type field must be a string",
            ))
        }
    };

    let seed_opt = struct_value
        .fields
        .get("Seed")
        .or_else(|| struct_value.fields.get("seed"))
        .map(|v| parse_seed_scalar(v, "rng: Seed"))
        .transpose()?;
    let state_value = struct_value
        .fields
        .get("State")
        .or_else(|| struct_value.fields.get("state"))
        .ok_or_else(|| {
            rng_error_with(
                &RNG_ERROR_INVALID_ARGUMENT,
                "rng: state struct is missing the 'State' field",
            )
        })?;
    let state = parse_state_scalar(state_value)?;
    Ok(RngSnapshot {
        state,
        seed: seed_opt,
        algorithm: generator,
    })
}

fn parse_generator(value: &Value) -> BuiltinResult<RngAlgorithm> {
    match keyword_of(value) {
        Some(keyword) => parse_generator_keyword(&keyword),
        None => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            "rng: generator name must be a string",
        )),
    }
}

fn parse_generator_keyword(keyword: &str) -> BuiltinResult<RngAlgorithm> {
    match keyword {
        "twister" | "default" | "runmat-lcg" => Ok(RngAlgorithm::RunMatLcg),
        other => Err(rng_error_with(
            &RNG_ERROR_GENERATOR_UNSUPPORTED,
            format!("rng: generator '{other}' is not supported"),
        )),
    }
}

fn parse_seed_scalar(value: &Value, label: &str) -> BuiltinResult<u64> {
    match value {
        Value::Int(i) => i.try_to_u64().ok_or_else(|| {
            rng_error_with(
                &RNG_ERROR_SEED_NONNEGATIVE,
                format!("{label}: seed must be non-negative"),
            )
        }),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    format!("{label}: seed must be finite"),
                ));
            }
            if *n < 0.0 {
                return Err(rng_error_with(
                    &RNG_ERROR_SEED_NONNEGATIVE,
                    format!("{label}: seed must be non-negative"),
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    format!("{label}: seed must be an integer"),
                ));
            }
            if rounded > (1u64 << 53) as f64 {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    format!("{label}: seed exceeds 53-bit integer precision"),
                ));
            }
            Ok(rounded as u64)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                int.try_to_u64().ok_or_else(|| {
                    rng_error_with(
                        &RNG_ERROR_SEED_NONNEGATIVE,
                        format!("{label}: seed must be non-negative"),
                    )
                })
            } else {
                parse_seed_scalar(&Value::Num(tensor::tensor_value_f64(t, 0)), label)
            }
        }
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: expected a numeric seed"),
        )),
        _ => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: expected a scalar numeric seed"),
        )),
    }
}

fn ensure_direct_seed_range(seed: u64) -> BuiltinResult<()> {
    if seed >= MATLAB_SEED_UPPER_BOUND {
        crate::compatibility::ensure_builtin_extension_enabled(&RNG_WIDE_SEED_EXTENSION, NAME)?;
    }
    Ok(())
}

fn ensure_rng_state_extensions(args: &[Value]) -> BuiltinResult<()> {
    if let [Value::Struct(state)] = args {
        let has_native_integer_field = ["Seed", "seed", "State", "state"]
            .into_iter()
            .filter_map(|name| state.fields.get(name))
            .any(crate::builtins::common::validation::value_has_native_integer_class);
        if has_native_integer_field {
            crate::compatibility::ensure_builtin_extension_enabled(
                &RNG_TYPED_STATE_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

fn parse_state_scalar(value: &Value) -> BuiltinResult<u64> {
    match value {
        Value::Tensor(t) => match tensor::element_count(&t.shape) {
            1 => {
                if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                    int.try_to_u64().ok_or_else(|| {
                        rng_error_with(
                            &RNG_ERROR_INVALID_ARGUMENT,
                            "rng: State must be non-negative",
                        )
                    })
                } else {
                    parse_state_scalar(&Value::Num(tensor::tensor_value_f64(t, 0)))
                }
            }
            2 => {
                let (lo, hi) = if let Some(storage) = t.integer_storage() {
                    let lo = storage.value_at(0).ok_or_else(|| {
                        rng_error_with(
                            &RNG_ERROR_INVALID_ARGUMENT,
                            "rng: State tensor must contain one or two elements",
                        )
                    })?;
                    let hi = storage.value_at(1).ok_or_else(|| {
                        rng_error_with(
                            &RNG_ERROR_INVALID_ARGUMENT,
                            "rng: State tensor must contain one or two elements",
                        )
                    })?;
                    (
                        parse_state_word_int(lo, "rng: State[1]")?,
                        parse_state_word_int(hi, "rng: State[2]")?,
                    )
                } else {
                    (
                        parse_state_word(tensor::tensor_value_f64(t, 0), "rng: State[1]")?,
                        parse_state_word(tensor::tensor_value_f64(t, 1), "rng: State[2]")?,
                    )
                };
                Ok(lo | (hi << 32))
            }
            _ => Err(rng_error_with(
                &RNG_ERROR_INVALID_ARGUMENT,
                "rng: State tensor must contain one or two elements",
            )),
        },
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    "rng: State must be finite",
                ));
            }
            if *n < 0.0 {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    "rng: State must be non-negative",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(rng_error_with(
                    &RNG_ERROR_INVALID_ARGUMENT,
                    "rng: State must be an integer vector",
                ));
            }
            Ok(rounded as u64)
        }
        Value::Int(i) => i.try_to_u64().ok_or_else(|| {
            rng_error_with(
                &RNG_ERROR_INVALID_ARGUMENT,
                "rng: State must be non-negative",
            )
        }),
        other => Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("rng: unsupported State value {other:?}"),
        )),
    }
}

fn parse_state_word_int(value: IntValue, label: &str) -> BuiltinResult<u64> {
    let word = value.try_to_u64().ok_or_else(|| {
        rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must be non-negative"),
        )
    })?;
    if word > u32::MAX as u64 {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must fit in uint32"),
        ));
    }
    Ok(word)
}

fn parse_state_word(value: f64, label: &str) -> BuiltinResult<u64> {
    if !value.is_finite() {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must be finite"),
        ));
    }
    if value < 0.0 {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must be non-negative"),
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: must be an integer"),
        ));
    }
    if rounded > (u32::MAX as f64) {
        return Err(rng_error_with(
            &RNG_ERROR_INVALID_ARGUMENT,
            format!("{label}: exceeds uint32 precision"),
        ));
    }
    Ok(rounded as u64)
}

fn shuffle_seed() -> u64 {
    if let Ok(env) = std::env::var("RUNMAT_RNG_SHUFFLE_SEED") {
        if let Ok(parsed) = env.parse::<u64>() {
            return parsed;
        }
    }
    let now = unix_timestamp_ns();
    let mut seed = now as u64 ^ (now >> 32) as u64;
    let addr = (&seed as *const u64 as u64).rotate_left(21);
    seed ^= addr ^ (seed << 7);
    if seed == 0 {
        DEFAULT_USER_SEED.wrapping_add(1)
    } else {
        seed
    }
}

fn sync_provider_state(state: u64) {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Err(err) = provider.set_rng_state(state) {
            debug!("rng: provider seed sync failed: {err}");
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};
    use crate::dispatcher::download_handle_async;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, ResolveContext, Tensor, Type};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_returns_current_state() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        let value = block_on(rng_builtin(Vec::new())).expect("rng");
        let snapshot = snapshot_from_value(&value).expect("snapshot");
        assert_eq!(snapshot.state, random::default_snapshot().state);
        assert_eq!(snapshot.seed, Some(DEFAULT_USER_SEED));
        assert_eq!(snapshot.algorithm, RngAlgorithm::RunMatLcg);
    }

    #[test]
    fn rng_type_returns_struct() {
        let out = rng_type(&[], &ResolveContext::new(Vec::new()));
        assert_eq!(
            out,
            Type::Struct {
                known_fields: Some(vec![
                    "Seed".to_string(),
                    "State".to_string(),
                    "Type".to_string(),
                ])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_seed_is_reproducible() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        block_on(rng_builtin(vec![Value::Int(IntValue::U32(42))])).expect("rng");
        let seq1 = random::generate_uniform(5, "rng test").expect("uniform");
        block_on(rng_builtin(vec![Value::Int(IntValue::U32(42))])).expect("rng");
        let seq2 = random::generate_uniform(5, "rng test").expect("uniform");
        assert_eq!(seq1, seq2);
    }

    #[test]
    fn rng_integer_seed_and_state_preserve_full_uint64_range() {
        assert_eq!(
            parse_seed_scalar(&Value::Int(IntValue::U64(u64::MAX)), "rng: seed")
                .expect("uint64 seed"),
            u64::MAX
        );
        assert_eq!(
            parse_state_scalar(&Value::Int(IntValue::U64(u64::MAX))).expect("uint64 state"),
            u64::MAX
        );
        assert!(parse_seed_scalar(&Value::Int(IntValue::I64(-1)), "rng: seed").is_err());
    }

    #[test]
    fn rng_wide_seed_is_gated_and_query_preserves_exact_seed() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        let wide = (1_u64 << 53) + 1;
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(rng_builtin(vec![Value::Int(IntValue::U64(wide))]))
                .expect_err("wide seed must gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:RngWideSeedExtension")
            );
        }
        {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            block_on(rng_builtin(vec![Value::Int(IntValue::U64(wide))])).expect("RunMat wide seed");
            let Value::Struct(snapshot) = block_on(rng_builtin(Vec::new())).expect("snapshot")
            else {
                panic!("expected snapshot struct");
            };
            assert_eq!(
                snapshot.fields.get("Seed"),
                Some(&Value::Int(IntValue::U64(wide)))
            );
        }
    }

    #[test]
    fn rng_typed_state_fields_are_an_independent_extension() {
        let mut state = StructValue::new();
        state
            .fields
            .insert("Type".to_string(), Value::from("twister"));
        state
            .fields
            .insert("Seed".to_string(), Value::Int(IntValue::U32(7)));
        state.fields.insert(
            "State".to_string(),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U32(vec![1, 2]), vec![1, 2])
                    .expect("state words"),
            ),
        );
        let value = || Value::Struct(state.clone());
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(rng_builtin(vec![value()])).expect_err("typed state must gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:RngTypedStateFieldsExtension")
            );
        }
        {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            block_on(rng_builtin(vec![value()])).expect("typed state extension");
        }
    }

    #[test]
    fn rng_integer_capabilities_cover_public_and_extended_seed_domains() {
        assert_eq!(RNG_INTEGER_CAPABILITIES.len(), 3);
        assert_eq!(
            RNG_INTEGER_CAPABILITIES[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::Documented
        );
        assert!(RNG_INTEGER_CAPABILITIES[1..].iter().all(|capability| {
            capability.inputs[0].availability == BuiltinIntegerInputAvailability::RunMatOnly
        }));
    }

    #[test]
    fn rng_typed_integer_tensor_seed_and_state_are_exact() {
        let seed =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("seed");
        let seed = seed;
        assert_eq!(
            parse_seed_scalar(&Value::Tensor(seed), "rng: seed").expect("typed seed"),
            u64::MAX
        );

        let scalar_state =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("state");
        let scalar_state = scalar_state;
        assert_eq!(
            parse_state_scalar(&Value::Tensor(scalar_state)).expect("typed state"),
            u64::MAX
        );

        let word_state = Tensor::new_integer(
            IntegerStorage::U64(vec![u32::MAX as u64, u32::MAX as u64]),
            vec![1, 2],
        )
        .expect("state words");
        let word_state = word_state;
        assert_eq!(
            parse_state_scalar(&Value::Tensor(word_state)).expect("typed state words"),
            u64::MAX
        );

        let negative_seed =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("seed");
        let negative_seed = negative_seed;
        assert!(parse_seed_scalar(&Value::Tensor(negative_seed), "rng: seed").is_err());

        let wide_word = Tensor::new_integer(
            IntegerStorage::U64(vec![u32::MAX as u64 + 1, 0]),
            vec![1, 2],
        )
        .expect("state word");
        let wide_word = wide_word;
        assert!(parse_state_scalar(&Value::Tensor(wide_word)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_restore_struct_roundtrip() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        let saved = block_on(rng_builtin(Vec::new())).expect("rng");
        block_on(rng_builtin(vec![Value::Int(IntValue::U32(7))])).expect("rng");
        block_on(rng_builtin(vec![saved.clone()])).expect("rng restore");
        let current = random::snapshot().expect("snapshot");
        assert_eq!(current.state, random::default_snapshot().state);
        assert_eq!(current.seed, Some(DEFAULT_USER_SEED));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_default_restores_state() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        block_on(rng_builtin(vec![Value::Int(IntValue::U32(99))])).expect("seed rng");
        let previous = block_on(rng_builtin(vec![Value::from("default")])).expect("rng default");
        let restored = random::snapshot().expect("snapshot");
        assert_eq!(restored.state, random::default_snapshot().state);
        assert_eq!(restored.seed, Some(DEFAULT_USER_SEED));
        let prev_snapshot = snapshot_from_value(&previous).expect("prev snapshot");
        assert_eq!(prev_snapshot.seed, Some(99));
        assert_ne!(prev_snapshot.state, restored.state);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_seed_with_twister_alias() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        block_on(rng_builtin(vec![Value::Int(IntValue::U32(123))])).expect("rng seed first");
        let host_seq = random::generate_uniform(4, "twister alias host").expect("uniform");
        random::reset_rng();
        block_on(rng_builtin(vec![
            Value::Int(IntValue::U32(123)),
            Value::from("twister"),
        ]))
        .expect("rng seed twister");
        let alias_seq = random::generate_uniform(4, "twister alias verify").expect("uniform");
        assert_eq!(host_seq, alias_seq);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_rejects_negative_seed() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        let err = block_on(rng_builtin(vec![Value::Int(IntValue::I32(-5))])).unwrap_err();
        assert_eq!(err.identifier(), RNG_ERROR_SEED_NONNEGATIVE.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_rejects_unknown_generator() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        let err = block_on(rng_builtin(vec![
            Value::from("default"),
            Value::from("philox"),
        ]))
        .unwrap_err();
        assert_eq!(err.identifier(), RNG_ERROR_GENERATOR_UNSUPPORTED.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_state_struct_requires_type() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        let tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).expect("tensor");
        let mut st = StructValue::new();
        st.fields.insert("Seed".to_string(), Value::Num(0.0));
        st.fields.insert("State".to_string(), Value::Tensor(tensor));
        let err = block_on(rng_builtin(vec![Value::Struct(st)])).unwrap_err();
        assert_eq!(
            err.identifier(),
            RNG_ERROR_STATE_TYPE_FIELD_MISSING.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_syncs_provider_state() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            block_on(rng_builtin(vec![Value::Int(IntValue::U32(9))])).expect("rng");
            let handle = provider.random_uniform(&[4, 1]).expect("gpu uniform");
            let host_after_gpu = random::generate_uniform(4, "rng provider sync").expect("uniform");
            let gpu = block_on(download_handle_async(provider, &handle)).expect("download");
            assert_eq!(gpu.data, host_after_gpu);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rng_wgpu_uniform_matches_cpu() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        block_on(rng_builtin(vec![Value::Int(IntValue::U32(2024))])).expect("rng wgpu seed");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let handle = provider
            .random_uniform(&[1, 6])
            .expect("wgpu random uniform");
        let gpu = block_on(download_handle_async(provider, &handle)).expect("wgpu download");
        let host = random::generate_uniform(6, "rng wgpu parity").expect("host uniform sequence");
        assert_eq!(gpu.data.len(), host.len());
        for (idx, value) in gpu.data.iter().enumerate() {
            assert!(value.is_finite(), "gpu value at {idx} not finite: {value}");
            assert!(
                *value >= 0.0 && *value < 1.0,
                "gpu value at {idx} out of [0,1): {value}"
            );
        }
        for (idx, value) in host.iter().enumerate() {
            assert!(value.is_finite(), "host value at {idx} not finite: {value}");
            assert!(
                *value >= 0.0 && *value < 1.0,
                "host value at {idx} out of [0,1): {value}"
            );
        }
        let _ = provider.free(&handle);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rng_shuffle_uses_entropy_or_override() {
        let _guard = random::test_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        random::reset_rng();
        unsafe { std::env::set_var("RUNMAT_RNG_SHUFFLE_SEED", "12345") };
        block_on(rng_builtin(vec![Value::from("shuffle")])).expect("rng shuffle");
        unsafe { std::env::remove_var("RUNMAT_RNG_SHUFFLE_SEED") };
        let current = random::snapshot().expect("snapshot");
        assert_eq!(current.seed, Some(12345));
    }
}
