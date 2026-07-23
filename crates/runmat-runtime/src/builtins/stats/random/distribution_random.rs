//! Additional distribution-specific random-number generators.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

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

fn finish(name: &'static str, data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| random_internal(name, format!("{name}: {err}")))
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
        if !value.is_finite() || *value < 0.0 || value.fract() != 0.0 {
            return Err(random_error(
                name,
                format!("{name}: number of trials must be a nonnegative integer"),
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

    #[runtime_builtin(
        name = "gamrnd",
        category = "stats/random",
        summary = "Generate gamma-distributed random samples.",
        keywords = "gamrnd,gamma,random,distribution,statistics",
        type_resolver(super::random_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::random::distribution_random::gamrnd"
    )]
    pub(crate) async fn gamrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let args = parse_two_parameter_args("gamrnd", args).await?;
        validate_gamma("gamrnd", &args)?;
        let len = tensor::element_count(&args.shape);
        let data = random::generate_gamma_shape_scale(&args.first, &args.second, len, "gamrnd")
            .map_err(|err| random_internal("gamrnd", err.message().to_string()))?;
        finish("gamrnd", data, args.shape)
    }
}

pub mod binornd {
    use super::*;
    random_descriptor!("binornd", BINORND_SIGNATURES);

    #[runtime_builtin(
        name = "binornd",
        category = "stats/random",
        summary = "Generate binomially-distributed random samples.",
        keywords = "binornd,binomial,random,distribution,statistics",
        type_resolver(super::random_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::random::distribution_random::binornd"
    )]
    pub(crate) async fn binornd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let args = parse_two_parameter_args("binornd", args).await?;
        validate_binomial("binornd", &args)?;
        let len = tensor::element_count(&args.shape);
        let data = random::generate_binomial(&args.first, &args.second, len, "binornd")
            .map_err(|err| random_internal("binornd", err.message().to_string()))?;
        finish("binornd", data, args.shape)
    }
}

pub mod wblrnd {
    use super::*;
    random_descriptor!("wblrnd", WBLRND_SIGNATURES);

    #[runtime_builtin(
        name = "wblrnd",
        category = "stats/random",
        summary = "Generate Weibull-distributed random samples.",
        keywords = "wblrnd,weibull,random,distribution,statistics",
        type_resolver(super::random_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::random::distribution_random::wblrnd"
    )]
    pub(crate) async fn wblrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let args = parse_two_parameter_args("wblrnd", args).await?;
        validate_weibull("wblrnd", &args)?;
        let len = tensor::element_count(&args.shape);
        let data = random::generate_weibull(&args.first, &args.second, len, "wblrnd")
            .map_err(|err| random_internal("wblrnd", err.message().to_string()))?;
        finish("wblrnd", data, args.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn reset() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    #[test]
    fn gamrnd_accepts_broadcast_and_size_forms() {
        let _guard = random::test_guard();
        reset();
        let out = block_on(gamrnd::gamrnd_builtin(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
            Value::Num(2.0),
        ]))
        .expect("gamrnd");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert!(tensor.data.iter().all(|value| *value >= 0.0));
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
    fn binornd_accepts_array_parameters_and_rejects_fractional_trials() {
        let _guard = random::test_guard();
        reset();
        let out = block_on(binornd::binornd_builtin(vec![
            Value::Tensor(Tensor::new(vec![5.0, 10.0], vec![1, 2]).unwrap()),
            Value::Num(0.5),
        ]))
        .expect("binornd");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert!(tensor.data[0] >= 0.0 && tensor.data[0] <= 5.0);
                assert!(tensor.data[1] >= 0.0 && tensor.data[1] <= 10.0);
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
        let _guard = random::test_guard();
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
    fn nonscalar_parameters_must_match_explicit_size() {
        let _guard = random::test_guard();
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
        let _guard = random::test_guard();
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
                assert!(tensor.data.iter().all(|value| *value >= 0.0));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}
