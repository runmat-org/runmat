use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::random;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;

const BUILTIN_NAME: &str = "normrnd";

const NORMRND_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from normal distribution.",
}];

const NORMRND_INPUTS_MU_SIGMA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean parameter.",
    },
    BuiltinParamDescriptor {
        name: "sigma",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard deviation parameter (must be >= 0).",
    },
];

const NORMRND_INPUTS_MU_SIGMA_SZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean parameter.",
    },
    BuiltinParamDescriptor {
        name: "sigma",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard deviation parameter (must be >= 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size scalar or size vector argument.",
    },
];

const NORMRND_INPUTS_MU_SIGMA_DIMS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean parameter.",
    },
    BuiltinParamDescriptor {
        name: "sigma",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard deviation parameter (must be >= 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension extents for output shape.",
    },
];

const NORMRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = normrnd(mu, sigma)",
        inputs: &NORMRND_INPUTS_MU_SIGMA,
        outputs: &NORMRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = normrnd(mu, sigma, sz)",
        inputs: &NORMRND_INPUTS_MU_SIGMA_SZ,
        outputs: &NORMRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = normrnd(mu, sigma, sz1, sz2, ...)",
        inputs: &NORMRND_INPUTS_MU_SIGMA_DIMS,
        outputs: &NORMRND_OUTPUT_R,
    },
];

const NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMRND.SIGMA_MUST_BE_NONNEGATIVE",
    identifier: Some("RunMat:normrnd:SigmaMustBeNonnegative"),
    when: "sigma is negative.",
    message: "normrnd: sigma must be non-negative",
};

const NORMRND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:normrnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing or malformed.",
    message: "normrnd: invalid argument",
};

const NORMRND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMRND.INTERNAL",
    identifier: Some("RunMat:normrnd:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "normrnd: internal operation failed",
};

const NORMRND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE,
    NORMRND_ERROR_INVALID_ARGUMENT,
    NORMRND_ERROR_INTERNAL,
];

pub const NORMRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NORMRND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NORMRND_ERRORS,
};

fn normrnd_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn normrnd_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    normrnd_error_with(error, error.message)
}

fn normrnd_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    normrnd_error_with(&NORMRND_ERROR_INTERNAL, message)
}

fn normrnd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() <= 2 {
        Type::Num
    } else {
        Type::Unknown
    }
}

#[runtime_builtin(
    name = "normrnd",
    category = "stats/random",
    summary = "Normally-distributed random numbers with mean mu and standard deviation sigma.",
    keywords = "normrnd,normal,gaussian,random,distribution,statistics",
    type_resolver(normrnd_type),
    descriptor(crate::builtins::stats::random::normrnd::NORMRND_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::random::normrnd"
)]
async fn normrnd_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (mu, sigma, shape) = parse_args(args).await?;
    if sigma < 0.0 {
        return Err(normrnd_error(&NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE));
    }
    if let Some(value) = try_gpu_normrnd(mu, sigma, &shape)? {
        return Ok(value);
    }
    let len = tensor::element_count(&shape);
    let data = random::generate_normal_scaled(mu, sigma, len, "normrnd")?;
    let t =
        Tensor::new(data, shape).map_err(|e| normrnd_internal_error(format!("normrnd: {e}")))?;
    Ok(tensor::tensor_into_value(t))
}

async fn parse_args(args: Vec<Value>) -> crate::BuiltinResult<(f64, f64, Vec<usize>)> {
    if args.len() < 2 {
        return Err(normrnd_error_with(
            &NORMRND_ERROR_INVALID_ARGUMENT,
            "normrnd: requires at least two arguments (mu, sigma)",
        ));
    }
    let mu = scalar_f64(&args[0])?;
    let sigma = scalar_f64(&args[1])?;
    let shape = parse_shape_args(&args[2..]).await?;
    Ok((mu, sigma, shape))
}

fn scalar_f64(value: &Value) -> crate::BuiltinResult<f64> {
    match value {
        Value::Num(v) => Ok(*v),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(normrnd_error_with(
            &NORMRND_ERROR_INVALID_ARGUMENT,
            format!("normrnd: expected scalar parameter, got {other:?}"),
        )),
    }
}

async fn parse_shape_args(rest: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims: Vec<usize> = Vec::new();
    for arg in rest {
        match extract_dims(arg, "normrnd").await? {
            Some(d) => dims.extend(d),
            None => {
                return Err(normrnd_error_with(
                    &NORMRND_ERROR_INVALID_ARGUMENT,
                    format!("normrnd: invalid size argument: {arg:?}"),
                ))
            }
        }
    }
    Ok(normalize_dims(dims))
}

fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        dims
    }
}

fn try_gpu_normrnd(mu: f64, sigma: f64, shape: &[usize]) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
        return Ok(None);
    }
    match provider.random_normrnd(mu, sigma, shape) {
        Ok(handle) => {
            let len = tensor::element_count(shape);
            // Box-Muller emits two normals per two uniform samples.
            let uniform_count = len.saturating_add(1) / 2 * 2;
            random::skip_uniform(uniform_count, "normrnd")?;
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;
    use futures::executor::block_on;

    fn reset() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    #[test]
    fn normrnd_scalar_deterministic() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let result =
            block_on(normrnd_builtin(vec![Value::Num(0.0), Value::Num(1.0)])).expect("normrnd");
        let expected = random::expected_normal_scaled_sequence(0.0, 1.0, 1)[0];
        match result {
            Value::Num(v) => assert!((v - expected).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_matrix_dims() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let args = vec![
            Value::Num(5.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let result = block_on(normrnd_builtin(args)).expect("normrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_size_vec() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let size = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let args = vec![Value::Num(0.0), Value::Num(1.0), Value::Tensor(size)];
        let result = block_on(normrnd_builtin(args)).expect("normrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_rejects_negative_sigma() {
        let args = vec![Value::Num(0.0), Value::Num(-1.0)];
        let err = block_on(normrnd_builtin(args)).expect_err("negative sigma should error");
        assert_eq!(
            err.identifier(),
            NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE.identifier
        );
    }

    #[test]
    fn normrnd_distribution_mean_and_std() {
        let _guard = random::test_lock().lock().unwrap();
        reset();
        let mu = 5.0_f64;
        let sigma = 2.0_f64;
        let n = 50_000_usize;
        let args = vec![
            Value::Num(mu),
            Value::Num(sigma),
            Value::Num(n as f64),
            Value::Num(1.0),
        ];
        let result = block_on(normrnd_builtin(args)).expect("normrnd");
        let data = match result {
            Value::Tensor(t) => t.data,
            other => panic!("expected tensor, got {other:?}"),
        };
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        assert!(
            (mean - mu).abs() / sigma.max(1.0) < 0.05,
            "sample mean {mean:.4} not within 5% tolerance of mu={mu}"
        );
        assert!(
            (std_dev - sigma).abs() / sigma < 0.05,
            "sample std {std_dev:.4} not within 5% of sigma={sigma}"
        );
    }
}
