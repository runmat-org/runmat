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

const BUILTIN_NAME: &str = "unifrnd";

const UNIFRND_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from uniform distribution.",
}];

const UNIFRND_INPUTS_A_B: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower bound parameter.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound parameter (must be > a).",
    },
];

const UNIFRND_INPUTS_A_B_SZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower bound parameter.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound parameter (must be > a).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size scalar or size vector argument.",
    },
];

const UNIFRND_INPUTS_A_B_DIMS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Lower bound parameter.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound parameter (must be > a).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension extents for output shape.",
    },
];

const UNIFRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = unifrnd(a, b)",
        inputs: &UNIFRND_INPUTS_A_B,
        outputs: &UNIFRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unifrnd(a, b, sz)",
        inputs: &UNIFRND_INPUTS_A_B_SZ,
        outputs: &UNIFRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unifrnd(a, b, sz1, sz2, ...)",
        inputs: &UNIFRND_INPUTS_A_B_DIMS,
        outputs: &UNIFRND_OUTPUT_R,
    },
];

const UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.UNIFRND.LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND",
        identifier: Some("RunMat:unifrnd:LowerBoundMustBeLessThanUpperBound"),
        when: "a is greater than or equal to b.",
        message: "unifrnd: a must be less than b",
    };

const UNIFRND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIFRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:unifrnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing or malformed.",
    message: "unifrnd: invalid argument",
};

const UNIFRND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNIFRND.INTERNAL",
    identifier: Some("RunMat:unifrnd:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "unifrnd: internal operation failed",
};

const UNIFRND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND,
    UNIFRND_ERROR_INVALID_ARGUMENT,
    UNIFRND_ERROR_INTERNAL,
];

pub const UNIFRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNIFRND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &UNIFRND_ERRORS,
};

fn unifrnd_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn unifrnd_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    unifrnd_error_with(error, error.message)
}

fn unifrnd_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    unifrnd_error_with(&UNIFRND_ERROR_INTERNAL, message)
}

fn unifrnd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() <= 2 {
        Type::Num
    } else {
        Type::Unknown
    }
}

#[runtime_builtin(
    name = "unifrnd",
    category = "stats/random",
    summary = "Generate uniform random samples on interval [a, b).",
    keywords = "unifrnd,uniform,random,distribution,statistics",
    type_resolver(unifrnd_type),
    descriptor(crate::builtins::stats::random::unifrnd::UNIFRND_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::random::unifrnd"
)]
async fn unifrnd_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (a, b, shape) = parse_args(args).await?;
    if a >= b {
        return Err(unifrnd_error(
            &UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND,
        ));
    }
    if let Some(value) = try_gpu_unifrnd(a, b, &shape)? {
        return Ok(value);
    }
    let len = tensor::element_count(&shape);
    let data = random::generate_uniform_scaled(a, b, len, "unifrnd")?;
    let t =
        Tensor::new(data, shape).map_err(|e| unifrnd_internal_error(format!("unifrnd: {e}")))?;
    Ok(tensor::tensor_into_value(t))
}

async fn parse_args(args: Vec<Value>) -> crate::BuiltinResult<(f64, f64, Vec<usize>)> {
    if args.len() < 2 {
        return Err(unifrnd_error_with(
            &UNIFRND_ERROR_INVALID_ARGUMENT,
            "unifrnd: requires at least two arguments (a, b)",
        ));
    }
    let a = scalar_f64(&args[0])?;
    let b = scalar_f64(&args[1])?;
    let shape = parse_shape_args(&args[2..]).await?;
    Ok((a, b, shape))
}

fn scalar_f64(value: &Value) -> crate::BuiltinResult<f64> {
    match value {
        Value::Num(v) => Ok(*v),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(unifrnd_error_with(
            &UNIFRND_ERROR_INVALID_ARGUMENT,
            format!("unifrnd: expected scalar parameter, got {other:?}"),
        )),
    }
}

async fn parse_shape_args(rest: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims: Vec<usize> = Vec::new();
    for arg in rest {
        match extract_dims(arg, "unifrnd").await? {
            Some(d) => dims.extend(d),
            None => {
                return Err(unifrnd_error_with(
                    &UNIFRND_ERROR_INVALID_ARGUMENT,
                    format!("unifrnd: invalid size argument: {arg:?}"),
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

fn try_gpu_unifrnd(a: f64, b: f64, shape: &[usize]) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if provider.precision() != runmat_accelerate_api::ProviderPrecision::F64 {
        return Ok(None);
    }
    match provider.random_unifrnd(a, b, shape) {
        Ok(handle) => {
            let len = tensor::element_count(shape);
            random::skip_uniform(len, "unifrnd")?;
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

    struct CpuOnlyProvider;

    impl runmat_accelerate_api::AccelProvider for CpuOnlyProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            Err(anyhow::anyhow!("cpu-only test provider does not upload"))
        }

        fn download<'a>(
            &'a self,
            _handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow::anyhow!("cpu-only test provider does not download")) })
        }

        fn free(&self, _handle: &runmat_accelerate_api::GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "cpu-only test provider".to_string()
        }

        fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
            runmat_accelerate_api::ProviderPrecision::F32
        }
    }

    static CPU_ONLY_PROVIDER: CpuOnlyProvider = CpuOnlyProvider;

    fn reset_cpu_path() -> runmat_accelerate_api::ThreadProviderGuard {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
        runmat_accelerate_api::ThreadProviderGuard::set(Some(&CPU_ONLY_PROVIDER))
    }

    #[test]
    fn unifrnd_scalar_deterministic() {
        let _guard = random::test_lock().lock().unwrap();
        let _provider_guard = reset_cpu_path();
        let result =
            block_on(unifrnd_builtin(vec![Value::Num(2.0), Value::Num(5.0)])).expect("unifrnd");
        let expected = random::expected_uniform_scaled_sequence(2.0, 5.0, 1)[0];
        match result {
            Value::Num(v) => {
                assert!((2.0..5.0).contains(&v));
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn unifrnd_matrix_dims() {
        let _guard = random::test_lock().lock().unwrap();
        let _provider_guard = reset_cpu_path();
        let args = vec![
            Value::Num(0.0),
            Value::Num(10.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let result = block_on(unifrnd_builtin(args)).expect("unifrnd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 4]);
                assert!(t.data.iter().all(|&v| (0.0..10.0).contains(&v)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unifrnd_size_vec() {
        let _guard = random::test_lock().lock().unwrap();
        let _provider_guard = reset_cpu_path();
        let size = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let args = vec![Value::Num(0.0), Value::Num(1.0), Value::Tensor(size)];
        let result = block_on(unifrnd_builtin(args)).expect("unifrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unifrnd_rejects_a_ge_b() {
        let args = vec![Value::Num(5.0), Value::Num(2.0)];
        let err = block_on(unifrnd_builtin(args)).expect_err("a >= b should error");
        assert_eq!(
            err.identifier(),
            UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND.identifier
        );
    }

    #[test]
    fn unifrnd_rejects_a_eq_b() {
        let args = vec![Value::Num(3.0), Value::Num(3.0)];
        let err = block_on(unifrnd_builtin(args)).expect_err("a == b should error");
        assert_eq!(
            err.identifier(),
            UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND.identifier
        );
    }

    #[test]
    fn unifrnd_distribution_bounds() {
        let _guard = random::test_lock().lock().unwrap();
        let _provider_guard = reset_cpu_path();
        let a = 2.0_f64;
        let b = 7.0_f64;
        let n = 50_000_usize;
        let args = vec![
            Value::Num(a),
            Value::Num(b),
            Value::Num(n as f64),
            Value::Num(1.0),
        ];
        let result = block_on(unifrnd_builtin(args)).expect("unifrnd");
        let data = match result {
            Value::Tensor(t) => t.data,
            other => panic!("expected tensor, got {other:?}"),
        };
        assert!(
            data.iter().all(|&v| v >= a && v < b),
            "some values outside [{a}, {b})"
        );
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let expected_mean = (a + b) / 2.0;
        assert!(
            (mean - expected_mean).abs() / (b - a) < 0.05,
            "sample mean {mean:.4} not within 5% of expected {expected_mean:.4}"
        );
    }
}
