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
use runmat_value::{Tensor, Value};

use crate::build_runtime_error;
use crate::builtins::common::random;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;

const BUILTIN_NAME: &str = "unifrnd";

const INTEGER_LOWER_BOUND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unifrnd-integer-lower-bound",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unifrnd with a typed-integer lower bound is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnifrndIntegerLowerBoundExtension"),
};
const INTEGER_UPPER_BOUND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unifrnd-integer-upper-bound",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unifrnd with a typed-integer upper bound is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnifrndIntegerUpperBoundExtension"),
};
const INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unifrnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unifrnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnifrndIntegerSizeExtension"),
};
pub const UNIFRND_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_LOWER_BOUND_EXTENSION,
    INTEGER_UPPER_BOUND_EXTENSION,
    INTEGER_SIZE_EXTENSION,
];
const INTEGER_BOUND_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "a",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double lower bounds; typed integers cross a checked binary64 sampling boundary in RunMat mode.",
    },
    BuiltinIntegerInputCapability {
        name: "b",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double upper bounds; typed integers cross a checked binary64 sampling boundary in RunMat mode.",
    },
];
const INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sz, sz1, ...",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The compatibility target documents single and double size controls; RunMat mode decodes typed integer extents exactly as structural values.",
}];
pub const UNIFRND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = unifrnd(integer_a, integer_b, ___)",
        inputs: &INTEGER_BOUND_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each typed bound is independently gated and must be exactly representable before entering the binary64 uniform generator.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = unifrnd(a, b, integer_sz)",
        inputs: &INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed extents are extension-gated before provider access and parsed exactly without selecting the distribution computation domain.",
    },
];

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
    extensions(crate::builtins::stats::random::unifrnd::UNIFRND_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::random::unifrnd::UNIFRND_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::random::unifrnd"
)]
async fn unifrnd_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    enforce_integer_extensions(&args).await?;
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

async fn enforce_integer_extensions(args: &[Value]) -> crate::BuiltinResult<()> {
    if let Some(value) = args.first() {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            value,
            &INTEGER_LOWER_BOUND_EXTENSION,
            BUILTIN_NAME,
            "lower-bound",
        )
        .await?;
    }
    if let Some(value) = args.get(1) {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            value,
            &INTEGER_UPPER_BOUND_EXTENSION,
            BUILTIN_NAME,
            "upper-bound",
        )
        .await?;
    }
    if args
        .iter()
        .skip(2)
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn parse_args(args: Vec<Value>) -> crate::BuiltinResult<(f64, f64, Vec<usize>)> {
    if args.len() < 2 {
        return Err(unifrnd_error_with(
            &UNIFRND_ERROR_INVALID_ARGUMENT,
            "unifrnd: requires at least two arguments (a, b)",
        ));
    }
    let a = scalar_f64(&args[0]).await?;
    let b = scalar_f64(&args[1]).await?;
    let shape = parse_shape_args(&args[2..]).await?;
    Ok((a, b, shape))
}

async fn scalar_f64(value: &Value) -> crate::BuiltinResult<f64> {
    tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| {
            unifrnd_error_with(&UNIFRND_ERROR_INVALID_ARGUMENT, format!("unifrnd: {err}"))
        })?
        .ok_or_else(|| {
            unifrnd_error_with(
                &UNIFRND_ERROR_INVALID_ARGUMENT,
                format!("unifrnd: expected scalar parameter, got {value:?}"),
            )
        })
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
    use runmat_value::IntegerStorage;

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

    fn reset_cpu_path() -> (impl Drop, runmat_accelerate_api::ThreadProviderGuard) {
        let state_guard = random::test_guard();
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
        (
            state_guard,
            runmat_accelerate_api::ThreadProviderGuard::set(Some(&CPU_ONLY_PROVIDER)),
        )
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("integer tensor")
    }

    #[test]
    fn unifrnd_scalar_deterministic() {
        let _guard = random::test_guard();
        let (_state_guard, _provider_guard) = reset_cpu_path();
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
        let _guard = random::test_guard();
        let (_state_guard, _provider_guard) = reset_cpu_path();
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
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|&v| (0.0..10.0).contains(&v)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unifrnd_size_vec() {
        let _guard = random::test_guard();
        let (_state_guard, _provider_guard) = reset_cpu_path();
        let size = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let args = vec![Value::Num(0.0), Value::Num(1.0), Value::Tensor(size)];
        let result = block_on(unifrnd_builtin(args)).expect("unifrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unifrnd_reads_typed_integer_parameters_and_size_exactly() {
        let _guard = random::test_guard();
        let _provider_guard = reset_cpu_path();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let a = poisoned_int_tensor(IntegerStorage::I16(vec![-2]), vec![1, 1]);
        let b = poisoned_int_tensor(IntegerStorage::U16(vec![3]), vec![1, 1]);
        let size = poisoned_int_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
        let result = block_on(unifrnd_builtin(vec![
            Value::Tensor(a),
            Value::Tensor(b),
            Value::Tensor(size),
        ]))
        .expect("unifrnd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|&v| (-2.0..3.0).contains(&v)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unifrnd_typed_integer_roles_are_gated_and_wide_bounds_must_be_exact() {
        let _guard = random::test_lock().lock().unwrap();
        let _provider_guard = reset_cpu_path();
        let compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
        let bound_error = block_on(unifrnd_builtin(vec![
            Value::Int(runmat_value::IntValue::I16(0)),
            Value::Num(2.0),
        ]))
        .expect_err("typed lower bound must be gated");
        assert_eq!(
            bound_error.identifier(),
            INTEGER_LOWER_BOUND_EXTENSION.error_identifier
        );
        let size_error = block_on(unifrnd_builtin(vec![
            Value::Num(0.0),
            Value::Num(2.0),
            Value::Int(runmat_value::IntValue::U8(2)),
        ]))
        .expect_err("typed size must be gated");
        assert_eq!(
            size_error.identifier(),
            INTEGER_SIZE_EXTENSION.error_identifier
        );
        drop(compatibility);

        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let lossy = block_on(unifrnd_builtin(vec![
            Value::Num(0.0),
            Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
        ]))
        .expect_err("lossy upper bound must reject");
        assert!(lossy.message().contains("exactly representable"));
        drop(extensions);
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
        let _guard = random::test_guard();
        let (_state_guard, _provider_guard) = reset_cpu_path();
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
            Value::Tensor(t) => t.materialize_f64(),
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
