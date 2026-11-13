use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    UNIFRND_ERROR_INTERNAL, UNIFRND_ERROR_INVALID_ARGUMENT,
    UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND,
    UNIFRND_INTEGER_LOWER_BOUND_EXTENSION as INTEGER_LOWER_BOUND_EXTENSION,
    UNIFRND_INTEGER_SIZE_EXTENSION as INTEGER_SIZE_EXTENSION,
    UNIFRND_INTEGER_UPPER_BOUND_EXTENSION as INTEGER_UPPER_BOUND_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Tensor;
use runmat_value::Value;

use crate::build_runtime_error;
use crate::builtins::common::random;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;
#[path = "unifrnd/output.rs"]
mod output;

const BUILTIN_NAME: &str = "unifrnd";

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

#[runtime_builtin(
    name = "unifrnd",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::unifrnd"
)]
async fn unifrnd_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    enforce_integer_extensions(args.view()).await?;
    let (a, b, shape) = parse_args(args).await?;
    if a >= b {
        return Err(unifrnd_error(
            &UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND,
        ));
    }
    let mut turn = random::acquire_turn(None, "unifrnd").await?;
    output::generate(&mut turn, a, b, &shape).await
}

async fn enforce_integer_extensions(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<()> {
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

async fn parse_args(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<(f64, f64, Vec<usize>)> {
    if args.len() < 2 {
        return Err(unifrnd_error_with(
            &UNIFRND_ERROR_INVALID_ARGUMENT,
            "unifrnd: requires at least two arguments (a, b)",
        ));
    }
    let a = scalar_f64(&args[0]).await?;
    let b = scalar_f64(&args[1]).await?;
    let shape = parse_shape_args(args.view().range(2..)?).await?;
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

async fn parse_shape_args(
    rest: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Vec<usize>> {
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
        let result = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::Num(5.0),
            ]),
        ))
        .expect("unifrnd");
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
        let result = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("unifrnd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[3, 4]);
                assert_eq!(
                    t.materialize_f64(),
                    random::expected_uniform_scaled_sequence(0.0, 10.0, 12)
                );
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
        let result = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("unifrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape(), &[3, 4]),
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
        let result = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(a),
                Value::Tensor(b),
                Value::Tensor(size),
            ]),
        ))
        .expect("unifrnd");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 3]);
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
        let _guard = random::test_guard();
        let _provider_guard = reset_cpu_path();
        let compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
        let bound_error = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Int(runmat_value::IntValue::I16(0)),
                Value::Num(2.0),
            ]),
        ))
        .expect_err("typed lower bound must be gated");
        assert_eq!(
            bound_error.identifier(),
            INTEGER_LOWER_BOUND_EXTENSION.error_identifier
        );
        let size_error = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(2.0),
                Value::Int(runmat_value::IntValue::U8(2)),
            ]),
        ))
        .expect_err("typed size must be gated");
        assert_eq!(
            size_error.identifier(),
            INTEGER_SIZE_EXTENSION.error_identifier
        );
        drop(compatibility);

        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let lossy = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
            ]),
        ))
        .expect_err("lossy upper bound must reject");
        assert!(lossy.message().contains("exactly representable"));
        drop(extensions);
    }

    #[test]
    fn unifrnd_rejects_a_ge_b() {
        let args = vec![Value::Num(5.0), Value::Num(2.0)];
        let err = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect_err("a >= b should error");
        assert_eq!(
            err.identifier(),
            UNIFRND_ERROR_LOWER_BOUND_MUST_BE_LESS_THAN_UPPER_BOUND.identifier
        );
    }

    #[test]
    fn unifrnd_rejects_a_eq_b() {
        let args = vec![Value::Num(3.0), Value::Num(3.0)];
        let err = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect_err("a == b should error");
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
        let result = block_on(unifrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("unifrnd");
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
