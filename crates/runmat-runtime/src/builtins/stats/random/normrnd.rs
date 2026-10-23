use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    NORMRND_ERROR_INTERNAL, NORMRND_ERROR_INVALID_ARGUMENT,
    NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE, NORMRND_INTEGER_MU_EXTENSION as INTEGER_MU_EXTENSION,
    NORMRND_INTEGER_SIGMA_EXTENSION as INTEGER_SIGMA_EXTENSION,
    NORMRND_INTEGER_SIZE_EXTENSION as INTEGER_SIZE_EXTENSION,
    NORMRND_LOGICAL_INPUT_EXTENSION as LOGICAL_INPUT_EXTENSION,
    NORMRND_RESIDENT_SIZE_EXTENSION as RESIDENT_SIZE_EXTENSION,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::Tensor;
use runmat_value::Value;
use runmat_value::{IntValue, NumericDType};

use crate::build_runtime_error;
use crate::builtins::common::random_args::extract_dims;
use crate::builtins::common::tensor;
use crate::builtins::common::{gpu_helpers, random};
#[path = "normrnd/output.rs"]
mod output;

const BUILTIN_NAME: &str = "normrnd";

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

#[runtime_builtin(
    name = "normrnd",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::normrnd"
)]
async fn normrnd_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    ensure_extensions(args.view())?;
    let output = NormrndOutputPlan::inspect(args.view())?;
    let (mu, sigma, shape) = parse_args(args).await?;
    if sigma < 0.0 {
        return Err(normrnd_error(&NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE));
    }
    let mut turn = random::acquire_turn(None, "normrnd").await?;
    let host = output::generate(&mut turn, mu, sigma, &shape, output.single).await?;
    output.finish(host)
}

struct NormrndOutputPlan {
    single: bool,
    source: Option<runmat_accelerate_api::GpuTensorHandle>,
}

impl NormrndOutputPlan {
    fn inspect(args: crate::call::FunctionArgumentView<'_>) -> crate::BuiltinResult<Self> {
        let data = args.iter().take(2);
        let single = data.clone().any(|value| {
            matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::GpuTensor(handle)
                    if runmat_accelerate_api::handle_integer_type(handle).is_none()
                        && !runmat_accelerate_api::handle_is_logical(handle)
                        && runmat_accelerate_api::handle_precision(handle)
                            == Some(runmat_accelerate_api::ProviderPrecision::F32))
        });
        let source = gpu_helpers::select_resident_output_source(
            args.iter().take(2).filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            BUILTIN_NAME,
        )?;
        Ok(Self { single, source })
    }

    fn finish(&self, host: Value) -> crate::BuiltinResult<Value> {
        match &self.source {
            Some(source) => {
                let restored =
                    gpu_helpers::restore_class_preserving_value(source, host, BUILTIN_NAME)?;
                if runmat_accelerate_api::handle_is_explicit(source)
                    && !matches!(restored, Value::GpuTensor(_))
                {
                    return Err(normrnd_internal_error(
                        "normrnd: provider cannot preserve explicit gpuArray output residency",
                    ));
                }
                Ok(restored)
            }
            None => Ok(host),
        }
    }
}

fn ensure_extensions(args: crate::call::FunctionArgumentView<'_>) -> crate::BuiltinResult<()> {
    if args.first().is_some_and(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_MU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.get(1).is_some_and(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_SIGMA_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.iter().skip(2).any(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.iter().any(is_logical) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.iter().skip(2).any(|value| {
        matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_SIZE_EXTENSION, BUILTIN_NAME)?;
    }
    Ok(())
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

async fn parse_args(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<(f64, f64, Vec<usize>)> {
    if args.len() < 2 {
        return Err(normrnd_error_with(
            &NORMRND_ERROR_INVALID_ARGUMENT,
            "normrnd: requires at least two arguments (mu, sigma)",
        ));
    }
    let mu = scalar_f64(&args[0]).await?;
    let sigma = scalar_f64(&args[1]).await?;
    let shape = parse_shape_args(args.view().range(2..)?).await?;
    Ok((mu, sigma, shape))
}

async fn scalar_f64(value: &Value) -> crate::BuiltinResult<f64> {
    let gathered = crate::gather_if_needed_async(value).await.map_err(|err| {
        normrnd_error_with(&NORMRND_ERROR_INVALID_ARGUMENT, format!("normrnd: {err}"))
    })?;
    let scalar = match &gathered {
        Value::Int(value) => Some(exact_integer_as_f64(value).ok_or_else(|| {
            normrnd_error_with(
                &NORMRND_ERROR_INVALID_ARGUMENT,
                "normrnd: integer parameter must be exactly representable as double",
            )
        })?),
        Value::Tensor(tensor)
            if tensor::is_scalar_tensor(tensor) && tensor.integer_storage().is_some() =>
        {
            let integer = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
                .ok_or_else(|| {
                    normrnd_error_with(
                        &NORMRND_ERROR_INTERNAL,
                        "normrnd: scalar integer storage is inconsistent with its shape",
                    )
                })?;
            Some(exact_integer_as_f64(&integer).ok_or_else(|| {
                normrnd_error_with(
                    &NORMRND_ERROR_INVALID_ARGUMENT,
                    "normrnd: integer parameter must be exactly representable as double",
                )
            })?)
        }
        other => tensor::scalar_f64_from_value_async(other)
            .await
            .map_err(|err| {
                normrnd_error_with(&NORMRND_ERROR_INVALID_ARGUMENT, format!("normrnd: {err}"))
            })?,
    };
    scalar.ok_or_else(|| {
        normrnd_error_with(
            &NORMRND_ERROR_INVALID_ARGUMENT,
            format!("normrnd: expected scalar parameter, got {value:?}"),
        )
    })
}

fn exact_integer_as_f64(value: &IntValue) -> Option<f64> {
    const MAX_EXACT_INTEGER: u64 = 1 << 53;
    match value {
        IntValue::I8(value) => Some(*value as f64),
        IntValue::I16(value) => Some(*value as f64),
        IntValue::I32(value) => Some(*value as f64),
        IntValue::I64(value) if value.unsigned_abs() <= MAX_EXACT_INTEGER => Some(*value as f64),
        IntValue::U8(value) => Some(*value as f64),
        IntValue::U16(value) => Some(*value as f64),
        IntValue::U32(value) => Some(*value as f64),
        IntValue::U64(value) if *value <= MAX_EXACT_INTEGER => Some(*value as f64),
        _ => None,
    }
}

async fn parse_shape_args(
    rest: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Vec<usize>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn reset() -> impl Drop {
        let guard = random::test_guard();
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
        guard
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("integer tensor")
    }

    #[test]
    fn normrnd_scalar_deterministic() {
        let _guard = random::test_guard();
        let _guard = reset();
        let result = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(0.0),
                Value::Num(1.0),
            ]),
        ))
        .expect("normrnd");
        let expected = random::expected_normal_scaled_sequence(0.0, 1.0, 1)[0];
        match result {
            Value::Num(v) => assert!((v - expected).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_matrix_dims() {
        let _guard = random::test_guard();
        let _guard = reset();
        let args = vec![
            Value::Num(5.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let result = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("normrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape(), &[3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_size_vec() {
        let _guard = random::test_guard();
        let _guard = reset();
        let size = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let args = vec![Value::Num(0.0), Value::Num(1.0), Value::Tensor(size)];
        let result = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("normrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape(), &[3, 4]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_reads_typed_integer_parameters_and_size_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let _guard = random::test_guard();
        reset();
        let mu = poisoned_int_tensor(IntegerStorage::I16(vec![5]), vec![1, 1]);
        let sigma = poisoned_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]);
        let size = poisoned_int_tensor(IntegerStorage::U64(vec![2, 3]), vec![1, 2]);
        let result = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(mu),
                Value::Tensor(sigma),
                Value::Tensor(size),
            ]),
        ))
        .expect("normrnd");
        match result {
            Value::Tensor(t) => assert_eq!(t.shape(), &[2, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn normrnd_integer_roles_gate_before_conversion_and_wide_values_reject() {
        let integer = Value::Int(IntValue::I16(0));
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(normrnd_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    integer.clone(),
                    Value::Num(1.0),
                ]),
            ))
            .expect_err("integer mu must gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:NormrndIntegerMuExtension")
            );
        }
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = poisoned_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]);
        let error = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(wide),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("wide integer mu must reject");
        assert!(error.message.contains("exactly representable"));
    }

    #[test]
    fn normrnd_single_data_parameter_selects_single_output() {
        let _guard = random::test_guard();
        reset();
        let mu = Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap());
        let out = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                mu,
                Value::Num(1.0),
                Value::Num(1.0),
                Value::Num(3.0),
            ]),
        ))
        .unwrap();
        let Value::Tensor(out) = out else {
            panic!("expected single tensor output");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        assert_eq!(out.shape(), &[1, 3]);
        let expected = random::expected_normal_scaled_sequence(0.0, 1.0, 3);
        assert_eq!(
            out.materialize_f64(),
            expected
                .iter()
                .map(|value| (*value as f32) as f64)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn normrnd_wgpu_integer_parameter_preserves_class_and_explicit_intent() {
        let _guard = random::test_guard();
        reset();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("wgpu provider");
        let mu = Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &mu).expect("integer upload");
        let out = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(2.0),
            ]),
        ))
        .expect("normrnd");
        let Value::GpuTensor(out) = out else {
            panic!("expected automatic resident double output");
        };
        let host = crate::builtins::common::test_support::gather(Value::GpuTensor(out))
            .expect("gather automatic output");
        assert_eq!(host.numeric_dtype(), NumericDType::F64);
        assert_eq!(host.shape(), &[2, 2]);
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let out = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(handle),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(2.0),
            ]),
        ))
        .expect("explicit resident double output");
        let Value::GpuTensor(out) = out else {
            panic!("expected explicit resident output");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&out));
        assert_eq!(
            runmat_accelerate_api::handle_precision(&out),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
    }

    #[test]
    fn normrnd_rejects_negative_sigma() {
        let args = vec![Value::Num(0.0), Value::Num(-1.0)];
        let err = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect_err("negative sigma should error");
        assert_eq!(
            err.identifier(),
            NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE.identifier
        );
    }

    #[test]
    fn normrnd_distribution_mean_and_std() {
        let _guard = random::test_guard();
        let _guard = reset();
        let mu = 5.0_f64;
        let sigma = 2.0_f64;
        let n = 50_000_usize;
        let args = vec![
            Value::Num(mu),
            Value::Num(sigma),
            Value::Num(n as f64),
            Value::Num(1.0),
        ];
        let result = block_on(normrnd_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("normrnd");
        let data = match result {
            Value::Tensor(t) => t.materialize_f64(),
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
