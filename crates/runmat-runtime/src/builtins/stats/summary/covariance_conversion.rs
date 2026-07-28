//! Convert covariance matrices to correlation matrices.

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderCovarianceToCorrelationResult,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Correlation matrix corresponding to the covariance matrix.",
};

const OUTPUT_SIGMA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sigma",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Column vector of standard deviations computed from diag(C).",
};

const OUTPUTS_R: [BuiltinParamDescriptor; 1] = [OUTPUT_R];
const OUTPUTS_R_SIGMA: [BuiltinParamDescriptor; 2] = [OUTPUT_R, OUTPUT_SIGMA];

const INPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square covariance matrix.",
}];

const CORRCOV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "R = corrcov(C)",
        inputs: &INPUT_C,
        outputs: &OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, sigma] = corrcov(C)",
        inputs: &INPUT_C,
        outputs: &OUTPUTS_R_SIGMA,
    },
];

const COV2CORR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "R = cov2corr(C)",
        inputs: &INPUT_C,
        outputs: &OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, sigma] = cov2corr(C)",
        inputs: &INPUT_C,
        outputs: &OUTPUTS_R_SIGMA,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COVARIANCE_CONVERSION.INVALID_ARGUMENT",
    identifier: None,
    when: "Input is nonnumeric, complex, not single/double precision, non-square, has negative variances, violates covariance bounds, or too many arguments/outputs are supplied.",
    message: "covariance conversion: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COVARIANCE_CONVERSION.INTERNAL",
    identifier: None,
    when: "Internal tensor allocation fails.",
    message: "covariance conversion: internal error",
};

macro_rules! conversion_descriptor {
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
            output_mode: BuiltinOutputMode::ByRequestedOutputCount,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

fn corrcov_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn conversion_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:InvalidArgument"))
        .build()
}

fn internal_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:Internal"))
        .build()
}

async fn covariance_tensor(
    name: &'static str,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Tensor> {
    if !rest.is_empty() {
        return Err(conversion_error(
            name,
            format!("{name}: expected exactly one input argument"),
        ));
    }
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| conversion_error(name, format!("{name}: unable to gather input: {err}")))?;
    match gathered {
        Value::Tensor(tensor) if matches!(tensor.dtype, NumericDType::F64 | NumericDType::F32) => {
            Ok(tensor)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            tensor::integer_tensor_to_f64(tensor)
                .map_err(|err| conversion_error(name, format!("{name}: {err}")))
        }
        Value::Tensor(tensor) => Err(conversion_error(
            name,
            format!(
                "{name}: covariance matrix must be numeric real, got {}",
                tensor.dtype.class_name()
            ),
        )),
        Value::LogicalArray(logical) => {
            let data = logical.data.into_iter().map(f64::from).collect::<Vec<_>>();
            Tensor::new(data, logical.shape)
                .map_err(|err| internal_error(name, format!("{name}: {err}")))
        }
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| internal_error(name, format!("{name}: {err}"))),
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1])
            .map_err(|err| internal_error(name, format!("{name}: {err}"))),
        Value::Bool(value) => Tensor::new(vec![if value { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|err| internal_error(name, format!("{name}: {err}"))),
        Value::Complex(..) | Value::ComplexTensor(_) => Err(conversion_error(
            name,
            format!("{name}: complex covariance matrices are not supported"),
        )),
        other => Err(conversion_error(
            name,
            format!("{name}: expected a numeric covariance matrix, got {other:?}"),
        )),
    }
}

fn covariance_to_correlation(
    name: &'static str,
    covariance: Tensor,
) -> BuiltinResult<(Value, Value)> {
    validate_covariance_matrix(name, &covariance)?;
    let n = covariance.rows;
    let covariance_values = tensor::tensor_values_f64_cow(&covariance);
    let mut sigma = Vec::with_capacity(n);
    for idx in 0..n {
        sigma.push(covariance_values[idx + idx * n].sqrt());
    }

    let mut r = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            let denom = sigma[row] * sigma[col];
            let idx = row + col * n;
            r[idx] = if denom == 0.0 {
                f64::NAN
            } else {
                covariance_values[idx] / denom
            };
        }
    }

    let r = Tensor::new_with_dtype(r, vec![n, n], covariance.dtype)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_error(name, format!("{name}: {err}")))?;
    let sigma = Tensor::new_with_dtype(sigma, vec![n, 1], covariance.dtype)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_error(name, format!("{name}: {err}")))?;
    Ok((r, sigma))
}

fn validate_covariance_matrix(name: &'static str, tensor: &Tensor) -> BuiltinResult<()> {
    if tensor.shape.len() > 2 {
        return Err(conversion_error(
            name,
            format!("{name}: covariance matrix must be two-dimensional"),
        ));
    }
    if tensor.rows != tensor.cols {
        return Err(conversion_error(
            name,
            format!("{name}: covariance matrix must be square"),
        ));
    }
    let values = tensor::tensor_values_f64_cow(tensor);
    for &value in values.iter() {
        if value.is_nan() {
            continue;
        }
        if !value.is_finite() {
            return Err(conversion_error(
                name,
                format!("{name}: covariance matrix must contain finite values or NaN"),
            ));
        }
    }
    for idx in 0..tensor.rows {
        let variance = values[idx + idx * tensor.rows];
        if variance < 0.0 {
            return Err(conversion_error(
                name,
                format!("{name}: covariance matrix diagonal entries must be nonnegative"),
            ));
        }
    }
    for col in 0..tensor.cols {
        for row in 0..col {
            let a = values[row + col * tensor.rows];
            let b = values[col + row * tensor.rows];
            if a.is_nan() && b.is_nan() {
                continue;
            }
            if a.is_nan() || b.is_nan() {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance matrix must be symmetric"),
                ));
            }
            let tol = 1.0e-10 * a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > tol {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance matrix must be symmetric"),
                ));
            }
            let variance_row = values[row + row * tensor.rows];
            let variance_col = values[col + col * tensor.rows];
            if variance_row.is_nan() || variance_col.is_nan() {
                continue;
            }
            let max_covariance = (variance_row * variance_col).sqrt();
            let bound_tol = 1.0e-10 * max_covariance.max(a.abs()).max(1.0);
            if a.abs() > max_covariance + bound_tol {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance magnitude exceeds variance bounds"),
                ));
            }
        }
    }
    Ok(())
}

fn output_values(name: &'static str, r: Value, sigma: Value) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![r])),
        Some(2) => Ok(Value::OutputList(vec![r, sigma])),
        Some(_) => Err(conversion_error(
            name,
            format!("{name}: too many output arguments; maximum is 2"),
        )),
        None => Ok(r),
    }
}

struct CovarianceGpuEval {
    provider: &'static dyn AccelProvider,
    correlation: GpuTensorHandle,
    sigma: GpuTensorHandle,
}

impl CovarianceGpuEval {
    fn from_provider(
        provider: &'static dyn AccelProvider,
        result: ProviderCovarianceToCorrelationResult,
    ) -> Self {
        Self {
            provider,
            correlation: result.correlation,
            sigma: result.sigma,
        }
    }

    fn output(self, name: &'static str) -> BuiltinResult<Value> {
        match crate::output_count::current_output_count() {
            Some(0) => {
                let _ = self.provider.free(&self.correlation);
                let _ = self.provider.free(&self.sigma);
                Ok(Value::OutputList(Vec::new()))
            }
            Some(1) => {
                let _ = self.provider.free(&self.sigma);
                Ok(Value::OutputList(vec![gpu_helpers::resident_gpu_value(
                    self.correlation,
                )]))
            }
            Some(2) => Ok(Value::OutputList(vec![
                gpu_helpers::resident_gpu_value(self.correlation),
                gpu_helpers::resident_gpu_value(self.sigma),
            ])),
            Some(_) => {
                let _ = self.provider.free(&self.correlation);
                let _ = self.provider.free(&self.sigma);
                Err(conversion_error(
                    name,
                    format!("{name}: too many output arguments; maximum is 2"),
                ))
            }
            None => {
                let _ = self.provider.free(&self.sigma);
                Ok(gpu_helpers::resident_gpu_value(self.correlation))
            }
        }
    }
}

fn provider_is_unsupported(err: &anyhow::Error) -> bool {
    let message = err.to_string();
    message.contains("not supported") || message.contains("unsupported")
}

fn try_covariance_gpu(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<Option<CovarianceGpuEval>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(conversion_error(
            name,
            format!("{name}: complex covariance matrices are not supported"),
        ));
    }
    let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) else {
        return Ok(None);
    };
    match provider.covariance_to_correlation(handle) {
        Ok(result) => Ok(Some(CovarianceGpuEval::from_provider(provider, result))),
        Err(err) if provider_is_unsupported(&err) => Ok(None),
        Err(err) => Err(conversion_error(name, format!("{name}: {err}"))),
    }
}

async fn covariance_conversion_builtin(
    name: &'static str,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(conversion_error(
            name,
            format!("{name}: expected exactly one input argument"),
        ));
    }
    if matches!(crate::output_count::current_output_count(), Some(0)) {
        return Ok(Value::OutputList(Vec::new()));
    }
    if let Some(eval) = try_covariance_gpu(name, &value)? {
        return eval.output(name);
    }
    let covariance = covariance_tensor(name, value, Vec::new()).await?;
    let (r, sigma) = covariance_to_correlation(name, covariance)?;
    output_values(name, r, sigma)
}

pub mod corrcov {
    use super::*;

    conversion_descriptor!("corrcov", CORRCOV_SIGNATURES);

    #[runmat_macros::register_gpu_spec(
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::corrcov"
    )]
    pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
        name: "corrcov",
        op_kind: GpuOpKind::Custom("summary-stats"),
        supported_precisions: &[ScalarType::F32, ScalarType::F64],
        broadcast: BroadcastSemantics::None,
        provider_hooks: &[ProviderHook::Custom("covariance_to_correlation")],
        constant_strategy: ConstantStrategy::InlineLiteral,
        residency: ResidencyPolicy::NewHandle,
        nan_mode: ReductionNaN::Include,
        two_pass_threshold: None,
        workgroup_size: None,
        accepts_nan_mode: false,
        notes: "Resident real gpuArray covariance matrices use the provider covariance_to_correlation hook and return resident correlation/sigma outputs; unsupported providers fall back to the host reference path.",
    };

    #[runmat_macros::register_fusion_spec(
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::corrcov"
    )]
    pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
        name: "corrcov",
        shape: ShapeRequirements::Any,
        constant_strategy: ConstantStrategy::InlineLiteral,
        elementwise: None,
        reduction: None,
        emits_nan: true,
        notes: "corrcov is a matrix-normalization operation and remains a fusion boundary.",
    };

    #[runtime_builtin(
        name = "corrcov",
        category = "stats/summary",
        summary = "Convert a covariance matrix to a correlation matrix.",
        keywords = "corrcov,covariance,correlation,standard deviation,statistics",
        accel = "sink",
        type_resolver(super::corrcov_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::corrcov"
    )]
    pub(crate) async fn corrcov_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        covariance_conversion_builtin("corrcov", value, rest).await
    }
}

pub mod cov2corr {
    use super::*;

    conversion_descriptor!("cov2corr", COV2CORR_SIGNATURES);

    #[runmat_macros::register_gpu_spec(
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::cov2corr"
    )]
    pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
        name: "cov2corr",
        op_kind: GpuOpKind::Custom("summary-stats"),
        supported_precisions: &[ScalarType::F32, ScalarType::F64],
        broadcast: BroadcastSemantics::None,
        provider_hooks: &[ProviderHook::Custom("covariance_to_correlation")],
        constant_strategy: ConstantStrategy::InlineLiteral,
        residency: ResidencyPolicy::NewHandle,
        nan_mode: ReductionNaN::Include,
        two_pass_threshold: None,
        workgroup_size: None,
        accepts_nan_mode: false,
        notes: "Compatibility alias for corrcov; resident real gpuArray covariance matrices use the provider covariance_to_correlation hook.",
    };

    #[runmat_macros::register_fusion_spec(
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::cov2corr"
    )]
    pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
        name: "cov2corr",
        shape: ShapeRequirements::Any,
        constant_strategy: ConstantStrategy::InlineLiteral,
        elementwise: None,
        reduction: None,
        emits_nan: true,
        notes: "cov2corr is a matrix-normalization operation and remains a fusion boundary.",
    };

    #[runtime_builtin(
        name = "cov2corr",
        category = "stats/summary",
        summary = "Convert a covariance matrix to a correlation matrix.",
        keywords = "cov2corr,covariance,correlation,standard deviation,statistics",
        accel = "sink",
        type_resolver(super::corrcov_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::cov2corr"
    )]
    pub(crate) async fn cov2corr_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        covariance_conversion_builtin("cov2corr", value, rest).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "expected {expected}, got {actual}"
        );
    }

    fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let mut tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        tensor.data.clear();
        Value::Tensor(tensor)
    }

    #[test]
    fn corrcov_converts_covariance_and_returns_sigma() {
        let c = Value::Tensor(Tensor::new(vec![4.0, 2.0, 2.0, 9.0], vec![2, 2]).expect("tensor"));
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(corrcov::corrcov_builtin(c, Vec::new())).expect("corrcov");
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.shape, vec![2, 2]);
                        assert_close(tensor.data[0], 1.0);
                        assert_close(tensor.data[1], 1.0 / 3.0);
                        assert_close(tensor.data[2], 1.0 / 3.0);
                        assert_close(tensor.data[3], 1.0);
                    }
                    other => panic!("expected correlation tensor, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.shape, vec![2, 1]);
                        assert_close(tensor.data[0], 2.0);
                        assert_close(tensor.data[1], 3.0);
                    }
                    other => panic!("expected sigma tensor, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn corrcov_gpu_input_returns_resident_outputs() {
        test_support::with_test_provider(|provider| {
            let covariance = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[4.0, 2.0, 2.0, 9.0],
                    shape: &[2, 2],
                })
                .expect("upload covariance");
            provider.reset_telemetry();

            let _guard = crate::output_count::push_output_count(Some(2));
            let out = block_on(corrcov::corrcov_builtin(
                Value::GpuTensor(covariance),
                Vec::new(),
            ))
            .expect("corrcov gpu");
            let Value::OutputList(values) = out else {
                panic!("expected output list");
            };

            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.upload_bytes, 0);
            assert_eq!(telemetry.download_bytes, 0);
            assert!(matches!(values[0], Value::GpuTensor(_)));
            assert!(matches!(values[1], Value::GpuTensor(_)));

            let correlation = test_support::gather(values[0].clone()).expect("correlation");
            let sigma = test_support::gather(values[1].clone()).expect("sigma");
            assert_eq!(correlation.shape, vec![2, 2]);
            assert_eq!(sigma.shape, vec![2, 1]);
            assert_close(correlation.data[0], 1.0);
            assert_close(correlation.data[1], 1.0 / 3.0);
            assert_close(correlation.data[2], 1.0 / 3.0);
            assert_close(correlation.data[3], 1.0);
            assert_close(sigma.data[0], 2.0);
            assert_close(sigma.data[1], 3.0);
        });
    }

    #[test]
    fn cov2corr_gpu_input_preserves_invalid_covariance_error() {
        test_support::with_test_provider(|provider| {
            let covariance = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0, 0.1, 0.2, 1.0],
                    shape: &[2, 2],
                })
                .expect("upload covariance");

            let err = block_on(cov2corr::cov2corr_builtin(
                Value::GpuTensor(covariance),
                Vec::new(),
            ))
            .expect_err("invalid covariance");

            assert_eq!(err.identifier(), Some("RunMat:cov2corr:InvalidArgument"));
            assert!(err.message().contains("symmetric"));
        });
    }

    #[test]
    fn cov2corr_alias_matches_corrcov() {
        let c = Value::Tensor(Tensor::new(vec![16.0, 4.0, 4.0, 25.0], vec![2, 2]).expect("tensor"));
        let out = block_on(cov2corr::cov2corr_builtin(c, Vec::new())).expect("cov2corr");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert_close(tensor.data[0], 1.0);
                assert_close(tensor.data[1], 0.2);
                assert_close(tensor.data[2], 0.2);
                assert_close(tensor.data[3], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn corrcov_zero_variance_produces_nan_correlations() {
        let c = Value::Tensor(Tensor::new(vec![0.0, 0.0, 0.0, 9.0], vec![2, 2]).expect("tensor"));
        let out = block_on(corrcov::corrcov_builtin(c, Vec::new())).expect("corrcov");
        match out {
            Value::Tensor(tensor) => {
                assert!(tensor.data[0].is_nan());
                assert!(tensor.data[1].is_nan());
                assert!(tensor.data[2].is_nan());
                assert_close(tensor.data[3], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn corrcov_rejects_invalid_covariance_matrix() {
        let nonsquare = Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).expect("tensor"),
        );
        let err = block_on(corrcov::corrcov_builtin(nonsquare, Vec::new())).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:corrcov:InvalidArgument"));
        assert!(err.message().contains("square"));

        let negative =
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, -1.0], vec![2, 2]).expect("tensor"));
        let err = block_on(corrcov::corrcov_builtin(negative, Vec::new())).unwrap_err();
        assert!(err.message().contains("nonnegative"));

        let asymmetric =
            Value::Tensor(Tensor::new(vec![1.0, 0.1, 0.2, 1.0], vec![2, 2]).expect("tensor"));
        let err = block_on(corrcov::corrcov_builtin(asymmetric, Vec::new())).unwrap_err();
        assert!(err.message().contains("symmetric"));

        let one_sided_nan =
            Value::Tensor(Tensor::new(vec![1.0, 2.0, f64::NAN, 1.0], vec![2, 2]).expect("tensor"));
        let err = block_on(corrcov::corrcov_builtin(one_sided_nan, Vec::new())).unwrap_err();
        assert!(err.message().contains("symmetric"));

        let impossible =
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).expect("tensor"));
        let err = block_on(corrcov::corrcov_builtin(impossible, Vec::new())).unwrap_err();
        assert!(err.message().contains("variance bounds"));
    }

    #[test]
    fn corrcov_accepts_typed_integer_tensors_and_scalar_logical_extensions() {
        {
            let integer = mirrorless_int_tensor(IntegerStorage::U16(vec![4, 2, 2, 9]), vec![2, 2]);
            let _guard = crate::output_count::push_output_count(Some(2));
            let out = block_on(corrcov::corrcov_builtin(integer, Vec::new())).unwrap();
            match out {
                Value::OutputList(values) => {
                    assert_eq!(values.len(), 2);
                    match &values[0] {
                        Value::Tensor(tensor) => {
                            assert_eq!(tensor.dtype, NumericDType::F64);
                            assert_eq!(tensor.shape, vec![2, 2]);
                            assert_close(tensor.data[0], 1.0);
                            assert_close(tensor.data[1], 1.0 / 3.0);
                            assert_close(tensor.data[2], 1.0 / 3.0);
                            assert_close(tensor.data[3], 1.0);
                        }
                        other => panic!("expected correlation tensor, got {other:?}"),
                    }
                    match &values[1] {
                        Value::Tensor(tensor) => {
                            assert_eq!(tensor.dtype, NumericDType::F64);
                            assert_eq!(tensor.shape, vec![2, 1]);
                            assert_close(tensor.data[0], 2.0);
                            assert_close(tensor.data[1], 3.0);
                        }
                        other => panic!("expected sigma tensor, got {other:?}"),
                    }
                }
                other => panic!("expected output list, got {other:?}"),
            }
        }

        let scalar = block_on(corrcov::corrcov_builtin(Value::Num(4.0), Vec::new())).unwrap();
        match scalar {
            Value::Num(value) => assert_close(value, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }

        let logical = Value::LogicalArray(
            runmat_builtins::LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical"),
        );
        let out = block_on(corrcov::corrcov_builtin(logical, Vec::new())).unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.dtype, NumericDType::F64);
                assert_eq!(tensor.shape, vec![2, 2]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn corrcov_rejects_too_many_outputs() {
        let c = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("tensor"));
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = block_on(corrcov::corrcov_builtin(c, Vec::new())).unwrap_err();
        assert!(err.message().contains("too many output"));
    }
}
