//! Convert covariance matrices to correlation matrices.

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderCovarianceToCorrelationResult,
};
use runmat_builtins::{
    CORRCOV_INTEGER_DATA_EXTENSION, CORRCOV_LOGICAL_DATA_EXTENSION, COV2CORR_GPU_INPUT_EXTENSION,
    COV2CORR_INTEGER_DATA_EXTENSION, COV2CORR_LOGICAL_DATA_EXTENSION,
    COV2CORR_SINGLE_DATA_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[derive(Clone, Copy)]
enum ConversionKind {
    Corrcov,
    Cov2corr,
}
mod admission;
mod computation;
mod matrix_validation;
mod output;
mod provider;
mod validation;
use admission::{conversion_error, ensure_conversion_extensions, internal_error};
use computation::covariance_to_correlation;
use matrix_validation::validate_covariance_matrix;
use output::output_values;
use provider::try_covariance_gpu;
use validation::validate_integer_covariance_structure;

async fn covariance_tensor(
    name: &'static str,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Tensor> {
    if !rest.is_empty() {
        return Err(conversion_error(
            name,
            format!("{name}: expected exactly one input argument"),
        ));
    }
    let gathered = gather_if_needed_async(&value).await.map_err(|err| {
        crate::runtime_error::with_provider_execution_source(
            conversion_error(name, format!("{name}: unable to gather input: {err}")),
            runmat_accelerate_api::ProviderExecutionError::from_source(err),
        )
    })?;
    match gathered {
        Value::Tensor(tensor)
            if matches!(
                tensor.numeric_dtype(),
                NumericDType::F64 | NumericDType::F32
            ) =>
        {
            Ok(tensor)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            validate_integer_covariance_structure(name, &tensor)?;
            tensor::integer_tensor_to_f64(tensor)
                .map_err(|err| conversion_error(name, format!("{name}: {err}")))
        }
        Value::Tensor(tensor) => Err(conversion_error(
            name,
            format!(
                "{name}: covariance matrix must be numeric real, got {}",
                tensor.numeric_dtype().class_name()
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

mod resident;
use resident::CovarianceGpuEval;

async fn covariance_conversion_builtin(
    kind: ConversionKind,
    name: &'static str,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(conversion_error(
            name,
            format!("{name}: expected exactly one input argument"),
        ));
    }
    ensure_conversion_extensions(kind, &value)?;
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(conversion_error(
            name,
            format!("{name}: too many output arguments; maximum is 2"),
        ));
    }
    if matches!(crate::output_count::current_output_count(), Some(0)) {
        return Ok(Value::output_list(Vec::new()));
    }
    if let Some(eval) = try_covariance_gpu(name, &value)? {
        return eval.output(kind, name);
    }
    let covariance = covariance_tensor(name, value, rest).await?;
    let (r, sigma) = covariance_to_correlation(name, covariance)?;
    output_values(kind, name, r, sigma)
}

pub mod corrcov {
    use super::*;

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
        binding_variant = "default",
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::corrcov"
    )]
    pub(crate) async fn corrcov_builtin(
        value: Value,
        rest: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        covariance_conversion_builtin(ConversionKind::Corrcov, "corrcov", value, rest).await
    }
}

pub mod cov2corr {
    use super::*;

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
        notes: "RunMat-only resident covariance inputs use the provider covariance_to_correlation hook; Finance Toolbox compatibility documents host double input.",
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
        binding_variant = "default",
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::cov2corr"
    )]
    pub(crate) async fn cov2corr_builtin(
        value: Value,
        rest: crate::call::BuiltinArguments,
    ) -> BuiltinResult<Value> {
        covariance_conversion_builtin(ConversionKind::Cov2corr, "cov2corr", value, rest).await
    }
}

#[cfg(test)]
mod tests;
