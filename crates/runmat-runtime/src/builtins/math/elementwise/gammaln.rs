//! MATLAB-compatible `gammaln` builtin with GPU-aware semantics for RunMat.
//!
//! `gammaln` evaluates the natural logarithm of the gamma function for real,
//! nonnegative inputs. The CPU implementation uses a log-Lanczos form so large
//! arguments do not overflow through `log(gamma(x))`.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinErrorDescriptor, GAMMALN_CHARACTER_INPUT_EXTENSION, GAMMALN_ERROR_DOMAIN,
    GAMMALN_ERROR_INTERNAL, GAMMALN_ERROR_INVALID_INPUT, GAMMALN_ERROR_TOO_MANY_OUTPUTS,
    GAMMALN_INTEGER_INPUT_EXTENSION, GAMMALN_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, NumericDType, NumericScalar, NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::logarithm_common::{
    probe_gpu_lower_bound, GpuDomainProbeError, GpuLowerBoundResult,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "gammaln";
const PI: f64 = std::f64::consts::PI;
const LN_SQRT_TWO_PI: f64 = 0.918_938_533_204_672_7;
const LANCZOS_G: f64 = 7.0;
const SMALL_REFLECTION_CUTOFF: f64 = 1.0e-305;

const LANCZOS_COEFFS: [f64; 8] = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984_369_578_019_572e-6,
    1.5056327351493116e-7,
];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::gammaln")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gammaln",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction { name: "reduce_min" },
        ProviderHook::Unary {
            name: "unary_gammaln",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat uses provider gammaln kernels only after proving gpuArray inputs are nonnegative; otherwise it gathers to enforce MATLAB's real-domain input rule.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::elementwise::gammaln")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gammaln",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion sink because negative inputs must raise a domain error instead of producing an elementwise NaN.",
};

#[runtime_builtin(
    name = "gammaln",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::gammaln"
)]
async fn gammaln_builtin(value: Value) -> BuiltinResult<Value> {
    reject_excess_outputs()?;
    ensure_gammaln_extensions(&value)?;
    match value {
        Value::GpuTensor(handle) => gammaln_gpu(handle).await,
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(error_with_detail(
            &GAMMALN_ERROR_INVALID_INPUT,
            "complex input is not supported",
        )),
        Value::String(_) | Value::StringArray(_) => Err(error_with_detail(
            &GAMMALN_ERROR_INVALID_INPUT,
            "expected real nonnegative numeric input",
        )),
        Value::SparseTensor(_) => Err(error_with_detail(
            &GAMMALN_ERROR_INVALID_INPUT,
            "sparse input is not supported",
        )),
        Value::CharArray(chars) => gammaln_char_array(chars),
        other => gammaln_real(other),
    }
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(error_with_detail(
            &GAMMALN_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    Ok(())
}

fn ensure_gammaln_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAMMALN_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAMMALN_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GAMMALN_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn gammaln_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(error_with_detail(
            &GAMMALN_ERROR_INVALID_INPUT,
            "complex gpuArray input is not supported",
        ));
    }

    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        error_with_detail(
            &GAMMALN_ERROR_INTERNAL,
            "GPU input has no registered owning provider",
        )
    })?;
    let requires_authoritative_host_path = runmat_accelerate_api::handle_integer_type(&handle)
        .is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    if !requires_authoritative_host_path {
        match probe_gpu_lower_bound(provider, &handle, 0.0).await {
            Ok(GpuLowerBoundResult::Below) => {
                return Err(error_with_detail(
                    &GAMMALN_ERROR_DOMAIN,
                    "gpuArray contains negative values",
                ))
            }
            Ok(GpuLowerBoundResult::AtOrAbove) => match provider.unary_gammaln(&handle).await {
                Ok(output) => {
                    if !gpu_helpers::unary_gpu_output_matches(
                        &output,
                        &handle,
                        provider,
                        gpu_helpers::UnaryGpuOutputContract {
                            storage: GpuTensorStorage::Real,
                            precision: runmat_accelerate_api::handle_precision(&handle),
                            integer: None,
                            logical: false,
                            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                        },
                    ) {
                        gpu_helpers::free_rejected_provider_output(&output, &[&handle], provider);
                        return Err(terminal_error(
                            &GAMMALN_ERROR_INTERNAL,
                            "provider unary_gammaln returned malformed output",
                        ));
                    }
                    let mut output = output;
                    runmat_accelerate_api::set_handle_provenance(
                        &mut output,
                        runmat_accelerate_api::handle_provenance(&handle)
                            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
                    );
                    return Ok(gpu_helpers::resident_gpu_value(output));
                }
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                Err(error) => {
                    return Err(terminal_error(
                        &GAMMALN_ERROR_INTERNAL,
                        format!("provider unary_gammaln failed: {error}"),
                    ))
                }
            },
            Ok(GpuLowerBoundResult::Unknown) => {}
            Err(GpuDomainProbeError::ProviderDownload(error)) => return Err(*error),
            Err(error) => {
                return Err(terminal_error(
                    &GAMMALN_ERROR_INTERNAL,
                    format!("provider domain proof failed: {error}"),
                ))
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let output = gammaln_tensor(tensor)?;
    crate::builtins::math::trigonometry::inverse_helpers::upload_value_like_protected(
        provider,
        output,
        BUILTIN_NAME,
        &handle,
        std::slice::from_ref(&handle),
    )
}

fn gammaln_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|detail| error_with_detail(&GAMMALN_ERROR_INVALID_INPUT, detail))?;
    gammaln_tensor(tensor)
}

fn gammaln_tensor(tensor: Tensor) -> BuiltinResult<Value> {
    if let Some(storage) = tensor.integer_storage() {
        for integer in storage.exact_values() {
            if !crate::builtins::common::validation::integer_is_exact_f64(&integer) {
                return Err(error_with_detail(
                    &GAMMALN_ERROR_INVALID_INPUT,
                    "integer values must be exactly representable as double",
                ));
            }
        }
    }
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|detail| error_with_detail(&GAMMALN_ERROR_INTERNAL, detail))?;
    let output = match storage {
        NumericStorage::F32(values) => {
            if values.iter().any(|&value| value < 0.0) {
                return Err(error_with_detail(
                    &GAMMALN_ERROR_DOMAIN,
                    "input values must be nonnegative",
                ));
            }
            NumericStorage::F32(
                values
                    .into_iter()
                    .map(|value| gammaln_nonnegative_scalar(f64::from(value)) as f32)
                    .collect(),
            )
        }
        storage => {
            let values = storage.materialize_f64();
            ensure_nonnegative(&values)?;
            NumericStorage::F64(values.into_iter().map(gammaln_nonnegative_scalar).collect())
        }
    };
    let out = Tensor::from_numeric_storage(output, shape)
        .map_err(|detail| error_with_detail(&GAMMALN_ERROR_INTERNAL, detail))?;
    Ok(gammaln_tensor_into_value(out))
}

fn gammaln_tensor_into_value(tensor: Tensor) -> Value {
    if tensor.len() == 1 && tensor.numeric_dtype() == NumericDType::F64 {
        if let Some(NumericScalar::F64(value)) = tensor.numeric_value_at(0) {
            return Value::Num(value);
        }
    }
    Value::Tensor(tensor)
}

fn gammaln_char_array(chars: CharArray) -> BuiltinResult<Value> {
    let data = chars
        .data
        .iter()
        .map(|&ch| gammaln_nonnegative_scalar(ch as u32 as f64))
        .collect::<Vec<_>>();
    let out = Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|detail| error_with_detail(&GAMMALN_ERROR_INTERNAL, detail))?;
    Ok(gammaln_tensor_into_value(out))
}

pub(crate) fn gammaln_nonnegative_scalar(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    if value == 0.0 || value == f64::INFINITY {
        return f64::INFINITY;
    }
    if value < 0.0 {
        return f64::NAN;
    }
    if value < SMALL_REFLECTION_CUTOFF {
        return -value.ln();
    }
    if value < 0.5 {
        return PI.ln() - (PI * value).sin().ln() - lanczos_gammaln(1.0 - value);
    }
    lanczos_gammaln(value)
}

fn lanczos_gammaln(value: f64) -> f64 {
    let z_minus_one = value - 1.0;
    let mut sum = 0.999_999_999_999_809_9;
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate() {
        sum += coeff / (z_minus_one + (idx + 1) as f64);
    }
    let t = z_minus_one + LANCZOS_G + 0.5;
    LN_SQRT_TWO_PI + (z_minus_one + 0.5) * t.ln() - t + sum.ln()
}

fn ensure_nonnegative(data: &[f64]) -> BuiltinResult<()> {
    if data.iter().any(|&value| value < 0.0) {
        Err(error_with_detail(
            &GAMMALN_ERROR_DOMAIN,
            "input values must be nonnegative",
        ))
    } else {
        Ok(())
    }
}

fn error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn terminal_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail))
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "gammaln/tests.rs"]
mod tests;
