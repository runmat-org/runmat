//! MATLAB-compatible real gamma-function semantics for RunMat.

use num_complex::Complex64;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, GAMMA_ERROR_INTERNAL, GAMMA_ERROR_INVALID_ARGUMENT,
    GAMMA_ERROR_INVALID_INPUT, GAMMA_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericStorage, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::resident_real_unary;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const PI: f64 = std::f64::consts::PI;
const SQRT_TWO_PI: f64 = 2.506_628_274_631_000_5;
const LANCZOS_G: f64 = 7.0;
const EPSILON: f64 = 1e-12;
const BUILTIN_NAME: &str = "gamma";

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

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::elementwise::gamma_functions::gamma"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gamma",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_gamma" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute gamma directly on real floating device buffers; the runtime gathers to the host when unary_gamma is unavailable.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::gamma_functions::gamma"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gamma",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently falls back to host evaluation; providers may supply specialised kernels.",
};

#[runtime_builtin(
    name = "gamma",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::gamma_functions::gamma"
)]
async fn gamma_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    reject_excess_outputs()?;
    if !rest.is_empty() {
        return Err(gamma_error_with_detail(
            &GAMMA_ERROR_INVALID_ARGUMENT,
            "gamma accepts exactly one input",
        ));
    }
    match value {
        Value::GpuTensor(handle) => gamma_gpu(handle).await,
        Value::Tensor(tensor) => Ok(tensor::tensor_into_value(gamma_tensor(tensor)?)),
        Value::Num(value) => Ok(Value::Num(gamma_real_scalar(value))),
        _ => Err(gamma_error_with_detail(
            &GAMMA_ERROR_INVALID_INPUT,
            "expected real single or double input",
        )),
    }
}

fn reject_excess_outputs() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(gamma_error_with_detail(
            &GAMMA_ERROR_TOO_MANY_OUTPUTS,
            "only one output is defined",
        ));
    }
    Ok(())
}

async fn gamma_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if resident_real_unary::validate_input(&handle).is_err() {
        return Err(gamma_error_with_detail(
            &GAMMA_ERROR_INVALID_INPUT,
            "expected real single or double gpuArray input",
        ));
    }
    let provider = resident_real_unary::exact_owner(&handle).ok_or_else(|| {
        gamma_error_with_detail(&GAMMA_ERROR_INTERNAL, "GPU input has no owning provider")
    })?;
    match provider.unary_gamma(&handle).await {
        Ok(mut output) => {
            if !resident_real_unary::output_matches(&output, &handle, provider) {
                resident_real_unary::reject_output(&output, &handle, provider);
                return Err(gamma_terminal_error(
                    &GAMMA_ERROR_INTERNAL,
                    "provider unary_gamma returned malformed output",
                ));
            }
            resident_real_unary::preserve_residency_intent(&mut output, &handle);
            return Ok(gpu_helpers::resident_gpu_value(output));
        }
        Err(error) if resident_real_unary::hook_is_unsupported(&error) => {}
        Err(error) => {
            return Err(gamma_terminal_error(&GAMMA_ERROR_INTERNAL, error));
        }
    }
    let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
    let output = gamma_tensor(gathered)?;
    let output = resident_real_unary::restore_fallback(&output, &handle, provider)
        .map_err(|detail| gamma_terminal_error(&GAMMA_ERROR_INTERNAL, detail))?;
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn gamma_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|detail| gamma_error_with_detail(&GAMMA_ERROR_INTERNAL, detail))?;
    let output = match storage {
        NumericStorage::F64(values) => {
            NumericStorage::F64(values.into_iter().map(gamma_real_scalar).collect())
        }
        NumericStorage::F32(values) => NumericStorage::F32(
            values
                .into_iter()
                .map(|value| gamma_real_scalar(f64::from(value)) as f32)
                .collect(),
        ),
        _ => {
            return Err(gamma_error_with_detail(
                &GAMMA_ERROR_INVALID_INPUT,
                "expected real single or double input",
            ))
        }
    };
    Tensor::from_numeric_storage(output, shape)
        .map_err(|detail| gamma_error_with_detail(&GAMMA_ERROR_INTERNAL, detail))
}

fn gamma_real_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if is_non_positive_integer(x) {
        return f64::INFINITY;
    }
    // Form large positive real results through log-gamma so the Lanczos power
    // and exponential factors cannot overflow separately before their product.
    if x >= 128.0 {
        return super::gammaln::gammaln_nonnegative_scalar(x).exp();
    }
    let result = gamma_complex_scalar(Complex64::new(x, 0.0));
    if result.im.abs() <= EPSILON * result.re.abs().max(1.0) {
        result.re
    } else {
        f64::NAN
    }
}

fn gamma_complex_scalar(z: Complex64) -> Complex64 {
    if z.re.is_nan() || z.im.is_nan() {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    if z.im.abs() <= EPSILON && z.re.is_infinite() {
        return Complex64::new(f64::INFINITY, 0.0);
    }
    if is_complex_pole(z) {
        return Complex64::new(f64::INFINITY, 0.0);
    }
    if z.re < 0.5 {
        let sin_term = (Complex64::new(PI, 0.0) * z).sin();
        if sin_term.norm_sqr() <= EPSILON * EPSILON {
            return Complex64::new(f64::INFINITY, 0.0);
        }
        let gamma_one_minus_z = gamma_complex_scalar(Complex64::new(1.0, 0.0) - z);
        return Complex64::new(PI, 0.0) / (sin_term * gamma_one_minus_z);
    }
    lanczos_gamma(z)
}

fn lanczos_gamma(z: Complex64) -> Complex64 {
    let z_minus_one = z - Complex64::new(1.0, 0.0);
    let mut sum = Complex64::new(0.999_999_999_999_809_9, 0.0);
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate() {
        let denom = z_minus_one + Complex64::new((idx + 1) as f64, 0.0);
        sum += Complex64::new(*coeff, 0.0) / denom;
    }
    let t = z_minus_one + Complex64::new(LANCZOS_G + 0.5, 0.0);
    let power = t.powc(z_minus_one + Complex64::new(0.5, 0.0));
    let exponential = (-t).exp();
    Complex64::new(SQRT_TWO_PI, 0.0) * power * exponential * sum
}

fn is_non_positive_integer(x: f64) -> bool {
    x <= 0.0 && is_close_to_integer(x)
}

fn is_complex_pole(z: Complex64) -> bool {
    z.im.abs() <= EPSILON && is_non_positive_integer(z.re)
}

fn is_close_to_integer(x: f64) -> bool {
    if !x.is_finite() {
        return false;
    }
    let nearest = x.round();
    let diff = (x - nearest).abs();
    if nearest == 0.0 {
        diff <= EPSILON * EPSILON
    } else {
        diff <= EPSILON * nearest.abs().max(1.0)
    }
}

fn gamma_error_with_detail(
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

fn gamma_terminal_error(
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
mod tests;
