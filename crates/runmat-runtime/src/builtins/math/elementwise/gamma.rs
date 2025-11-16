//! MATLAB-compatible `gamma` builtin with GPU-aware semantics for RunMat.
//!
//! Provides element-wise evaluation of the gamma function for real and complex inputs while
//! preserving MATLAB broadcasting semantics. GPU execution uses provider hooks when available and
//! falls back to host computation otherwise.

use num_complex::Complex64;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

const PI: f64 = std::f64::consts::PI;
const SQRT_TWO_PI: f64 = 2.506_628_274_631_000_5;
const LANCZOS_G: f64 = 7.0;
const EPSILON: f64 = 1e-12;

const LANCZOS_COEFFS: [f64; 8] = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "gamma"
category: "math/elementwise"
keywords: ["gamma", "factorial", "special function", "elementwise", "gpu", "complex"]
summary: "Element-wise gamma function for scalars, vectors, matrices, and complex inputs."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses the provider unary_gamma hook (Lanczos approximation on WGPU); falls back to host evaluation when the hook is unavailable."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::gamma::tests"
  integration: "builtins::math::elementwise::gamma::tests::gamma_gpu_provider_fallback"
---

# What does the `gamma` function do in MATLAB / RunMat?
`Y = gamma(X)` evaluates the Euler gamma function element-by-element. For positive integers the
result is `(n-1)!`, half-integers map to scaled square-roots of π, and complex arguments follow the
analytic continuation `Γ(z) = ∫_0^∞ t^{z-1} e^{-t} dt`.

## How does the `gamma` function behave in MATLAB / RunMat?
- Respects MATLAB’s broadcasting rules for all dense tensor inputs.
- Promotes logical and integer values to double precision before evaluation; character arrays are
  converted through their Unicode code points.
- Computes complex values with a Lanczos approximation and the reflection identity so results match
  MATLAB for every quadrant.
- Returns `Inf` at non-positive integers, mirroring the poles in the analytic definition.
- Keeps real-valued GPU tensors on device when the provider implements `unary_gamma`; otherwise it
  gathers to the host, evaluates, and reapplies any requested residency via `'like'`.

## `gamma` Function GPU Execution Behaviour
RunMat Accelerate first calls the active provider’s `unary_gamma` hook. With the WGPU backend this
kernel runs entirely on device using a Lanczos approximation. Providers that decline the hook trigger
an automatic gather to the host. After computing the double-precision result, RunMat re-uploads the
tensor when you pass `'like', gpuArray(...)`. Complex outputs always stay on the host because current
providers expose real-valued buffers only.

## Examples of using the `gamma` function in MATLAB / RunMat

### Converting integers to factorials automatically

```matlab
gamma(5)
```

Expected output:

```matlab
ans = 24
```

### Evaluating half-integer inputs

```matlab
gamma(0.5)
```

Expected output:

```matlab
ans = 1.7725
```

### Handling negative non-integers

```matlab
gamma(-0.5)
```

Expected output:

```matlab
ans = -3.5449
```

### Applying gamma element-wise to arrays

```matlab
A = [1 2; 3 4];
B = gamma(A);
```

Expected output:

```matlab
B =
     1     1
     2     6
```

### Working with complex numbers

```matlab
z = 0.5 + 1i;
g = gamma(z);
```

Expected output (rounded):

```matlab
g = 0.8182 - 0.7633i
```

### Using `gamma` with GPU tensors

```matlab
G = gpuArray([0.5 1.5 2.5]);
out = gamma(G);
result = gather(out);
```

Expected output:

```matlab
result = [1.7725 0.8862 1.3293]
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Usually not. Accelerate keeps tensors on the GPU when the provider exposes a `unary_gamma` kernel
(the default WGPU backend does). Otherwise RunMat gathers the tensor, evaluates the gamma function on
the CPU, and uploads the result again only when you explicitly request GPU residency via `'like',
gpuArray(...)`. Complex results remain on the host because today’s GPU providers operate on real
buffers.

## FAQ

### How is `gamma(n)` related to factorials?
For positive integers `n`, `gamma(n) = (n-1)!`. This identity underpins the factorial extension used
throughout probability, statistics, and combinatorics.

### What happens at non-positive integers?
`gamma` has simple poles at `0, -1, -2, ...`. RunMat mirrors MATLAB by returning `Inf` at those
points and signalling the singularity without throwing an error.

### Are half-integers supported exactly?
Yes. Values such as `gamma(0.5) = sqrt(pi)` are computed with a Lanczos approximation that provides
double-precision accuracy consistent with MATLAB.

### Do complex inputs work?
Absolutely. RunMat evaluates the analytic continuation using the reflection identity
`Γ(z) = π / (sin(πz) Γ(1-z))` for `Re(z) < 0.5`, so complex arguments behave the same as in MATLAB.

### Can I keep results on the GPU?
Yes for real-valued outputs whenever the active provider exposes `unary_gamma` (including the WGPU
backend). If the provider lacks the hook, RunMat falls back to the host but re-uploads the tensor when
you request a real-valued GPU prototype via `'like'`. Complex outputs stay on the host for now.

### What about overflow?
Large positive inputs eventually overflow to `Inf`, just like MATLAB. Negative inputs near poles
produce signed infinities consistent with the analytic behaviour.

### Does `gamma` accept `'like'` prototypes?
Yes. Provide `'like', T` to control residency and numeric class. Complex prototypes are honoured on
the host; GPU prototypes must be real.

## See Also
[log](./log), [exp](./exp), [sqrt](./sqrt), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `gamma` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/gamma.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/gamma.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

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
        "Providers may execute gamma directly on device buffers via unary_gamma; runtimes gather to the host when the hook is unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gamma",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently falls back to host evaluation; providers may supply specialised kernels in the future.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("gamma", DOC_MD);

#[runtime_builtin(
    name = "gamma",
    category = "math/elementwise",
    summary = "Element-wise gamma function for scalars, vectors, matrices, or N-D tensors.",
    keywords = "gamma,factorial,special,gpu",
    accel = "unary"
)]
fn gamma_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let output = parse_output_template(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => gamma_gpu(handle)?,
        Value::Complex(re, im) => gamma_complex_scalar_value(Complex64::new(re, im)),
        Value::ComplexTensor(ct) => gamma_complex_tensor(ct)?,
        Value::CharArray(ca) => gamma_char_array(ca)?,
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            gamma_tensor(tensor).map(tensor::tensor_into_value)?
        }
        Value::String(_) | Value::StringArray(_) => {
            return Err("gamma: expected numeric input".to_string())
        }
        Value::Tensor(tensor) => gamma_tensor(tensor).map(tensor::tensor_into_value)?,
        Value::Num(n) => Value::Num(gamma_real_scalar(n)),
        Value::Int(i) => Value::Num(gamma_real_scalar(i.to_f64())),
        Value::Bool(b) => Value::Num(gamma_real_scalar(if b { 1.0 } else { 0.0 })),
        other => {
            return Err(format!(
                "gamma: unsupported input type {:?}; expected numeric or gpuArray input",
                other
            ));
        }
    };
    apply_output_template(base, &output)
}

fn gamma_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_gamma(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    gamma_tensor(tensor).map(tensor::tensor_into_value)
}

fn gamma_tensor(tensor: Tensor) -> Result<Tensor, String> {
    let mut data = Vec::with_capacity(tensor.data.len());
    for &v in &tensor.data {
        data.push(gamma_real_scalar(v));
    }
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("gamma: {e}"))
}

fn gamma_complex_tensor(ct: ComplexTensor) -> Result<Value, String> {
    let mut out = Vec::with_capacity(ct.data.len());
    for &(re, im) in &ct.data {
        let res = gamma_complex_scalar(Complex64::new(re, im));
        out.push((res.re, res.im));
    }
    let tensor = ComplexTensor::new(out, ct.shape.clone()).map_err(|e| format!("gamma: {e}"))?;
    Ok(complex_tensor_into_value(tensor))
}

fn gamma_complex_scalar_value(z: Complex64) -> Value {
    let res = gamma_complex_scalar(z);
    if res.im.abs() <= EPSILON * res.re.abs().max(1.0) {
        Value::Num(res.re)
    } else {
        Value::Complex(res.re, res.im)
    }
}

fn gamma_char_array(ca: CharArray) -> Result<Value, String> {
    let data = ca
        .data
        .iter()
        .map(|&ch| gamma_real_scalar(ch as u32 as f64))
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols]).map_err(|e| format!("gamma: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
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
    let mut sum = Complex64::new(0.999_999_999_999_809_93, 0.0);
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

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> Result<OutputTemplate, String> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err("gamma: expected prototype after 'like'".to_string())
            } else {
                Err("gamma: unrecognised argument for gamma".to_string())
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err("gamma: unsupported option; only 'like' is accepted".to_string())
            }
        }
        _ => Err("gamma: too many input arguments".to_string()),
    }
}

fn apply_output_template(value: Value, template: &OutputTemplate) -> Result<Value, String> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => apply_like_template(value, proto),
    }
}

fn apply_like_template(value: Value, prototype: &Value) -> Result<Value, String> {
    let analysis = analyse_like_prototype(prototype)?;
    match analysis.class {
        PrototypeClass::Real => match analysis.device {
            DevicePreference::Host => convert_to_host_real(value),
            DevicePreference::Gpu => convert_to_gpu_real(value),
        },
        PrototypeClass::Complex => match analysis.device {
            DevicePreference::Host => convert_to_host_complex(value),
            DevicePreference::Gpu => {
                Err("gamma: complex GPU prototypes are not supported yet".to_string())
            }
        },
    }
}

fn convert_to_gpu_real(value: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        "gamma: GPU output requested via 'like' but no acceleration provider is active".to_string()
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => {
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|e| format!("gamma: {e}"))?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("gamma: {e}"))?;
            convert_to_gpu_real(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_to_gpu_real(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu_real(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_gpu_real(Value::Tensor(tensor))
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("gamma: GPU prototypes for 'like' only support real numeric outputs".to_string())
        }
        other => Err(format!(
            "gamma: unsupported result type for GPU output via 'like' ({other:?})"
        )),
    }
}

fn convert_to_host_real(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value(&proxy).map_err(|e| format!("gamma: {e}"))
        }
        other => Ok(other),
    }
}

fn convert_to_host_complex(value: Value) -> Result<Value, String> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(tensor) => {
            let data = tensor.data.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::new(data, tensor.shape.clone())
                .map_err(|e| format!("gamma: {e}"))?;
            Ok(complex_tensor_into_value(complex))
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_tensor(&handle)?;
            convert_to_host_complex(Value::Tensor(gathered))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            convert_to_host_complex(Value::Tensor(tensor))
        }
        Value::Bool(b) => convert_to_host_complex(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::Int(i) => convert_to_host_complex(Value::Num(i.to_f64())),
        other => Err(format!(
            "gamma: cannot convert {other:?} to complex output via 'like'"
        )),
    }
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

#[derive(Clone, Copy)]
enum PrototypeClass {
    Real,
    Complex,
}

struct LikeAnalysis {
    class: PrototypeClass,
    device: DevicePreference,
}

fn analyse_like_prototype(proto: &Value) -> Result<LikeAnalysis, String> {
    match proto {
        Value::GpuTensor(_) => Ok(LikeAnalysis {
            class: PrototypeClass::Real,
            device: DevicePreference::Gpu,
        }),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_) => Ok(LikeAnalysis {
            class: PrototypeClass::Real,
            device: DevicePreference::Host,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(LikeAnalysis {
            class: PrototypeClass::Complex,
            device: DevicePreference::Host,
        }),
        other => {
            let gathered =
                crate::dispatcher::gather_if_needed(other).map_err(|e| format!("gamma: {e}"))?;
            analyse_like_prototype(&gathered)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor};

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a} (tol {tol})");
    }

    #[test]
    fn gamma_positive_integer() {
        match gamma_builtin(Value::Num(5.0), Vec::new()).expect("gamma") {
            Value::Num(v) => approx_eq(v, 24.0, 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn gamma_half_integer() {
        match gamma_builtin(Value::Num(0.5), Vec::new()).expect("gamma") {
            Value::Num(v) => approx_eq(v, PI.sqrt(), 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn gamma_negative_non_integer() {
        match gamma_builtin(Value::Num(-0.5), Vec::new()).expect("gamma") {
            Value::Num(v) => approx_eq(v, -2.0 * PI.sqrt(), 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn gamma_matrix() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        match gamma_builtin(Value::Tensor(tensor), Vec::new()).expect("gamma") {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                approx_eq(t.data[0], 1.0, 1e-12);
                approx_eq(t.data[1], 2.0, 1e-12);
                approx_eq(t.data[2], 1.0, 1e-12);
                approx_eq(t.data[3], 6.0, 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn gamma_pole_returns_inf() {
        match gamma_builtin(Value::Num(0.0), Vec::new()).expect("gamma") {
            Value::Num(v) => assert!(v.is_infinite() && v.is_sign_positive()),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match gamma_builtin(Value::Num(-3.0), Vec::new()).expect("gamma") {
            Value::Num(v) => assert!(v.is_infinite()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn gamma_small_negative_not_infinite() {
        match gamma_builtin(Value::Num(-1.0e-10), Vec::new()).expect("gamma") {
            Value::Num(v) => {
                assert!(v.is_finite(), "expected finite value, got {v}");
                assert!(v.is_sign_negative(), "expected negative value, got {v}");
                assert!(v.abs() > 1.0e9, "expected large magnitude, got {v}");
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn gamma_complex_identity() {
        let z = Complex64::new(0.5, 1.0);
        let gamma_z = gamma_complex_scalar(z);
        let gamma_z_plus_one = gamma_complex_scalar(z + Complex64::new(1.0, 0.0));
        let lhs = gamma_z_plus_one;
        let rhs = z * gamma_z;
        approx_eq(lhs.re, rhs.re, 1e-10);
        approx_eq(lhs.im, rhs.im, 1e-10);
    }

    #[test]
    fn gamma_char_array() {
        let chars = CharArray::new("ab".chars().collect(), 1, 2).unwrap();
        match gamma_builtin(Value::CharArray(chars), Vec::new()).expect("gamma") {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                approx_eq(t.data[0], gamma_real_scalar(97.0), 1e-12);
                approx_eq(t.data[1], gamma_real_scalar(98.0), 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn gamma_string_input_errors() {
        let err = gamma_builtin(Value::from("hello"), Vec::new()).expect_err("expected error");
        assert!(err.contains("expected numeric input"));
    }

    #[test]
    fn gamma_gpu_provider_fallback() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.5, 1.5, 2.5], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = gamma_builtin(Value::GpuTensor(handle), Vec::new()).expect("gamma");
            let gathered = test_support::gather(result).expect("gather");
            approx_eq(gathered.data[0], PI.sqrt(), 1e-12);
            approx_eq(gathered.data[1], 0.8862269254527579, 1e-12);
            approx_eq(gathered.data[2], 1.329340388179137, 1e-12);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gamma_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.5, 1.5, 2.5, -0.5, 4.2, -1.3], vec![3, 2]).expect("tensor");
        let cpu = gamma_tensor(tensor.clone()).expect("cpu gamma");
        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = gamma_gpu(handle).expect("gamma gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-4,
        };
        for (got, expected) in gathered.data.iter().zip(cpu.data.iter()) {
            approx_eq(*got, *expected, tol);
        }
    }

    #[test]
    fn gamma_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = gamma_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("gamma");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    approx_eq(gathered.data[0], 1.0, 1e-12);
                    approx_eq(gathered.data[1], 1.0, 1e-12);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn gamma_like_gpu_complex_result_errors() {
        test_support::with_test_provider(|provider| {
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let err = gamma_builtin(
                Value::Complex(0.5, 0.75),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect_err("expected error");
            assert!(err.contains("only support real numeric outputs"));
        });
    }

    #[test]
    fn gamma_like_complex_requires_host() {
        let result = gamma_builtin(
            Value::Num(2.0),
            vec![Value::from("like"), Value::Complex(1.0, 0.0)],
        )
        .expect("gamma");
        match result {
            Value::Complex(re, im) => {
                approx_eq(re, 1.0, 1e-12);
                approx_eq(im, 0.0, 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn gamma_like_rejects_extra_args() {
        let err = gamma_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("too many input arguments"));
    }

    #[test]
    fn gamma_int_promotes() {
        let value = Value::Int(IntValue::I32(4));
        match gamma_builtin(value, Vec::new()).expect("gamma") {
            Value::Num(v) => approx_eq(v, 6.0, 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn gamma_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
