//! MATLAB-compatible `polyder` builtin with GPU-aware semantics for RunMat.

use log::trace;
use num_complex::Complex64;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{tensor, tensor::tensor_into_value};
use crate::dispatcher;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const EPS: f64 = 1.0e-12;
const BUILTIN_NAME: &str = "polyder";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "polyder",
        builtin_path = "crate::builtins::math::poly::polyder"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "polyder"
category: "math/poly"
keywords: ["polyder", "polynomial derivative", "product rule", "quotient rule", "gpu"]
summary: "Differentiate polynomials, products, and ratios with MATLAB-compatible coefficient vectors."
references:
  - title: "MATLAB polyder documentation"
    url: "https://www.mathworks.com/help/matlab/ref/polyder.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "When the active provider implements the polyder hooks, differentiation runs entirely on the GPU; otherwise coefficients are gathered back to the host. Complex coefficients continue to fall back to the CPU."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::poly::polyder::tests"
  integration: "builtins::math::poly::polyder::tests::{gpu_inputs_remain_on_device,gpu_product_matches_cpu,gpu_quotient_matches_cpu,wgpu_polyder_single_matches_cpu,wgpu_polyder_product_matches_cpu,wgpu_polyder_quotient_matches_cpu}"
  vm: "ignition::vm polyder multi-output dispatch"
  doc: "builtins::math::poly::polyder::tests::doc_examples_present"
---

# What does the `polyder` function do in MATLAB / RunMat?
`polyder` differentiates polynomials represented by their coefficient vectors. The coefficients
follow MATLAB’s convention: the first element corresponds to the highest power of `x`. The builtin
supports three related operations:

1. `polyder(p)` returns the derivative of a polynomial `p`.
2. `polyder(p, a)` applies the product rule to the convolution `conv(p, a)`.
3. `[num, den] = polyder(u, v)` returns the derivative of a rational function `u(x) / v(x)`
   using the quotient rule, yielding the numerator `num` and denominator `den`.

## How does the `polyder` function behave in MATLAB / RunMat?
- Accepts real or complex scalars, row vectors, column vectors, or empty vectors. Inputs with more
  than one non-singleton dimension raise MATLAB-compatible errors.
- Logical and integer coefficients are promoted to double precision before differentiation.
- Leading zeros are removed in outputs (unless the polynomial is identically zero, in which case a
  single zero coefficient is returned).
- The orientation of the first input polynomial is preserved for the derivative of a single
  polynomial or a product; the denominator in the quotient rule preserves the orientation of `v`.
- Calling `polyder(p, a)` with a single output applies the product rule. Capturing two outputs
  alongside two input polynomials returns the quotient-rule numerator and denominator
  (`u' * v - u * v'`, `v * v`).
- Empty inputs are treated as the zero polynomial.
- When inputs live on the GPU (e.g., `gpuArray`) and the active provider exposes the polyder
  hooks, differentiation runs entirely on-device and returns trimmed GPU tensors. Providers that
  lack the hooks fall back to gathering coefficients to the host, executing the reference CPU
  implementation, and returning host-resident results. Explicit calls to `gpuArray` remain
  available if you need to force residency.

## `polyder` Function GPU Execution Behaviour
When a provider advertises the `polyder` hooks (the in-process and WGPU backends both do), single
polynomials and the product/quotient forms execute fully on the GPU. Outputs preserve the
orientation of the leading input while trimming leading zeros exactly as the CPU path does. If the
provider declines the request—because the coefficients are complex or the backend lacks support—
RunMat automatically gathers the inputs and falls back to the CPU reference implementation to
preserve MATLAB compatibility.

## Examples of using the `polyder` function in MATLAB / RunMat

### Differentiating a cubic polynomial

```matlab
p = [3 -2 5 7];   % 3x^3 - 2x^2 + 5x + 7
dp = polyder(p);
```

Expected output:

```matlab
dp = [9 -4 5];
```

### Applying the product rule

```matlab
p = [1 0 -2];    % x^2 - 2
a = [1 1];       % x + 1
dp = polyder(p, a);   % derivative of conv(p, a)
```

Expected output:

```matlab
dp = [3 2 -2];
```

### Differentiating a rational function

```matlab
u = [1 0 -4];      % x^2 - 4
v = [1 -1];        % x - 1
[num, den] = polyder(u, v);    % derivative of (u / v)
```

Expected output:

```matlab
num = [1 -2 4];
den = [1 -2 1];
```

### Preserving column-vector orientation

```matlab
p = [1; 0; -3];    % column vector coefficients
dp = polyder(p);
```

Expected output:

```matlab
dp =
     2
     0
```

### Differentiating complex-valued coefficients

```matlab
p = [1+2i, -3, 4i];
dp = polyder(p);
```

Expected output:

```matlab
dp = [2+4i, -3];
```

### Working with gpuArray inputs

```matlab
g = gpuArray([2 0 -5 4]);
dp = polyder(g);         % stays on the GPU when provider hooks are available
result = gather(dp);
```

Expected behaviour:

```matlab
result = [6 0 -5];
```

## FAQ

### What happens if I pass an empty coefficient vector?
The empty vector represents the zero polynomial. `polyder([])` returns `[0]`, and the product and
quotient forms treat empty inputs as zeros.

### Does `polyder` support column-vector coefficients?
Yes. The orientation of the first polynomial is preserved for single-polynomial and product
derivatives. For the quotient rule, the numerator inherits the orientation of `u` and the
denominator inherits the orientation of `v`.

### How are leading zeros handled in the result?
Leading zeros are removed automatically to mirror MATLAB. If all coefficients cancel out, a single
zero coefficient is returned.

### Can I differentiate logical or integer coefficient vectors?
Yes. Logical and integer inputs are promoted to double precision before differentiation, matching
MATLAB semantics.

### How do I compute the derivative of a rational function?
Call `[num, den] = polyder(u, v)`. The numerator and denominator are the coefficients of
`(u' * v - u * v')` and `v * v`, respectively, with leading zeros removed.

### Does the builtin launch GPU kernels?
Yes whenever the active acceleration provider implements the `polyder` hooks. The in-process and
WGPU backends both execute the derivative on the device. Providers that lack the hooks—or inputs that
require complex arithmetic—fall back to the CPU reference implementation, returning the result on the
host. Wrap the output in `gpuArray` manually if you need to restore device residency in that case.

### What if I request more than two outputs?
`polyder` only defines one or two outputs. Additional requested outputs are filled with `0`, just as
RunMat currently does for other MATLAB builtins whose extra outputs are ignored.

### Are complex coefficients supported in the quotient rule?
Yes. Both the numerator and denominator are computed using full complex arithmetic, so mixed
real/complex inputs work without additional steps.

### Do I need to normalise or pad my coefficient vectors?
No. Provide coefficients exactly as MATLAB expects (highest power first). The builtin takes care of
padding, trimming, and orientation.

### How precise is the computation?
All arithmetic uses IEEE 754 double precision (`f64`), matching MATLAB’s default numeric type.

## See Also
[polyval](./polyval), [polyfit](./polyfit), [conv](./conv), [deconv](./deconv), [gpuArray](./gpuarray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::poly::polyder")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "polyder",
    op_kind: GpuOpKind::Custom("polynomial-derivative"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("polyder-single"),
        ProviderHook::Custom("polyder-product"),
        ProviderHook::Custom("polyder-quotient"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs on-device when providers expose polyder hooks; falls back to the host for complex coefficients or unsupported shapes.",
};

fn polyder_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::poly::polyder")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "polyder",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Symbolic operation on coefficient vectors; fusion bypasses this builtin.",
};

#[runtime_builtin(
    name = "polyder",
    category = "math/poly",
    summary = "Differentiate polynomials, products, and ratios with MATLAB-compatible coefficient vectors.",
    keywords = "polyder,polynomial,derivative,product,quotient",
    builtin_path = "crate::builtins::math::poly::polyder"
)]
fn polyder_builtin(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    match rest.len() {
        0 => derivative_single(first),
        1 => derivative_product(first, rest.into_iter().next().unwrap()),
        _ => Err(polyder_error("polyder: too many input arguments")),
    }
}

fn try_gpu_derivative_single(value: &Value) -> BuiltinResult<Option<Value>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    match provider.polyder_single(handle) {
        Ok(out) => Ok(Some(Value::GpuTensor(out))),
        Err(err) => {
            trace!("polyder: provider polyder_single fallback: {err}");
            Ok(None)
        }
    }
}

fn try_gpu_derivative_product(first: &Value, second: &Value) -> BuiltinResult<Option<Value>> {
    match (first, second) {
        (Value::GpuTensor(p), Value::GpuTensor(q)) => {
            let Some(provider) = runmat_accelerate_api::provider() else {
                return Ok(None);
            };
            match provider.polyder_product(p, q) {
                Ok(out) => Ok(Some(Value::GpuTensor(out))),
                Err(err) => {
                    trace!("polyder: provider polyder_product fallback: {err}");
                    Ok(None)
                }
            }
        }
        _ => Ok(None),
    }
}

fn try_gpu_quotient(u: &Value, v: &Value) -> BuiltinResult<Option<PolyderEval>> {
    match (u, v) {
        (Value::GpuTensor(uh), Value::GpuTensor(vh)) => {
            let Some(provider) = runmat_accelerate_api::provider() else {
                return Ok(None);
            };
            match provider.polyder_quotient(uh, vh) {
                Ok(result) => Ok(Some(PolyderEval {
                    numerator: Value::GpuTensor(result.numerator),
                    denominator: Value::GpuTensor(result.denominator),
                })),
                Err(err) => {
                    trace!("polyder: provider polyder_quotient fallback: {err}");
                    Ok(None)
                }
            }
        }
        _ => Ok(None),
    }
}

/// Evaluate the quotient rule derivative `[num, den] = polyder(u, v)`.
pub fn evaluate_quotient(u: Value, v: Value) -> BuiltinResult<PolyderEval> {
    if let Some(eval) = try_gpu_quotient(&u, &v)? {
        return Ok(eval);
    }
    let u_poly = parse_polynomial("polyder", "U", u)?;
    let v_poly = parse_polynomial("polyder", "V", v)?;
    let numerator = quotient_numerator(&u_poly, &v_poly)?;
    let denominator = quotient_denominator(&v_poly)?;
    Ok(PolyderEval {
        numerator,
        denominator,
    })
}

/// Differentiated outputs for the quotient rule.
#[derive(Clone)]
pub struct PolyderEval {
    numerator: Value,
    denominator: Value,
}

impl PolyderEval {
    /// Numerator coefficients of the derivative `(u' * v - u * v')`.
    pub fn numerator(&self) -> Value {
        self.numerator.clone()
    }

    /// Denominator coefficients `v^2`.
    pub fn denominator(&self) -> Value {
        self.denominator.clone()
    }
}

pub fn derivative_single(value: Value) -> BuiltinResult<Value> {
    if let Some(out) = try_gpu_derivative_single(&value)? {
        return Ok(out);
    }
    let poly = parse_polynomial("polyder", "P", value)?;
    differentiate_polynomial(&poly)
}

pub fn derivative_product(first: Value, second: Value) -> BuiltinResult<Value> {
    if let Some(out) = try_gpu_derivative_product(&first, &second)? {
        return Ok(out);
    }
    let p = parse_polynomial("polyder", "P", first)?;
    let q = parse_polynomial("polyder", "A", second)?;
    product_derivative(&p, &q)
}

fn quotient_numerator(u: &Polynomial, v: &Polynomial) -> BuiltinResult<Value> {
    let du = raw_derivative(&u.coeffs);
    let dv = raw_derivative(&v.coeffs);
    let term1 = poly_convolve(&du, &v.coeffs);
    let term2 = poly_convolve(&u.coeffs, &dv);
    let mut numerator = poly_sub(&term1, &term2);
    numerator = trim_leading_zeros(&numerator);
    coeffs_to_value(&numerator, u.orientation)
}

fn quotient_denominator(v: &Polynomial) -> BuiltinResult<Value> {
    let mut denominator = poly_convolve(&v.coeffs, &v.coeffs);
    denominator = trim_leading_zeros(&denominator);
    coeffs_to_value(&denominator, v.orientation)
}

fn differentiate_polynomial(poly: &Polynomial) -> BuiltinResult<Value> {
    let mut coeffs = raw_derivative(&poly.coeffs);
    coeffs = trim_leading_zeros(&coeffs);
    coeffs_to_value(&coeffs, poly.orientation)
}

fn product_derivative(p: &Polynomial, q: &Polynomial) -> BuiltinResult<Value> {
    let dp = raw_derivative(&p.coeffs);
    let dq = raw_derivative(&q.coeffs);
    let term1 = poly_convolve(&dp, &q.coeffs);
    let term2 = poly_convolve(&p.coeffs, &dq);
    let mut result = poly_add(&term1, &term2);
    result = trim_leading_zeros(&result);
    coeffs_to_value(&result, p.orientation)
}

fn raw_derivative(coeffs: &[Complex64]) -> Vec<Complex64> {
    if coeffs.len() <= 1 {
        return vec![Complex64::new(0.0, 0.0)];
    }
    let mut output = Vec::with_capacity(coeffs.len() - 1);
    let mut power = coeffs.len() - 1;
    for coeff in coeffs.iter().take(coeffs.len() - 1) {
        output.push(*coeff * (power as f64));
        power -= 1;
    }
    output
}

fn poly_convolve(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut result = vec![Complex64::new(0.0, 0.0); a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

fn poly_add(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let len = a.len().max(b.len());
    let mut result = vec![Complex64::new(0.0, 0.0); len];
    for (idx, &value) in a.iter().enumerate() {
        result[len - a.len() + idx] += value;
    }
    for (idx, &value) in b.iter().enumerate() {
        result[len - b.len() + idx] += value;
    }
    result
}

fn poly_sub(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let len = a.len().max(b.len());
    let mut result = vec![Complex64::new(0.0, 0.0); len];
    for (idx, &value) in a.iter().enumerate() {
        result[len - a.len() + idx] += value;
    }
    for (idx, &value) in b.iter().enumerate() {
        result[len - b.len() + idx] -= value;
    }
    result
}

fn trim_leading_zeros(coeffs: &[Complex64]) -> Vec<Complex64> {
    let mut first = None;
    for (idx, coeff) in coeffs.iter().enumerate() {
        if coeff.norm() > EPS {
            first = Some(idx);
            break;
        }
    }
    match first {
        Some(idx) => coeffs[idx..].to_vec(),
        None => vec![Complex64::new(0.0, 0.0)],
    }
}

fn coeffs_to_value(coeffs: &[Complex64], orientation: Orientation) -> BuiltinResult<Value> {
    if coeffs.iter().all(|c| c.im.abs() <= EPS) {
        let data: Vec<f64> = coeffs.iter().map(|c| c.re).collect();
        let shape = orientation.shape_for_len(data.len());
        let tensor = Tensor::new(data, shape)
            .map_err(|e| polyder_error(format!("polyder: {e}")))?;
        Ok(tensor_into_value(tensor))
    } else {
        let data: Vec<(f64, f64)> = coeffs.iter().map(|c| (c.re, c.im)).collect();
        let shape = orientation.shape_for_len(data.len());
        let tensor = ComplexTensor::new(data, shape)
            .map_err(|e| polyder_error(format!("polyder: {e}")))?;
        Ok(complex_tensor_into_value(tensor))
    }
}

fn parse_polynomial(context: &str, label: &str, value: Value) -> BuiltinResult<Polynomial> {
    let gathered = dispatcher::gather_if_needed(&value)?;
    let (coeffs, orientation) = match gathered {
        Value::Tensor(tensor) => {
            ensure_vector_shape(context, label, &tensor.shape)?;
            let orientation = orientation_from_shape(&tensor.shape);
            if tensor.data.is_empty() {
                (vec![Complex64::new(0.0, 0.0)], orientation)
            } else {
                (
                    tensor
                        .data
                        .into_iter()
                        .map(|re| Complex64::new(re, 0.0))
                        .collect(),
                    orientation,
                )
            }
        }
        Value::ComplexTensor(tensor) => {
            ensure_vector_shape(context, label, &tensor.shape)?;
            let orientation = orientation_from_shape(&tensor.shape);
            if tensor.data.is_empty() {
                (vec![Complex64::new(0.0, 0.0)], orientation)
            } else {
                (
                    tensor
                        .data
                        .into_iter()
                        .map(|(re, im)| Complex64::new(re, im))
                        .collect(),
                    orientation,
                )
            }
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(polyder_error)?;
            ensure_vector_shape(context, label, &tensor.shape)?;
            let orientation = orientation_from_shape(&tensor.shape);
            if tensor.data.is_empty() {
                (vec![Complex64::new(0.0, 0.0)], orientation)
            } else {
                (
                    tensor
                        .data
                        .into_iter()
                        .map(|re| Complex64::new(re, 0.0))
                        .collect(),
                    orientation,
                )
            }
        }
        Value::Num(n) => (vec![Complex64::new(n, 0.0)], Orientation::Scalar),
        Value::Int(i) => (vec![Complex64::new(i.to_f64(), 0.0)], Orientation::Scalar),
        Value::Bool(b) => (
            vec![Complex64::new(if b { 1.0 } else { 0.0 }, 0.0)],
            Orientation::Scalar,
        ),
        Value::Complex(re, im) => (vec![Complex64::new(re, im)], Orientation::Scalar),
        other => {
            return Err(polyder_error(format!(
                "{context}: expected {label} to be a numeric vector, got {other:?}"
            )));
        }
    };

    Ok(Polynomial {
        coeffs,
        orientation,
    })
}

fn ensure_vector_shape(context: &str, label: &str, shape: &[usize]) -> BuiltinResult<()> {
    let non_unit = shape.iter().copied().filter(|&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(polyder_error(format!(
            "{context}: {label} must be a vector of coefficients"
        )))
    }
}

#[derive(Clone, Copy)]
enum Orientation {
    Scalar,
    Row,
    Column,
}

impl Orientation {
    fn shape_for_len(self, len: usize) -> Vec<usize> {
        if len <= 1 {
            return vec![1, 1];
        }
        match self {
            Orientation::Scalar => vec![1, len],
            Orientation::Row => vec![1, len],
            Orientation::Column => vec![len, 1],
        }
    }
}

fn orientation_from_shape(shape: &[usize]) -> Orientation {
    for (idx, &dim) in shape.iter().enumerate() {
        if dim != 1 {
            return match idx {
                0 => Orientation::Column,
                1 => Orientation::Row,
                _ => Orientation::Column,
            };
        }
    }
    Orientation::Scalar
}

#[derive(Clone)]
struct Polynomial {
    coeffs: Vec<Complex64>,
    orientation: Orientation,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    fn assert_error_contains(err: crate::RuntimeError, needle: &str) {
        assert!(
            err.message().contains(needle),
            "expected error containing '{needle}', got '{}'",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn derivative_of_cubic_polynomial_is_correct() {
        let tensor = Tensor::new(vec![3.0, -2.0, 5.0, 7.0], vec![1, 4]).unwrap();
        let result = derivative_single(Value::Tensor(tensor)).expect("polyder");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert!(t
                    .data
                    .iter()
                    .zip([9.0, -4.0, 5.0])
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn derivative_of_product_matches_manual_rule() {
        let p = Tensor::new(vec![1.0, 0.0, -2.0], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let result =
            derivative_product(Value::Tensor(p), Value::Tensor(a)).expect("polyder product");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert!(t
                    .data
                    .iter()
                    .zip([3.0, 2.0, -2.0])
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn quotient_rule_produces_expected_num_and_den() {
        let u = Tensor::new(vec![1.0, 0.0, -4.0], vec![1, 3]).unwrap();
        let v = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let eval = evaluate_quotient(Value::Tensor(u), Value::Tensor(v)).expect("polyder quotient");
        match eval.numerator() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert!(t
                    .data
                    .iter()
                    .zip([1.0, -2.0, 4.0])
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected tensor numerator, got {other:?}"),
        }
        match eval.denominator() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert!(t
                    .data
                    .iter()
                    .zip([1.0, -2.0, 1.0])
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected tensor denominator, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn column_vector_orientation_is_preserved() {
        let tensor = Tensor::new(vec![1.0, 0.0, -3.0], vec![3, 1]).unwrap();
        let result = derivative_single(Value::Tensor(tensor)).expect("polyder column");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!(t
                    .data
                    .iter()
                    .zip([2.0, 0.0])
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected column tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_coefficients_are_supported() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.0), (0.0, 4.0)], vec![1, 3]).unwrap();
        let result = derivative_single(Value::ComplexTensor(tensor)).expect("polyder complex");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected = [Complex64::new(2.0, 4.0), Complex64::new(-3.0, 0.0)];
                assert!(t
                    .data
                    .iter()
                    .zip(expected.iter())
                    .all(|((re, im), expected)| {
                        (re - expected.re).abs() < 1e-12 && (im - expected.im).abs() < 1e-12
                    }));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_polynomial_returns_zero() {
        let tensor = Tensor::new(Vec::new(), vec![1, 0]).unwrap();
        let result = derivative_single(Value::Tensor(tensor)).expect("polyder empty");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_matrix_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = derivative_single(Value::Tensor(tensor)).unwrap_err();
        assert_error_contains(err, "vector of coefficients");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_string_input() {
        let err = derivative_single(Value::String("abc".into())).unwrap_err();
        assert_error_contains(err, "numeric vector");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mixed_gpu_cpu_product_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let p = Tensor::new(vec![1.0, 0.0, -2.0], vec![1, 3]).unwrap();
            let q = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
            let cpu_expected =
                derivative_product(Value::Tensor(p.clone()), Value::Tensor(q.clone()))
                    .expect("cpu product");
            let Value::Tensor(cpu_tensor) = cpu_expected else {
                panic!("expected tensor result");
            };

            let view_p = runmat_accelerate_api::HostTensorView {
                data: &p.data,
                shape: &p.shape,
            };
            let handle_p = provider.upload(&view_p).expect("upload p");
            let result = derivative_product(Value::GpuTensor(handle_p), Value::Tensor(q))
                .expect("mixed product");
            let Value::Tensor(host_tensor) = result else {
                panic!("expected host tensor result");
            };
            assert_eq!(host_tensor.shape, cpu_tensor.shape);
            assert!(host_tensor
                .data
                .iter()
                .zip(cpu_tensor.data.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn builtin_rejects_too_many_inputs() {
        let err = super::polyder_builtin(Value::Num(1.0), vec![Value::Num(2.0), Value::Num(3.0)])
            .unwrap_err();
        assert_error_contains(err, "too many input arguments");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_inputs_remain_on_device() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 0.0, -5.0, 4.0], vec![1, 4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = derivative_single(Value::GpuTensor(handle)).expect("polyder gpu");
            let Value::GpuTensor(out_handle) = result else {
                panic!("expected GPU tensor result");
            };
            let gathered = test_support::gather(Value::GpuTensor(out_handle)).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert!(gathered
                .data
                .iter()
                .zip([6.0, 0.0, -5.0])
                .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_product_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let p = Tensor::new(vec![1.0, 0.0, -2.0], vec![1, 3]).unwrap();
            let q = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
            let expected = derivative_product(Value::Tensor(p.clone()), Value::Tensor(q.clone()))
                .expect("cpu product");
            let Value::Tensor(expected_tensor) = expected else {
                panic!("expected tensor output");
            };

            let view_p = runmat_accelerate_api::HostTensorView {
                data: &p.data,
                shape: &p.shape,
            };
            let view_q = runmat_accelerate_api::HostTensorView {
                data: &q.data,
                shape: &q.shape,
            };
            let handle_p = provider.upload(&view_p).expect("upload p");
            let handle_q = provider.upload(&view_q).expect("upload q");
            let gpu_result =
                derivative_product(Value::GpuTensor(handle_p), Value::GpuTensor(handle_q))
                    .expect("gpu product");
            let Value::GpuTensor(gpu_handle) = gpu_result else {
                panic!("expected GPU tensor");
            };
            let gathered = test_support::gather(Value::GpuTensor(gpu_handle)).expect("gather");
            assert_eq!(gathered.shape, expected_tensor.shape);
            assert!(gathered
                .data
                .iter()
                .zip(expected_tensor.data.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_quotient_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let u = Tensor::new(vec![1.0, 0.0, -4.0], vec![1, 3]).unwrap();
            let v = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
            let expected = evaluate_quotient(Value::Tensor(u.clone()), Value::Tensor(v.clone()))
                .expect("cpu quotient");
            let Value::Tensor(expected_num) = expected.numerator() else {
                panic!("expected tensor numerator");
            };
            let Value::Tensor(expected_den) = expected.denominator() else {
                panic!("expected tensor denominator");
            };

            let view_u = runmat_accelerate_api::HostTensorView {
                data: &u.data,
                shape: &u.shape,
            };
            let view_v = runmat_accelerate_api::HostTensorView {
                data: &v.data,
                shape: &v.shape,
            };
            let handle_u = provider.upload(&view_u).expect("upload u");
            let handle_v = provider.upload(&view_v).expect("upload v");
            let gpu_eval =
                evaluate_quotient(Value::GpuTensor(handle_u), Value::GpuTensor(handle_v))
                    .expect("gpu quotient");
            let gpu_num = test_support::gather(gpu_eval.numerator()).expect("gather num");
            let gpu_den = test_support::gather(gpu_eval.denominator()).expect("gather den");
            assert_eq!(gpu_num.shape, expected_num.shape);
            assert_eq!(gpu_den.shape, expected_den.shape);
            assert!(gpu_num
                .data
                .iter()
                .zip(expected_num.data.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            assert!(gpu_den
                .data
                .iter()
                .zip(expected_den.data.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_polyder_single_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![3.0, -2.0, 5.0, 7.0], vec![1, 4]).unwrap();
        let expected = derivative_single(Value::Tensor(tensor.clone())).expect("cpu polyder");
        let Value::Tensor(expected_tensor) = expected else {
            panic!("expected tensor");
        };
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_result = derivative_single(Value::GpuTensor(handle)).expect("gpu polyder");
        let gathered = test_support::gather(gpu_result).expect("gather");
        assert_eq!(gathered.shape, expected_tensor.shape);
        assert!(gathered
            .data
            .iter()
            .zip(expected_tensor.data.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_polyder_product_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let p = Tensor::new(vec![1.0, 0.0, -2.0], vec![1, 3]).unwrap();
        let q = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let expected = derivative_product(Value::Tensor(p.clone()), Value::Tensor(q.clone()))
            .expect("cpu product");
        let Value::Tensor(expected_tensor) = expected else {
            panic!("expected tensor");
        };
        let view_p = runmat_accelerate_api::HostTensorView {
            data: &p.data,
            shape: &p.shape,
        };
        let view_q = runmat_accelerate_api::HostTensorView {
            data: &q.data,
            shape: &q.shape,
        };
        let handle_p = provider.upload(&view_p).expect("upload p");
        let handle_q = provider.upload(&view_q).expect("upload q");
        let gpu_result = derivative_product(Value::GpuTensor(handle_p), Value::GpuTensor(handle_q))
            .expect("gpu product");
        let gathered = test_support::gather(gpu_result).expect("gather");
        assert_eq!(gathered.shape, expected_tensor.shape);
        assert!(gathered
            .data
            .iter()
            .zip(expected_tensor.data.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_polyder_quotient_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let u = Tensor::new(vec![1.0, 0.0, -4.0], vec![1, 3]).unwrap();
        let v = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let expected = evaluate_quotient(Value::Tensor(u.clone()), Value::Tensor(v.clone()))
            .expect("cpu quotient");
        let expected_num = match expected.numerator() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor numerator, got {other:?}"),
        };
        let expected_den = match expected.denominator() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor denominator, got {other:?}"),
        };
        let view_u = runmat_accelerate_api::HostTensorView {
            data: &u.data,
            shape: &u.shape,
        };
        let view_v = runmat_accelerate_api::HostTensorView {
            data: &v.data,
            shape: &v.shape,
        };
        let handle_u = provider.upload(&view_u).expect("upload u");
        let handle_v = provider.upload(&view_v).expect("upload v");
        let gpu_eval = evaluate_quotient(Value::GpuTensor(handle_u), Value::GpuTensor(handle_v))
            .expect("gpu quotient");
        let gpu_num = test_support::gather(gpu_eval.numerator()).expect("gather num");
        let gpu_den = test_support::gather(gpu_eval.denominator()).expect("gather den");
        assert_eq!(gpu_num.shape, expected_num.shape);
        assert_eq!(gpu_den.shape, expected_den.shape);
        assert!(gpu_num
            .data
            .iter()
            .zip(expected_num.data.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
        assert!(gpu_den
            .data
            .iter()
            .zip(expected_den.data.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn derivative_promotes_integers() {
        let value = Value::Int(IntValue::I32(5));
        let result = derivative_single(value).expect("polyder int");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
