//! MATLAB-compatible `roots` builtin with GPU-aware semantics for RunMat.
//!
//! This implementation mirrors MATLAB behaviour, including handling for leading
//! zeros, constant polynomials, and complex-valued coefficients. GPU inputs are
//! gathered to the host because companion matrix eigenvalue computations are
//! currently performed on the CPU.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

const LEADING_ZERO_TOL: f64 = 1.0e-12;
const RESULT_ZERO_TOL: f64 = 1.0e-10;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "roots"
category: "math/poly"
keywords: ["roots", "polynomial roots", "companion matrix", "eigenvalues", "gpu"]
summary: "Compute the roots of a polynomial specified by its coefficients, matching MATLAB semantics including complex output."
references:
  - title: "MATLAB roots documentation"
    url: "https://www.mathworks.com/help/matlab/ref/roots.html"
  - title: "Golub & Van Loan – Matrix Computations, Chapter 7"
    url: "https://doi.org/10.56021/9781421407944"
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Roots builds a companion matrix and computes its eigenvalues on the CPU; GPU inputs are gathered automatically."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::poly::roots::tests"
  integration: "builtins::math::poly::roots::tests::roots_gpu_input_gathers_to_host"
---

# What does the `roots` function do in MATLAB / RunMat?
`roots(p)` returns the zeros of the polynomial whose coefficients are stored in `p`, with coefficients ordered from the highest power of `x` to the constant term. The result is always a column vector whose entries may be complex.

## How does the `roots` function behave in MATLAB / RunMat?
- Leading zeros in the coefficient vector are discarded before solving. If all coefficients are zero, the result is an empty column vector.
- Constant polynomials (degree 0) produce an empty output because they have no finite roots.
- Linear polynomials return the single solution `-b/a`. Higher-degree polynomials are solved via the eigenvalues of the companion matrix.
- Real coefficients can generate complex conjugate root pairs. Small imaginary round-off terms are rounded to zero to match MATLAB formatting.
- Input vectors can be row or column vectors. Higher-dimensional arrays are rejected.
- Inputs may be real or complex. Logical and integer types are converted to double precision automatically.

## `roots` Function GPU Execution Behaviour
RunMat gathers GPU-resident coefficient vectors to the host because the companion matrix eigenvalue computation presently runs only on the CPU. The output is produced on the host as well. When future providers supply a dedicated polynomial root solver, the builtin can be updated to keep residency on-device transparently.

## Examples of using the `roots` function in MATLAB / RunMat

### Finding roots of a quadratic polynomial

```matlab
p = [1 -3 2];
r = roots(p);
```

Expected output:

```matlab
r =
     2
     1
```

### Computing roots that include repeated factors

```matlab
p = [1 -2 1 0];   % (x - 1)^2 * x
r = roots(p);
```

Expected output:

```matlab
r =
     1
     1
     0
```

### Handling leading zeros in the coefficient vector

```matlab
p = [0 0 1 -4];
r = roots(p);
```

Expected output:

```matlab
r =
     4
```

### Calculating complex roots from real coefficients

```matlab
p = [1 0 1];
r = roots(p);
```

Expected output:

```matlab
r =
   0.0000 + 1.0000i
   0.0000 - 1.0000i
```

### Solving roots of a polynomial stored on the GPU

```matlab
p = gpuArray([1 0 -9 0]);
r = roots(p);
```

Expected output:

```matlab
r =
    3.0000
   -3.0000
         0
```

The coefficients are gathered automatically, so no manual `gather` call is required.

## FAQ

### What shape must the coefficient vector have?
`roots` accepts row vectors, column vectors, or 1-D arrays. Higher-dimensional tensors are rejected with an error.

### How are leading zeros handled?
Leading zeros are removed before solving. If all coefficients are zero, `roots` returns an empty column vector.

### Does `roots` preserve the data type of the coefficients?
Coefficients are promoted to double precision internally. The output is a double vector when all roots are real and a complex double vector otherwise.

### Are the roots sorted?
Roots are returned in the order supplied by the eigenvalue computation (typically descending magnitude). MATLAB also does not sort the roots.

### Can I run `roots` entirely on the GPU?
Not yet. RunMat gathers coefficients from the GPU, solves the companion matrix on the CPU, and returns a host-resident vector. When GPU providers add a polynomial root solver, this builtin will automatically route to it.

### How does RunMat handle numerical round-off?
Small imaginary components (|imag| ≤ 1e-10·(1 + |real|)) are rounded to zero so that near-real roots are displayed as real numbers, matching MATLAB formatting.

## See Also
[polyval](./polyval), [polyfit](./polyfit), [residue](../signal/residue), [roots documentation (MathWorks)](https://www.mathworks.com/help/matlab/ref/roots.html)

## Source & Feedback
- The full source code for the implementation of the `roots` function is available at: [`crates/runmat-runtime/src/builtins/math/poly/roots.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/poly/roots.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "roots",
    op_kind: GpuOpKind::Custom("polynomial-roots"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Companion matrix eigenvalue solve executes on the host; providers currently fall back to the CPU implementation.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "roots",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Non-elementwise builtin that terminates fusion and gathers inputs to the host.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("roots", DOC_MD);

#[runtime_builtin(
    name = "roots",
    category = "math/poly",
    summary = "Compute the roots of a polynomial specified by its coefficients.",
    keywords = "roots,polynomial,eigenvalues,companion",
    accel = "sink"
)]
fn roots_builtin(coefficients: Value) -> Result<Value, String> {
    let coeffs = coefficients_to_complex(coefficients)?;
    let trimmed = trim_leading_zeros(coeffs);
    if trimmed.is_empty() || trimmed.len() == 1 {
        return empty_column();
    }
    let roots = solve_roots(&trimmed)?;
    roots_to_value(&roots)
}

fn coefficients_to_complex(value: Value) -> Result<Vec<Complex64>, String> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            tensor_to_complex(tensor)
        }
        Value::Tensor(tensor) => tensor_to_complex(tensor),
        Value::ComplexTensor(tensor) => complex_tensor_to_vec(tensor),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            tensor_to_complex(tensor)
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("roots: {e}"))?;
            tensor_to_complex(tensor)
        }
        Value::Int(i) => {
            let tensor =
                Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("roots: {e}"))?;
            tensor_to_complex(tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("roots: {e}"))?;
            tensor_to_complex(tensor)
        }
        other => Err(format!(
            "roots: expected a numeric vector of polynomial coefficients, got {other:?}"
        )),
    }
}

fn tensor_to_complex(tensor: Tensor) -> Result<Vec<Complex64>, String> {
    ensure_vector_shape("roots", &tensor.shape)?;
    Ok(tensor
        .data
        .into_iter()
        .map(|value| Complex64::new(value, 0.0))
        .collect())
}

fn complex_tensor_to_vec(tensor: ComplexTensor) -> Result<Vec<Complex64>, String> {
    ensure_vector_shape("roots", &tensor.shape)?;
    Ok(tensor
        .data
        .into_iter()
        .map(|(re, im)| Complex64::new(re, im))
        .collect())
}

fn ensure_vector_shape(name: &str, shape: &[usize]) -> Result<(), String> {
    let is_vector = match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1 || shape.iter().product::<usize>() == 0,
        _ => shape.iter().filter(|&&dim| dim > 1).count() <= 1,
    };
    if !is_vector {
        return Err(format!(
            "{name}: coefficients must be a vector (row or column), got shape {:?}",
            shape
        ));
    }
    Ok(())
}

fn trim_leading_zeros(mut coeffs: Vec<Complex64>) -> Vec<Complex64> {
    if coeffs.is_empty() {
        return coeffs;
    }
    let scale = coeffs.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
    let tol = if scale == 0.0 {
        LEADING_ZERO_TOL
    } else {
        LEADING_ZERO_TOL * scale
    };
    let first_nonzero = coeffs
        .iter()
        .position(|c| c.norm() > tol)
        .unwrap_or(coeffs.len());
    coeffs.split_off(first_nonzero)
}

fn solve_roots(coeffs: &[Complex64]) -> Result<Vec<Complex64>, String> {
    if coeffs.len() <= 1 {
        return Ok(Vec::new());
    }
    if coeffs.len() == 2 {
        let a = coeffs[0];
        let b = coeffs[1];
        if a.norm() <= LEADING_ZERO_TOL {
            return Err("roots: leading coefficient must be non-zero after trimming".to_string());
        }
        return Ok(vec![-b / a]);
    }

    let degree = coeffs.len() - 1;
    if degree == 3 {
        return Ok(cubic_roots(coeffs[0], coeffs[1], coeffs[2], coeffs[3]));
    }
    let leading = coeffs[0];
    if leading.norm() <= LEADING_ZERO_TOL {
        return Err("roots: leading coefficient must be non-zero after trimming".to_string());
    }

    let mut companion = DMatrix::<Complex64>::zeros(degree, degree);
    for row in 1..degree {
        companion[(row, row - 1)] = Complex64::new(1.0, 0.0);
    }

    for (idx, coeff) in coeffs.iter().enumerate().skip(1) {
        let value = -(*coeff) / leading;
        let column = idx - 1;
        if column < degree {
            companion[(0, column)] = value;
        }
    }

    let eigenvalues = companion.clone().eigenvalues().ok_or_else(|| {
        "roots: failed to compute eigenvalues of the companion matrix".to_string()
    })?;
    Ok(eigenvalues.iter().map(|&z| canonicalize_root(z)).collect())
}

fn cubic_roots(a: Complex64, b: Complex64, c: Complex64, d: Complex64) -> Vec<Complex64> {
    // Depressed cubic via Cardano: x = y - b/(3a), y^3 + p y + q = 0
    let three = 3.0;
    let nine = 9.0;
    let twenty_seven = 27.0;
    let a2 = a * a;
    let a3 = a2 * a;
    let p = (three * a * c - b * b) / (three * a2);
    let q = (twenty_seven * a2 * d - nine * a * b * c + Complex64::new(2.0, 0.0) * b * b * b)
        / (twenty_seven * a3);
    let half = Complex64::new(0.5, 0.0);
    let disc = (q * q) * half * half + (p * p * p) / Complex64::new(27.0, 0.0);
    let sqrt_disc = disc.sqrt();
    let u = (-q * half + sqrt_disc).powf(1.0 / 3.0);
    let v = (-q * half - sqrt_disc).powf(1.0 / 3.0);
    let omega = Complex64::new(-0.5, (3.0f64).sqrt() * 0.5);
    let omega2 = omega * omega;
    let shift = b / (three * a);
    let y0 = u + v;
    let y1 = u * omega + v * omega.conj();
    let y2 = u * omega2 + v * omega;
    vec![y0 - shift, y1 - shift, y2 - shift]
}

fn canonicalize_root(z: Complex64) -> Complex64 {
    if !z.re.is_finite() || !z.im.is_finite() {
        return z;
    }
    let mut real = z.re;
    let mut imag = z.im;
    let scale = 1.0 + real.abs();
    if imag.abs() <= RESULT_ZERO_TOL * scale {
        imag = 0.0;
    }
    if real.abs() <= RESULT_ZERO_TOL {
        real = 0.0;
    }
    Complex64::new(real, imag)
}

fn roots_to_value(roots: &[Complex64]) -> Result<Value, String> {
    if roots.is_empty() {
        return empty_column();
    }
    let all_real = roots
        .iter()
        .all(|z| z.im.abs() <= RESULT_ZERO_TOL * (1.0 + z.re.abs()));
    if all_real {
        let mut data: Vec<f64> = Vec::with_capacity(roots.len());
        for &root in roots {
            data.push(root.re);
        }
        let tensor = Tensor::new(data, vec![roots.len(), 1]).map_err(|e| format!("roots: {e}"))?;
        Ok(Value::Tensor(tensor))
    } else {
        let data: Vec<(f64, f64)> = roots.iter().map(|z| (z.re, z.im)).collect();
        let tensor =
            ComplexTensor::new(data, vec![roots.len(), 1]).map_err(|e| format!("roots: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn empty_column() -> Result<Value, String> {
    let tensor = Tensor::new(Vec::new(), vec![0, 1]).map_err(|e| format!("roots: {e}"))?;
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ComplexTensor, LogicalArray, Tensor};

    #[test]
    fn roots_quadratic_real() {
        let coeffs = Tensor::new(vec![1.0, -3.0, 2.0], vec![3, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let mut roots = t.data;
                roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert!((roots[0] - 1.0).abs() < 1e-10);
                assert!((roots[1] - 2.0).abs() < 1e-10);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[test]
    fn roots_leading_zeros_trimmed() {
        let coeffs = Tensor::new(vec![0.0, 0.0, 1.0, -4.0], vec![4, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.data[0] - 4.0).abs() < 1e-10);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn roots_complex_pair() {
        let coeffs = Tensor::new(vec![1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                let mut roots = t.data;
                roots.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                assert!((roots[0].0).abs() < 1e-10);
                assert!((roots[0].1 + 1.0).abs() < 1e-10);
                assert!((roots[1].0).abs() < 1e-10);
                assert!((roots[1].1 - 1.0).abs() < 1e-10);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn roots_quartic_all_zero_roots() {
        // p(x) = x^4 => 4 roots at 0
        let coeffs = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![5, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots quartic");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 1]);
                for &r in &t.data {
                    assert!(r.abs() < 1e-8);
                }
            }
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![4, 1]);
                for &(re, im) in &t.data {
                    assert!(re.abs() < 1e-7 && im.abs() < 1e-7);
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn roots_accepts_complex_coefficients_input() {
        // p(x) = x^2 + 1 with complex coefficients path
        let coeffs =
            ComplexTensor::new(vec![(1.0, 0.0), (0.0, 0.0), (1.0, 0.0)], vec![3, 1]).unwrap();
        let result = roots_builtin(Value::ComplexTensor(coeffs)).expect("roots complex input");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                // roots at i and -i
                let mut roots = t.data;
                roots.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                assert!(roots[0].0.abs() < 1e-10 && (roots[0].1 + 1.0).abs() < 1e-6);
                assert!(roots[1].0.abs() < 1e-10 && (roots[1].1 - 1.0).abs() < 1e-6);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn roots_accepts_logical_coefficients() {
        // p(x) = x with logical coefficients [1 0]
        let la = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = roots_builtin(Value::LogicalArray(la)).expect("roots logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!(t.data[0].abs() < 1e-12);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[test]
    fn roots_scalar_num_returns_empty() {
        let result = roots_builtin(Value::Num(5.0)).expect("roots scalar num");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 1]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn roots_rejects_non_vector_input() {
        let coeffs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = roots_builtin(Value::Tensor(coeffs)).expect_err("expected vector-shape error");
        assert!(err.to_lowercase().contains("vector"));
    }

    #[test]
    fn roots_all_zero_coefficients_returns_empty() {
        let coeffs = Tensor::new(vec![0.0, 0.0, 0.0], vec![3, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 1]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn roots_gpu_input_gathers_to_host() {
        test_support::with_test_provider(|provider| {
            let coeffs = Tensor::new(vec![1.0, 0.0, -9.0, 0.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &coeffs.data,
                shape: &coeffs.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = roots_builtin(Value::GpuTensor(handle)).expect("roots");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            let mut roots = gathered.data;
            roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((roots[0] + 3.0).abs() < 1e-9);
            assert!((roots[1]).abs() < 1e-9);
            assert!((roots[2] - 3.0).abs() < 1e-9);
        });
    }

    #[test]
    fn roots_constant_polynomial_returns_empty() {
        let coeffs = Tensor::new(vec![5.0], vec![1, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 1]);
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
