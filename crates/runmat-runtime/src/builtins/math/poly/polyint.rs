//! MATLAB-compatible `polyint` builtin with GPU-aware semantics for RunMat.

use log::{trace, warn};
use num_complex::Complex64;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::dispatcher;

const EPS: f64 = 1.0e-12;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "polyint",
        builtin_path = "crate::builtins::math::poly::polyint"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "polyint"
category: "math/poly"
keywords: ["polyint", "polynomial integral", "antiderivative", "integration constant", "gpu"]
summary: "Integrate polynomial coefficient vectors and append a constant of integration."
references:
  - title: "MATLAB polyint documentation"
    url: "https://www.mathworks.com/help/matlab/ref/polyint.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Real-valued coefficient vectors stay on the GPU when the provider exposes the polyint hook; complex outputs fall back to the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::poly::polyint::tests"
  integration: "builtins::math::poly::polyint::tests::polyint_gpu_roundtrip"
  doc: "builtins::math::poly::polyint::tests::doc_examples_present"
---

# What does the `polyint` function do in MATLAB / RunMat?
`polyint(p)` returns the polynomial that integrates the coefficient vector `p` once, appending the
constant of integration at the end of the coefficient list. Coefficients follow MATLAB's descending
power convention: `p(1)` multiplies the highest power of `x`, and `p(end)` is the constant term.

## How does the `polyint` function behave in MATLAB / RunMat?
- Accepts real or complex scalars, row vectors, column vectors, or empty vectors. Inputs with more
  than one non-singleton dimension raise MATLAB-compatible errors.
- Logical and integer coefficients are promoted to double precision before integration.
- The optional second argument supplies the constant of integration. It must be a scalar (real or
  complex). When omitted, the constant defaults to `0`.
- Leading zeros are preserved. Integrating `[0 0 5]` produces `[0 0 5 0]`, matching MATLAB.
- Empty inputs integrate to the constant of integration (default `0`). Specifying a constant `k` yields `[k]`.
- The orientation of the input vector is preserved: row vectors stay row vectors, column vectors
  stay column vectors, and scalars return a row vector.
- When coefficients reside on the GPU, RunMat gathers them to the host, performs the integration,
  and re-uploads real-valued results so downstream kernels retain residency.

## `polyint` Function GPU Execution Behaviour
When a GPU provider is registered and the coefficient vector is real-valued, RunMat calls the
provider's dedicated `polyint` kernel. The input stays on the device, the kernel divides each
coefficient by the appropriate power, and the supplied constant of integration is written directly
into device memory. If the coefficients or the constant are complex, or if the provider reports that
the hook is unavailable, the runtime gathers data back to the host, performs the integration in
double precision, and re-uploads the result when it is purely real.

## Examples of using the `polyint` function in MATLAB / RunMat

### Integrating a cubic polynomial

```matlab
p = [3 -2 5 7];        % 3x^3 - 2x^2 + 5x + 7
q = polyint(p);
```

Expected output:

```matlab
q = [0.75  -0.6667  2.5  7  0];
```

### Supplying a constant of integration

```matlab
p = [4 0 -8];
q = polyint(p, 3);
```

Expected output:

```matlab
q = [1.3333  0  -8  3];
```

### Preserving column-vector orientation

```matlab
p = [2; 0; -6];
q = polyint(p);
```

Expected output:

```matlab
q =
    0.6667
         0
   -6.0000
         0
```

### Integrating the zero polynomial

```matlab
q = polyint([]);
```

Expected output:

```matlab
q = 0;
```

### Integrating complex coefficients

```matlab
p = [1+2i  -3  4i];
q = polyint(p, -1i);
```

Expected output:

```matlab
q = [(1+2i)/3  -1.5  4i  -1i];
```

### Working with gpuArray inputs

```matlab
g = gpuArray([1 -4 6]);
q = polyint(g);          % Gathered to host, integrated, and re-uploaded
result = gather(q);
```

Expected behavior:

```matlab
result = [0.3333  -2  6  0];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` just for `polyint`. When inputs already live on the
GPU and a provider is active, RunMat keeps the data on the device and executes the integration there
for real-valued coefficients. For complex inputs, or when no provider hook is available, the runtime
falls back to the host implementation transparently.

## FAQ

### Does `polyint` change the orientation of my coefficients?
No. Row vectors stay row vectors, column vectors stay column vectors, and scalars return a row
vector with two elements after integration.

### What happens when I pass an empty vector?
An empty vector represents the zero polynomial. `polyint([])` returns the constant of integration,
so the default result is `0` and `polyint([], k)` returns `k`.

### Can the constant of integration be complex?
Yes. Provide any scalar numeric value (real or complex). The constant is appended to the integrated
polynomial exactly as MATLAB does.

### Are logical or integer coefficients supported?
Yes. They are promoted to double precision before integration, ensuring identical behaviour to
MATLAB.

### Will the result stay on the GPU?
Real-valued outputs are re-uploaded to the GPU when a provider is available. Complex outputs remain
on the host because current providers do not expose complex tensor handles.

### How precise is the computation?
All arithmetic uses IEEE 754 double precision (`f64`), mirroring MATLAB's default numeric type.

## See Also
[polyder](./polyder), [polyval](./polyval), [polyfit](./polyfit), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::poly::polyint")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "polyint",
    op_kind: GpuOpKind::Custom("polynomial-integral"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("polyint")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers implement the polyint hook for real coefficient vectors; complex inputs fall back to the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::poly::polyint")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "polyint",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Symbolic operation on coefficient vectors; fusion does not apply.",
};

#[runtime_builtin(
    name = "polyint",
    category = "math/poly",
    summary = "Integrate polynomial coefficient vectors and append a constant of integration.",
    keywords = "polyint,polynomial,integral,antiderivative",
    builtin_path = "crate::builtins::math::poly::polyint"
)]
fn polyint_builtin(coeffs: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() > 1 {
        return Err("polyint: too many input arguments".to_string());
    }

    let constant = rest
        .into_iter()
        .next()
        .map(parse_constant)
        .transpose()?
        .unwrap_or_else(|| Complex64::new(0.0, 0.0));

    if let Value::GpuTensor(handle) = &coeffs {
        if let Some(device_result) = try_polyint_gpu(handle, constant)? {
            return Ok(Value::GpuTensor(device_result));
        }
    }

    let was_gpu = matches!(coeffs, Value::GpuTensor(_));
    polyint_host_value(coeffs, constant, was_gpu)
}

fn polyint_host_value(coeffs: Value, constant: Complex64, was_gpu: bool) -> Result<Value, String> {
    let polynomial = parse_polynomial(coeffs)?;
    let mut integrated = integrate_coeffs(&polynomial.coeffs);
    if integrated.is_empty() {
        integrated.push(constant);
    } else if let Some(last) = integrated.last_mut() {
        *last += constant;
    }
    let value = coeffs_to_value(&integrated, polynomial.orientation)?;
    maybe_return_gpu(value, was_gpu)
}

fn try_polyint_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    constant: Complex64,
) -> Result<Option<runmat_accelerate_api::GpuTensorHandle>, String> {
    if constant.im.abs() > EPS {
        return Ok(None);
    }
    ensure_vector_shape(&handle.shape)?;
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    match provider.polyint(handle, constant.re) {
        Ok(result) => Ok(Some(result)),
        Err(err) => {
            trace!("polyint: provider hook unavailable, falling back to host: {err}");
            Ok(None)
        }
    }
}

fn integrate_coeffs(coeffs: &[Complex64]) -> Vec<Complex64> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(coeffs.len() + 1);
    for (idx, coeff) in coeffs.iter().enumerate() {
        let power = (coeffs.len() - idx) as f64;
        if power <= 0.0 {
            result.push(Complex64::new(0.0, 0.0));
        } else {
            result.push(*coeff / Complex64::new(power, 0.0));
        }
    }
    result.push(Complex64::new(0.0, 0.0));
    result
}

fn maybe_return_gpu(value: Value, was_gpu: bool) -> Result<Value, String> {
    if !was_gpu {
        return Ok(value);
    }
    match value {
        Value::Tensor(tensor) => {
            if let Some(provider) = runmat_accelerate_api::provider() {
                let view = HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                };
                match provider.upload(&view) {
                    Ok(handle) => return Ok(Value::GpuTensor(handle)),
                    Err(err) => {
                        warn!("polyint: provider upload failed, keeping result on host: {err}");
                    }
                }
            } else {
                trace!("polyint: no provider available to re-upload result");
            }
            Ok(Value::Tensor(tensor))
        }
        other => Ok(other),
    }
}

fn coeffs_to_value(coeffs: &[Complex64], orientation: Orientation) -> Result<Value, String> {
    if coeffs.iter().all(|c| c.im.abs() <= EPS) {
        let data: Vec<f64> = coeffs.iter().map(|c| c.re).collect();
        let shape = orientation.shape_for_len(data.len());
        let tensor = Tensor::new(data, shape).map_err(|e| format!("polyint: {e}"))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let data: Vec<(f64, f64)> = coeffs.iter().map(|c| (c.re, c.im)).collect();
        let shape = orientation.shape_for_len(data.len());
        let tensor = ComplexTensor::new(data, shape).map_err(|e| format!("polyint: {e}"))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn parse_polynomial(value: Value) -> Result<Polynomial, String> {
    let gathered = dispatcher::gather_if_needed(&value).map_err(|e| format!("polyint: {e}"))?;
    match gathered {
        Value::Tensor(tensor) => parse_tensor_coeffs(&tensor),
        Value::ComplexTensor(tensor) => parse_complex_tensor_coeffs(&tensor),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            parse_tensor_coeffs(&tensor)
        }
        Value::Num(n) => Ok(Polynomial {
            coeffs: vec![Complex64::new(n, 0.0)],
            orientation: Orientation::Scalar,
        }),
        Value::Int(i) => Ok(Polynomial {
            coeffs: vec![Complex64::new(i.to_f64(), 0.0)],
            orientation: Orientation::Scalar,
        }),
        Value::Bool(b) => Ok(Polynomial {
            coeffs: vec![Complex64::new(if b { 1.0 } else { 0.0 }, 0.0)],
            orientation: Orientation::Scalar,
        }),
        Value::Complex(re, im) => Ok(Polynomial {
            coeffs: vec![Complex64::new(re, im)],
            orientation: Orientation::Scalar,
        }),
        other => Err(format!(
            "polyint: expected a numeric coefficient vector, got {:?}",
            other
        )),
    }
}

fn parse_tensor_coeffs(tensor: &Tensor) -> Result<Polynomial, String> {
    ensure_vector_shape(&tensor.shape)?;
    let orientation = orientation_from_shape(&tensor.shape);
    Ok(Polynomial {
        coeffs: tensor
            .data
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect(),
        orientation,
    })
}

fn parse_complex_tensor_coeffs(tensor: &ComplexTensor) -> Result<Polynomial, String> {
    ensure_vector_shape(&tensor.shape)?;
    let orientation = orientation_from_shape(&tensor.shape);
    Ok(Polynomial {
        coeffs: tensor
            .data
            .iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect(),
        orientation,
    })
}

fn parse_constant(value: Value) -> Result<Complex64, String> {
    let gathered = dispatcher::gather_if_needed(&value).map_err(|e| format!("polyint: {e}"))?;
    match gathered {
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err("polyint: constant of integration must be a scalar".to_string());
            }
            Ok(Complex64::new(tensor.data[0], 0.0))
        }
        Value::ComplexTensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err("polyint: constant of integration must be a scalar".to_string());
            }
            let (re, im) = tensor.data[0];
            Ok(Complex64::new(re, im))
        }
        Value::Num(n) => Ok(Complex64::new(n, 0.0)),
        Value::Int(i) => Ok(Complex64::new(i.to_f64(), 0.0)),
        Value::Bool(b) => Ok(Complex64::new(if b { 1.0 } else { 0.0 }, 0.0)),
        Value::Complex(re, im) => Ok(Complex64::new(re, im)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            if tensor.data.len() != 1 {
                return Err("polyint: constant of integration must be a scalar".to_string());
            }
            Ok(Complex64::new(tensor.data[0], 0.0))
        }
        other => Err(format!(
            "polyint: constant of integration must be numeric, got {:?}",
            other
        )),
    }
}

fn ensure_vector_shape(shape: &[usize]) -> Result<(), String> {
    let non_unit = shape.iter().filter(|&&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err("polyint: coefficients must form a vector".to_string())
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
            Orientation::Scalar | Orientation::Row => vec![1, len],
            Orientation::Column => vec![len, 1],
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::LogicalArray;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integrates_polynomial_without_constant() {
        let tensor = Tensor::new(vec![3.0, -2.0, 5.0, 7.0], vec![1, 4]).unwrap();
        let result = polyint_builtin(Value::Tensor(tensor), Vec::new()).expect("polyint");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                let expected = [0.75, -2.0 / 3.0, 2.5, 7.0, 0.0];
                assert!(t
                    .data
                    .iter()
                    .zip(expected.iter())
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integrates_with_constant() {
        let tensor = Tensor::new(vec![4.0, 0.0, -8.0], vec![1, 3]).unwrap();
        let args = vec![Value::Num(3.0)];
        let result = polyint_builtin(Value::Tensor(tensor), args).expect("polyint");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [4.0 / 3.0, 0.0, -8.0, 3.0];
                assert!(t
                    .data
                    .iter()
                    .zip(expected.iter())
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integrates_scalar_value() {
        let result = polyint_builtin(Value::Num(5.0), Vec::new()).expect("polyint");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - 5.0).abs() < 1e-12);
                assert!(t.data[1].abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integrates_logical_coefficients() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let result =
            polyint_builtin(Value::LogicalArray(logical), Vec::new()).expect("polyint logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [1.0 / 3.0, 0.0, 1.0, 0.0];
                assert!(t
                    .data
                    .iter()
                    .zip(expected.iter())
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn preserves_column_vector_orientation() {
        let tensor = Tensor::new(vec![2.0, 0.0, -6.0], vec![3, 1]).unwrap();
        let result = polyint_builtin(Value::Tensor(tensor), Vec::new()).expect("polyint");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 1]);
                let expected = [2.0 / 3.0, 0.0, -6.0, 0.0];
                assert!(t
                    .data
                    .iter()
                    .zip(expected.iter())
                    .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
            }
            other => panic!("expected column tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integrates_complex_coefficients() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 0.0), (0.0, 4.0)], vec![1, 3]).unwrap();
        let args = vec![Value::Complex(0.0, -1.0)];
        let result = polyint_builtin(Value::ComplexTensor(tensor), args).expect("polyint");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [(1.0 / 3.0, 2.0 / 3.0), (-1.5, 0.0), (0.0, 4.0), (0.0, -1.0)];
                assert!(t
                    .data
                    .iter()
                    .zip(expected.iter())
                    .all(|((lre, lim), (rre, rim))| {
                        (lre - rre).abs() < 1e-12 && (lim - rim).abs() < 1e-12
                    }));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_matrix_coefficients() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = polyint_builtin(Value::Tensor(tensor), Vec::new()).expect_err("expected error");
        assert!(err.contains("vector"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_scalar_constant() {
        let coeffs = Tensor::new(vec![1.0, -4.0, 6.0], vec![1, 3]).unwrap();
        let constant = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err = polyint_builtin(Value::Tensor(coeffs), vec![Value::Tensor(constant)])
            .expect_err("expected error");
        assert!(err.contains("constant of integration must be a scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_excess_arguments() {
        let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let err = polyint_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(1.0), Value::Num(2.0)],
        )
        .expect_err("expected error");
        assert!(err.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handles_empty_input_as_zero_polynomial() {
        let tensor = Tensor::new(vec![], vec![1, 0]).unwrap();
        let result = polyint_builtin(Value::Tensor(tensor), Vec::new()).expect("polyint");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            Value::Tensor(t) => {
                // Allow tensor fallback if scalar auto-boxing changes in future
                assert_eq!(t.data.len(), 1);
                assert!(t.data[0].abs() < 1e-12);
            }
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_input_with_constant() {
        let tensor = Tensor::new(vec![], vec![1, 0]).unwrap();
        let result = polyint_builtin(Value::Tensor(tensor), vec![Value::Complex(1.5, -2.0)])
            .expect("polyint");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert_eq!(t.data.len(), 1);
                let (re, im) = t.data[0];
                assert!((re - 1.5).abs() < 1e-12);
                assert!((im + 2.0).abs() < 1e-12);
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyint_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, -4.0, 6.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = polyint_builtin(Value::GpuTensor(handle), Vec::new()).expect("polyint");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 4]);
                    let expected = [1.0 / 3.0, -2.0, 6.0, 0.0];
                    assert!(gathered
                        .data
                        .iter()
                        .zip(expected.iter())
                        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
                }
                other => panic!("expected GPU tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyint_gpu_complex_constant_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = polyint_builtin(Value::GpuTensor(handle), vec![Value::Complex(0.0, 2.0)])
                .expect("polyint");
            match result {
                Value::ComplexTensor(ct) => {
                    assert_eq!(ct.shape, vec![1, 3]);
                    let expected = [(0.5, 0.0), (0.0, 0.0), (0.0, 2.0)];
                    assert!(ct
                        .data
                        .iter()
                        .zip(expected.iter())
                        .all(|((lre, lim), (rre, rim))| {
                            (lre - rre).abs() < 1e-12 && (lim - rim).abs() < 1e-12
                        }));
                }
                other => panic!("expected complex tensor fall-back, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polyint_gpu_with_gpu_constant() {
        test_support::with_test_provider(|provider| {
            let coeffs = Tensor::new(vec![2.0, 0.0], vec![1, 2]).unwrap();
            let coeff_view = HostTensorView {
                data: &coeffs.data,
                shape: &coeffs.shape,
            };
            let coeff_handle = provider.upload(&coeff_view).expect("upload coeffs");
            let constant = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
            let constant_view = HostTensorView {
                data: &constant.data,
                shape: &constant.shape,
            };
            let constant_handle = provider.upload(&constant_view).expect("upload constant");
            let result = polyint_builtin(
                Value::GpuTensor(coeff_handle),
                vec![Value::GpuTensor(constant_handle)],
            )
            .expect("polyint");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(handle)).expect("gather result");
                    assert_eq!(gathered.shape, vec![1, 3]);
                    let expected = [1.0, 0.0, 3.0];
                    assert!(gathered
                        .data
                        .iter()
                        .zip(expected.iter())
                        .all(|(lhs, rhs)| (lhs - rhs).abs() < 1e-12));
                }
                other => panic!("expected gpu tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn polyint_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![3.0, -2.0, 5.0, 7.0], vec![1, 4]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = polyint_builtin(Value::GpuTensor(handle), Vec::new()).expect("polyint gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        let cpu_value =
            polyint_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("polyint cpu");
        let expected = match cpu_value {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu result {other:?}"),
        };
        assert_eq!(gathered.shape, expected.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        gathered
            .data
            .iter()
            .zip(expected.data.iter())
            .for_each(|(lhs, rhs)| assert!((lhs - rhs).abs() < tol));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
