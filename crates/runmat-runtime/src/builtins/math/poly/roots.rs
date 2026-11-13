//! MATLAB-compatible `roots` builtin with GPU-aware semantics for RunMat.
//!
//! This implementation mirrors MATLAB behaviour, including handling for leading
//! zeros, constant polynomials, and complex-valued coefficients. GPU inputs are
//! gathered to the host because companion matrix eigenvalue computations are
//! currently performed on the CPU.

use nalgebra::DMatrix;
use num_complex::Complex64;
#[cfg(test)]
use runmat_builtins::ROOTS_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, ROOTS_ERROR_INTERNAL, ROOTS_ERROR_INVALID_INPUT,
    ROOTS_INTEGER_COEFFICIENTS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "roots";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::poly::roots")]
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

fn roots_error(message: impl Into<String>) -> RuntimeError {
    roots_error_with(message, &ROOTS_ERROR_INVALID_INPUT)
}

fn roots_error_with(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::poly::roots")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "roots",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Non-elementwise builtin that terminates fusion and gathers inputs to the host.",
};

#[runtime_builtin(
    name = "roots",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::poly::roots"
)]
async fn roots_builtin(coefficients: Value) -> crate::BuiltinResult<Value> {
    roots_value(coefficients).await
}

pub(crate) async fn roots_value(coefficients: Value) -> crate::BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&coefficients, BUILTIN_NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &coefficients,
        &ROOTS_INTEGER_COEFFICIENTS_EXTENSION,
        BUILTIN_NAME,
        "coefficient",
    )
    .await?;
    let coeffs = coefficients_to_complex(coefficients).await?;
    let trimmed = trim_leading_zeros(coeffs);
    if trimmed.is_empty() || trimmed.len() == 1 {
        return empty_column();
    }
    let roots = solve_roots(&trimmed)?;
    roots_to_value(&roots)
}

async fn coefficients_to_complex(value: Value) -> BuiltinResult<Vec<Complex64>> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            tensor_to_complex(tensor)
        }
        Value::Tensor(tensor) => tensor_to_complex(tensor),
        Value::ComplexTensor(tensor) => complex_tensor_to_vec(tensor),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(roots_error)?;
            tensor_to_complex(tensor)
        }
        Value::Num(n) => {
            let tensor =
                Tensor::new(vec![n], vec![1, 1]).map_err(|e| roots_error(format!("roots: {e}")))?;
            tensor_to_complex(tensor)
        }
        Value::Int(i) => {
            let tensor = Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|e| roots_error(format!("roots: {e}")))?;
            tensor_to_complex(tensor)
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| roots_error(format!("roots: {e}")))?;
            tensor_to_complex(tensor)
        }
        other => Err(roots_error(format!(
            "roots: expected a numeric vector of polynomial coefficients, got {other:?}"
        ))),
    }
}

fn tensor_to_complex(tensor: Tensor) -> BuiltinResult<Vec<Complex64>> {
    ensure_vector_shape("roots", tensor.shape())?;
    Ok(tensor::tensor_values_f64(&tensor)
        .into_iter()
        .map(|value| Complex64::new(value, 0.0))
        .collect())
}

fn complex_tensor_to_vec(tensor: ComplexTensor) -> BuiltinResult<Vec<Complex64>> {
    ensure_vector_shape("roots", tensor.shape())?;
    Ok(tensor
        .materialize_f64()
        .into_iter()
        .map(|(re, im)| Complex64::new(re, im))
        .collect())
}

fn ensure_vector_shape(name: &str, shape: &[usize]) -> BuiltinResult<()> {
    let is_vector = match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1 || shape.contains(&0),
        _ => shape.iter().filter(|&&dim| dim > 1).count() <= 1,
    };
    if !is_vector {
        return Err(roots_error(format!(
            "{name}: coefficients must be a vector (row or column), got shape {:?}",
            shape
        )));
    }
    Ok(())
}

fn trim_leading_zeros(mut coeffs: Vec<Complex64>) -> Vec<Complex64> {
    if coeffs.is_empty() {
        return coeffs;
    }
    let first_nonzero = coeffs
        .iter()
        .position(|c| c.re != 0.0 || c.im != 0.0)
        .unwrap_or(coeffs.len());
    coeffs.split_off(first_nonzero)
}

fn solve_roots(coeffs: &[Complex64]) -> BuiltinResult<Vec<Complex64>> {
    if coeffs.len() <= 1 {
        return Ok(Vec::new());
    }
    if coeffs.len() == 2 {
        let a = coeffs[0];
        let b = coeffs[1];
        if a.re == 0.0 && a.im == 0.0 {
            return Err(roots_error(
                "roots: leading coefficient must be non-zero after trimming",
            ));
        }
        return Ok(vec![-b / a]);
    }

    let degree = coeffs.len() - 1;
    if degree == 3 {
        return Ok(cubic_roots(coeffs[0], coeffs[1], coeffs[2], coeffs[3]));
    }
    let leading = coeffs[0];
    if leading.re == 0.0 && leading.im == 0.0 {
        return Err(roots_error(
            "roots: leading coefficient must be non-zero after trimming",
        ));
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
        roots_error_with(
            "roots: failed to compute eigenvalues of the companion matrix",
            &ROOTS_ERROR_INTERNAL,
        )
    })?;
    Ok(eigenvalues.iter().copied().collect())
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

fn roots_to_value(roots: &[Complex64]) -> BuiltinResult<Value> {
    if roots.is_empty() {
        return empty_column();
    }
    let all_real = roots.iter().all(|z| z.im == 0.0);
    if all_real {
        let mut data: Vec<f64> = Vec::with_capacity(roots.len());
        for &root in roots {
            data.push(root.re);
        }
        let tensor = Tensor::new(data, vec![roots.len(), 1])
            .map_err(|e| roots_error_with(format!("roots: {e}"), &ROOTS_ERROR_INTERNAL))?;
        Ok(Value::Tensor(tensor))
    } else {
        let data: Vec<(f64, f64)> = roots.iter().map(|z| (z.re, z.im)).collect();
        let tensor = ComplexTensor::new(data, vec![roots.len(), 1])
            .map_err(|e| roots_error_with(format!("roots: {e}"), &ROOTS_ERROR_INTERNAL))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn empty_column() -> BuiltinResult<Value> {
    let tensor = Tensor::new(Vec::new(), vec![0, 1])
        .map_err(|e| roots_error_with(format!("roots: {e}"), &ROOTS_ERROR_INTERNAL))?;
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{ComplexTensor, IntegerStorage, LogicalArray, Tensor};

    fn assert_error_contains(err: crate::RuntimeError, needle: &str) {
        assert!(
            err.message().contains(needle),
            "expected error containing '{needle}', got '{}'",
            err.message()
        );
    }

    #[test]
    fn roots_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ROOTS_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"r = roots(c)"));
    }

    #[test]
    fn roots_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = ROOTS_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.ROOTS.INVALID_INPUT"));
        assert!(codes.contains(&"RM.ROOTS.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_quadratic_real() {
        let coeffs = Tensor::new(vec![1.0, -3.0, 2.0], vec![3, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 1]);
                let mut roots = t.materialize_f64();
                roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert!((roots[0] - 1.0).abs() < 1e-10);
                assert!((roots[1] - 2.0).abs() < 1e-10);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[test]
    fn roots_typed_integer_coefficients_cross_double_boundary_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let coeffs = Tensor::new_integer(IntegerStorage::I16(vec![1, -3, 2]), vec![3, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[2, 1]);
                let mut roots = t.materialize_f64();
                roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert!((roots[0] - 1.0).abs() < 1e-10);
                assert!((roots[1] - 2.0).abs() < 1e-10);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_leading_zeros_trimmed() {
        let coeffs = Tensor::new(vec![0.0, 0.0, 1.0, -4.0], vec![4, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[1, 1]);
                assert!((t.materialize_f64()[0] - 4.0).abs() < 1e-10);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_complex_pair() {
        let coeffs = Tensor::new(vec![1.0, 0.0, 1.0], vec![3, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), &[2, 1]);
                let mut roots = t.materialize_f64();
                roots.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                assert!((roots[0].0).abs() < 1e-10);
                assert!((roots[0].1 + 1.0).abs() < 1e-10);
                assert!((roots[1].0).abs() < 1e-10);
                assert!((roots[1].1 - 1.0).abs() < 1e-10);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_quartic_all_zero_roots() {
        // p(x) = x^4 => 4 roots at 0
        let coeffs = Tensor::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![5, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots quartic");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[4, 1]);
                for &r in &t.materialize_f64() {
                    assert!(r.abs() < 1e-8);
                }
            }
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), &[4, 1]);
                for &(re, im) in &t.materialize_f64() {
                    assert!(re.abs() < 1e-7 && im.abs() < 1e-7);
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_accepts_complex_coefficients_input() {
        // p(x) = x^2 + 1 with complex coefficients path
        let coeffs =
            ComplexTensor::new(vec![(1.0, 0.0), (0.0, 0.0), (1.0, 0.0)], vec![3, 1]).unwrap();
        let result = roots_builtin(Value::ComplexTensor(coeffs)).expect("roots complex input");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), &[2, 1]);
                // roots at i and -i
                let mut roots = t.materialize_f64();
                roots.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                assert!(roots[0].0.abs() < 1e-10 && (roots[0].1 + 1.0).abs() < 1e-6);
                assert!(roots[1].0.abs() < 1e-10 && (roots[1].1 - 1.0).abs() < 1e-6);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_accepts_logical_coefficients() {
        // p(x) = x with logical coefficients [1 0]
        let la = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let result = roots_builtin(Value::LogicalArray(la)).expect("roots logical");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[1, 1]);
                assert!(t.materialize_f64()[0].abs() < 1e-12);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_scalar_num_returns_empty() {
        let result = roots_builtin(Value::Num(5.0)).expect("roots scalar num");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0, 1]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_rejects_non_vector_input() {
        let coeffs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err = roots_builtin(Value::Tensor(coeffs)).expect_err("expected vector-shape error");
        assert_eq!(err.identifier(), ROOTS_ERROR_INVALID_INPUT.identifier);
        assert_error_contains(err, "vector");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_all_zero_coefficients_returns_empty() {
        let coeffs = Tensor::new(vec![0.0, 0.0, 0.0], vec![3, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0, 1]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_gpu_input_gathers_to_host() {
        test_support::with_test_provider(|provider| {
            let coeffs = Tensor::new(vec![1.0, 0.0, -9.0, 0.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &coeffs.materialize_f64(),
                shape: coeffs.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = roots_builtin(Value::GpuTensor(handle)).expect("roots");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape(), &[3, 1]);
            let mut roots = gathered.materialize_f64();
            roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((roots[0] + 3.0).abs() < 1e-9);
            assert!((roots[1]).abs() < 1e-9);
            assert!((roots[2] - 3.0).abs() < 1e-9);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn roots_constant_polynomial_returns_empty() {
        let coeffs = Tensor::new(vec![5.0], vec![1, 1]).unwrap();
        let result = roots_builtin(Value::Tensor(coeffs)).expect("roots");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0, 1]);
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn tiny_nonzero_coefficients_and_roots_are_not_trimmed() {
        for (coefficients, expected) in [([1.0e-13, 1.0], -1.0e13), ([1.0, 1.0e-12], -1.0e-12)] {
            let value = Value::Tensor(Tensor::new(coefficients.to_vec(), vec![1, 2]).unwrap());
            let root = roots_builtin(value).expect("linear root");
            let Value::Tensor(root) = root else {
                panic!("real linear root expected");
            };
            assert_eq!(root.shape(), &[1, 1]);
            assert!((root.materialize_f64()[0] / expected - 1.0).abs() < 1.0e-14);
        }
        let complex = roots_builtin(Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 0.0), (0.0, 1.0e-13)], vec![1, 2]).unwrap(),
        ))
        .expect("complex linear root");
        let Value::ComplexTensor(complex) = complex else {
            panic!("nonzero imaginary root must remain complex");
        };
        assert_eq!(complex.materialize_f64(), vec![(0.0, -1.0e-13)]);

        let quadratic = roots_builtin(Value::Tensor(
            Tensor::new(vec![1.0, 0.0, 1.0e-26], vec![1, 3]).unwrap(),
        ))
        .expect("small complex quadratic roots");
        let Value::ComplexTensor(quadratic) = quadratic else {
            panic!("quadratic with negative discriminant must remain complex");
        };
        let roots = quadratic.materialize_f64();
        assert_eq!(roots.len(), 2);
        assert!(roots
            .iter()
            .all(|(re, im)| re.abs() < 1.0e-25 && im.abs() > 0.0));
        assert!((roots[0].1 + roots[1].1).abs() < 1.0e-24);
    }

    fn roots_builtin(coefficients: Value) -> BuiltinResult<Value> {
        block_on(super::roots_builtin(coefficients))
    }
}
