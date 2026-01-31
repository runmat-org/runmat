//! MATLAB-compatible `deconv` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::type_resolvers::deconv_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::deconv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "deconv",
    op_kind: GpuOpKind::Custom("deconv1d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "When providers lack native deconvolution kernels, RunMat gathers inputs to the host and re-uploads real-valued outputs to maintain GPU residency.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::deconv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "deconv",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Polynomial division is not part of current fusion pipelines; metadata is present for completeness.",
};

const BUILTIN_NAME: &str = "deconv";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "deconv",
    category = "math/signal",
    summary = "Polynomial (1-D sequence) deconvolution returning quotient and remainder.",
    keywords = "deconv,deconvolution,polynomial division,signal,gpu",
    accel = "custom",
    type_resolver(deconv_type),
    builtin_path = "crate::builtins::math::signal::deconv"
)]
async fn deconv_builtin(numerator: Value, denominator: Value) -> crate::BuiltinResult<Value> {
    let eval = evaluate(numerator, denominator).await?;
    Ok(eval.quotient())
}

/// Evaluate `deconv` and retain both outputs for multi-value contexts.
pub async fn evaluate(numerator: Value, denominator: Value) -> BuiltinResult<DeconvEval> {
    let (num_input, mut prefer_gpu) = convert_value(numerator).await?;
    let (den_input, den_gpu) = convert_value(denominator).await?;
    prefer_gpu |= den_gpu;

    let (quotient_raw, remainder_raw) = polynomial_division(&num_input.data, &den_input.data)?;

    let orientation = orientation_from_hint(num_input.hint);
    let quotient = convert_output(quotient_raw, orientation, prefer_gpu)?;
    let remainder = convert_output(remainder_raw, orientation, prefer_gpu)?;

    Ok(DeconvEval {
        quotient,
        remainder,
    })
}

/// Evaluation envelope used by both builtin and bytecode multi-output paths.
#[derive(Clone)]
pub struct DeconvEval {
    quotient: Value,
    remainder: Value,
}

impl DeconvEval {
    /// Quotient polynomial (`q`).
    pub fn quotient(&self) -> Value {
        self.quotient.clone()
    }

    /// Remainder polynomial (`r`).
    pub fn remainder(&self) -> Value {
        self.remainder.clone()
    }
}

#[derive(Clone)]
struct PolyInput {
    data: Vec<Complex<f64>>,
    hint: OrientationHint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrientationHint {
    Row,
    Column,
    Scalar,
    General,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Row,
    Column,
}

#[async_recursion::async_recursion(?Send)]
async fn convert_value(value: Value) -> BuiltinResult<(PolyInput, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            let (input, _) = convert_value(gathered).await?;
            Ok((input, true))
        }
        Value::Tensor(tensor) => convert_tensor(tensor).map(|input| (input, false)),
        Value::ComplexTensor(tensor) => convert_complex_tensor(tensor).map(|input| (input, false)),
        Value::LogicalArray(logical) => convert_logical_array(logical).map(|input| (input, false)),
        Value::Num(n) => Ok((
            PolyInput {
                data: vec![Complex::new(n, 0.0)],
                hint: OrientationHint::Scalar,
            },
            false,
        )),
        Value::Int(int_val) => {
            let num = int_val.to_f64();
            Ok((
                PolyInput {
                    data: vec![Complex::new(num, 0.0)],
                    hint: OrientationHint::Scalar,
                },
                false,
            ))
        }
        Value::Bool(flag) => Ok((
            PolyInput {
                data: vec![Complex::new(if flag { 1.0 } else { 0.0 }, 0.0)],
                hint: OrientationHint::Scalar,
            },
            false,
        )),
        Value::Complex(re, im) => Ok((
            PolyInput {
                data: vec![Complex::new(re, im)],
                hint: OrientationHint::Scalar,
            },
            false,
        )),
        other => Err(runtime_error_for(format!(
            "deconv: unsupported input type {other:?}"
        ))),
    }
}

fn convert_tensor(tensor: Tensor) -> BuiltinResult<PolyInput> {
    let Tensor {
        data, rows, cols, ..
    } = tensor;
    let len = data.len();
    let hint = classify_orientation(rows, cols, len);
    ensure_vector(hint)?;
    let data = data.into_iter().map(|re| Complex::new(re, 0.0)).collect();
    Ok(PolyInput { data, hint })
}

fn convert_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<PolyInput> {
    let ComplexTensor {
        data, rows, cols, ..
    } = tensor;
    let len = data.len();
    let hint = classify_orientation(rows, cols, len);
    ensure_vector(hint)?;
    let data = data
        .into_iter()
        .map(|(re, im)| Complex::new(re, im))
        .collect();
    Ok(PolyInput { data, hint })
}

fn convert_logical_array(array: LogicalArray) -> BuiltinResult<PolyInput> {
    let hint = classify_orientation(
        array.shape.first().copied().unwrap_or(0),
        array.shape.get(1).copied().unwrap_or(0),
        array.data.len(),
    );
    ensure_vector(hint)?;
    let data = array
        .data
        .into_iter()
        .map(|bit| Complex::new(if bit != 0 { 1.0 } else { 0.0 }, 0.0))
        .collect();
    Ok(PolyInput { data, hint })
}

fn ensure_vector(hint: OrientationHint) -> BuiltinResult<()> {
    if matches!(hint, OrientationHint::General) {
        Err(runtime_error_for(
            "deconv: inputs must be scalars, row vectors, or column vectors",
        ))
    } else {
        Ok(())
    }
}

fn classify_orientation(rows: usize, cols: usize, len: usize) -> OrientationHint {
    if len == 0 {
        OrientationHint::Empty
    } else if rows == 1 && cols == 1 {
        OrientationHint::Scalar
    } else if rows == 1 {
        OrientationHint::Row
    } else if cols == 1 {
        OrientationHint::Column
    } else {
        OrientationHint::General
    }
}

fn orientation_from_hint(hint: OrientationHint) -> Orientation {
    match hint {
        OrientationHint::Column => Orientation::Column,
        OrientationHint::Row | OrientationHint::Scalar | OrientationHint::Empty => Orientation::Row,
        OrientationHint::General => Orientation::Column,
    }
}

type PolyDivision = (Vec<Complex<f64>>, Vec<Complex<f64>>);

fn polynomial_division(
    numerator: &[Complex<f64>],
    denominator: &[Complex<f64>],
) -> BuiltinResult<PolyDivision> {
    if denominator.is_empty() {
        return Err(runtime_error_for("denominator must not be empty"));
    }

    let (den_trim, _) = trim_leading_zeros(denominator);
    if den_trim.is_empty() {
        return Err(runtime_error_for(
            "denominator must contain at least one non-zero coefficient",
        ));
    }

    let (num_trim, num_all_zero) = trim_leading_zeros(numerator);
    if num_all_zero {
        return Ok((vec![Complex::new(0.0, 0.0)], vec![Complex::new(0.0, 0.0)]));
    }

    if num_trim.len() < den_trim.len() {
        let remainder = strip_leading_zeros(numerator.to_vec());
        return Ok((vec![Complex::new(0.0, 0.0)], remainder));
    }

    let divisor_lead = den_trim[0];
    if is_close_zero(&divisor_lead) {
        return Err(runtime_error_for(
            "denominator leading coefficient underflowed to zero",
        ));
    }

    let q_len = num_trim.len() - den_trim.len() + 1;
    let mut quotient = vec![Complex::new(0.0, 0.0); q_len];
    let mut working = num_trim.clone();

    for k in 0..q_len {
        let coeff = working[k] / divisor_lead;
        quotient[k] = coeff;
        if !is_close_zero(&coeff) {
            for j in 0..den_trim.len() {
                working[k + j] -= coeff * den_trim[j];
            }
        }
    }

    let mut remainder = if den_trim.len() > 1 {
        working[q_len..].to_vec()
    } else {
        Vec::new()
    };

    quotient = strip_leading_zeros(quotient);
    remainder = strip_leading_zeros(remainder);
    if remainder.is_empty() {
        remainder.push(Complex::new(0.0, 0.0));
    }

    Ok((quotient, remainder))
}

fn trim_leading_zeros(data: &[Complex<f64>]) -> (Vec<Complex<f64>>, bool) {
    if data.is_empty() {
        return (Vec::new(), true);
    }
    let first_non_zero = data.iter().position(|c| !is_close_zero(c));
    match first_non_zero {
        Some(idx) => {
            let trimmed = data[idx..].to_vec();
            (trimmed, false)
        }
        None => (Vec::new(), true),
    }
}

fn strip_leading_zeros(mut data: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let first_non_zero = data.iter().position(|c| !is_close_zero(c));
    match first_non_zero {
        Some(idx) if idx > 0 => data.drain(0..idx).for_each(drop),
        None => return vec![Complex::new(0.0, 0.0)],
        _ => {}
    }
    if data.is_empty() {
        vec![Complex::new(0.0, 0.0)]
    } else {
        data
    }
}

fn is_close_zero(value: &Complex<f64>) -> bool {
    value.norm_sqr() <= EPS * EPS
}

fn convert_output(
    data: Vec<Complex<f64>>,
    orientation: Orientation,
    prefer_gpu: bool,
) -> BuiltinResult<Value> {
    if data.is_empty() {
        return match orientation {
            Orientation::Row => finalize_real(vec![], vec![1, 0], prefer_gpu),
            Orientation::Column => finalize_real(vec![], vec![0, 1], prefer_gpu),
        };
    }

    let all_real = data.iter().all(|c| c.im.abs() <= EPS);
    let len = data.len();
    let shape = match (orientation, len) {
        (Orientation::Row, 1) | (Orientation::Column, 1) => vec![1, 1],
        (Orientation::Row, _) => vec![1, len],
        (Orientation::Column, _) => vec![len, 1],
    };

    if all_real {
        let real_data: Vec<f64> = data.into_iter().map(|c| c.re).collect();
        finalize_real(real_data, shape, prefer_gpu)
    } else if len == 1 {
        let Complex { re, im } = data[0];
        Ok(Value::Complex(re, im))
    } else {
        let complex_data: Vec<(f64, f64)> = data.into_iter().map(|c| (c.re, c.im)).collect();
        let tensor = ComplexTensor::new(complex_data, shape).map_err(|e| {
            runtime_error_for(format!("deconv: failed to build complex tensor: {e}"))
        })?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn finalize_real(data: Vec<f64>, shape: Vec<usize>, prefer_gpu: bool) -> BuiltinResult<Value> {
    let tensor = Tensor::new(data, shape.clone())
        .map_err(|e| runtime_error_for(format!("deconv: failed to build tensor: {e}")))?;
    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::Type;

    fn error_message(error: RuntimeError) -> String {
        error.message().to_string()
    }

    fn evaluate(numerator: Value, denominator: Value) -> BuiltinResult<DeconvEval> {
        block_on(super::evaluate(numerator, denominator))
    }

    #[test]
    fn deconv_type_uses_numerator_orientation() {
        let out = deconv_type(&[
            Type::Tensor {
                shape: Some(vec![Some(1), Some(5)]),
            },
            Type::Tensor {
                shape: Some(vec![Some(1), Some(3)]),
            },
        ]);
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), Some(3)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_exact_division() {
        let numerator = Tensor::new(vec![1.0, 3.0, 3.0, 1.0], vec![1, 4]).unwrap();
        let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let value =
            deconv_builtin(Value::Tensor(numerator), Value::Tensor(denominator)).expect("deconv");
        match value {
            Value::Tensor(q) => {
                assert_eq!(q.shape, vec![1, 3]);
                assert_eq!(q.data, vec![1.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor quotient, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_with_remainder() {
        let numerator = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
        let denominator = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(numerator), Value::Tensor(denominator)).expect("evaluate");
        let quotient = real_vector(eval.quotient());
        assert_eq!(quotient, vec![1.0, 2.0]);
        let remainder = real_vector(eval.remainder());
        assert_eq!(remainder, vec![3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_denominator_longer() {
        let numerator = Tensor::new(vec![3.0, 5.0], vec![1, 2]).unwrap();
        let denominator = Tensor::new(vec![1.0, 0.0, 2.0], vec![1, 3]).unwrap();
        let eval =
            evaluate(Value::Tensor(numerator), Value::Tensor(denominator)).expect("evaluate");
        let quotient = real_vector(eval.quotient());
        assert_eq!(quotient, vec![0.0]);
        let remainder = real_vector(eval.remainder());
        assert_eq!(remainder, vec![3.0, 5.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_leading_zeros() {
        let numerator = Tensor::new(vec![0.0, 0.0, 1.0, 2.0], vec![1, 4]).unwrap();
        let denominator = Tensor::new(vec![0.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let eval =
            evaluate(Value::Tensor(numerator), Value::Tensor(denominator)).expect("evaluate");
        let quotient = real_vector(eval.quotient());
        assert_eq!(quotient, vec![1.0]);
        let remainder = real_vector(eval.remainder());
        assert_eq!(remainder, vec![1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_complex_coefficients() {
        let numerator = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0), (2.0, 0.0)], vec![1, 3]).unwrap(),
        );
        let denominator = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, -1.0), (2.0, 1.0)], vec![1, 2]).unwrap(),
        );

        let eval = evaluate(numerator, denominator).expect("evaluate");
        match eval.quotient() {
            Value::ComplexTensor(q) => {
                assert_eq!(q.data.len(), 2);
            }
            other => panic!("unexpected quotient {other:?}"),
        }
        match eval.remainder() {
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                // Accept either scalar complex or tensor form depending on trimming.
            }
            Value::Tensor(r) => {
                assert!(r.data.iter().all(|v| v.abs() <= 1e-9));
            }
            other => panic!("unexpected remainder {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_reconstructs_original() {
        let numerator = vec![1.0, -3.0, 3.0, -1.0];
        let denominator = vec![1.0, -1.0];
        let eval = evaluate(
            Value::Tensor(Tensor::new(numerator.clone(), vec![1, 4]).unwrap()),
            Value::Tensor(Tensor::new(denominator.clone(), vec![1, 2]).unwrap()),
        )
        .expect("evaluate");

        let quotient = match eval.quotient() {
            Value::Tensor(t) => t.data,
            other => panic!("unexpected quotient {other:?}"),
        };
        let remainder = real_vector(eval.remainder());

        let reconstructed = add_polynomials(&convolve(&denominator, &quotient), &remainder);

        assert!(
            reconstructed
                .iter()
                .zip(numerator.iter())
                .all(|(a, b)| (a - b).abs() <= 1e-8),
            "reconstructed {:?} != {:?}",
            reconstructed,
            numerator
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_denominator_zero_error() {
        let numerator = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let denominator = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let err = error_message(
            deconv_builtin(Value::Tensor(numerator), Value::Tensor(denominator)).unwrap_err(),
        );
        assert!(err.contains("denominator"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_rejects_matrix_inputs() {
        let numerator = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let err = error_message(
            deconv_builtin(Value::Tensor(numerator), Value::Tensor(denominator)).unwrap_err(),
        );
        assert!(err.contains("vectors"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let numerator = Tensor::new(vec![1.0, 3.0, 3.0, 1.0], vec![1, 4]).unwrap();
            let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
            let n_view = HostTensorView {
                data: &numerator.data,
                shape: &numerator.shape,
            };
            let d_view = HostTensorView {
                data: &denominator.data,
                shape: &denominator.shape,
            };
            let n_handle = provider.upload(&n_view).expect("upload numerator");
            let d_handle = provider.upload(&d_view).expect("upload denominator");

            let eval =
                evaluate(Value::GpuTensor(n_handle), Value::GpuTensor(d_handle)).expect("evaluate");

            match eval.quotient() {
                Value::GpuTensor(handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(handle)).expect("gather quotient");
                    assert_eq!(gathered.data, vec![1.0, 2.0, 1.0]);
                }
                other => panic!("expected GPU quotient, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn deconv_wgpu_matches_cpu() {
        register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");

        let numerator = Tensor::new(vec![1.0, -2.0, 3.0, -4.0, 5.0], vec![1, 5]).unwrap();
        let denominator = Tensor::new(vec![1.0, -1.0, 2.0], vec![1, 3]).unwrap();

        let cpu_eval = evaluate(
            Value::Tensor(numerator.clone()),
            Value::Tensor(denominator.clone()),
        )
        .expect("cpu evaluate");
        let cpu_q = real_vector(cpu_eval.quotient());
        let cpu_r = real_vector(cpu_eval.remainder());

        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let num_handle = provider
            .upload(&HostTensorView {
                data: &numerator.data,
                shape: &numerator.shape,
            })
            .expect("upload numerator");
        let den_handle = provider
            .upload(&HostTensorView {
                data: &denominator.data,
                shape: &denominator.shape,
            })
            .expect("upload denominator");

        let gpu_eval = evaluate(Value::GpuTensor(num_handle), Value::GpuTensor(den_handle))
            .expect("gpu evaluate");
        let gpu_q = real_vector(gpu_eval.quotient());
        let gpu_r = real_vector(gpu_eval.remainder());

        assert_eq!(gpu_q.len(), cpu_q.len());
        assert_eq!(gpu_r.len(), cpu_r.len());
        for (a, b) in gpu_q.iter().zip(cpu_q.iter()) {
            assert!((a - b).abs() <= 1e-10, "gpu quotient {a} != cpu {b}");
        }
        for (a, b) in gpu_r.iter().zip(cpu_r.iter()) {
            assert!((a - b).abs() <= 1e-10, "gpu remainder {a} != cpu {b}");
        }
    }

    fn real_vector(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(t) => t.data,
            Value::Num(n) => vec![n],
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle)).expect("gather gpu output");
                gathered.data
            }
            Value::Complex(re, im) => {
                assert!(im.abs() <= 1e-9);
                vec![re]
            }
            Value::ComplexTensor(t) => {
                assert!(t.data.iter().all(|(_, im)| im.abs() <= 1e-9));
                t.data.into_iter().map(|(re, _)| re).collect()
            }
            other => panic!("expected real-valued tensor, got {other:?}"),
        }
    }

    fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                out[i + j] += ai * bj;
            }
        }
        out
    }

    fn add_polynomials(a: &[f64], b: &[f64]) -> Vec<f64> {
        let len = a.len().max(b.len());
        let mut out = vec![0.0; len];
        for (i, &v) in a.iter().rev().enumerate() {
            let idx = len - 1 - i;
            out[idx] += v;
        }
        for (i, &v) in b.iter().rev().enumerate() {
            let idx = len - 1 - i;
            out[idx] += v;
        }
        out
    }

    fn deconv_builtin(numerator: Value, denominator: Value) -> BuiltinResult<Value> {
        block_on(super::deconv_builtin(numerator, denominator))
    }
}
