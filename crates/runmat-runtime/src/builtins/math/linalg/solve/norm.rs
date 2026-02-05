//! MATLAB-compatible `norm` builtin with GPU-aware semantics for RunMat.

use nalgebra::{DMatrix, SVD};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderNormOrder};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::numeric_scalar_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "norm";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::norm")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("norm")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(1024),
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Awaiting specialized kernels; RunMat gathers to host when providers omit the optional norm hook.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::norm")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes:
        "Norm is a terminal reduction; fusion currently delegates to the shared CPU implementation.",
};

#[runtime_builtin(
    name = "norm",
    category = "math/linalg/solve",
    summary = "Vector and matrix norms with MATLAB semantics.",
    keywords = "norm,vector norm,matrix norm,frobenius,nuclear,gpu",
    accel = "reduction",
    type_resolver(numeric_scalar_type),
    builtin_path = "crate::builtins::math::linalg::solve::norm"
)]
async fn norm_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let order = parse_order(&rest)?;
    match value {
        Value::GpuTensor(handle) => norm_gpu(handle, order).await,
        Value::ComplexTensor(tensor) => {
            let norm = norm_complex_tensor(&tensor, order)?;
            Ok(Value::Num(norm))
        }
        Value::Tensor(tensor) => {
            let norm = norm_real_tensor(&tensor, order)?;
            Ok(Value::Num(norm))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            let norm = norm_complex_tensor(&tensor, order)?;
            Ok(Value::Num(norm))
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            let norm = norm_real_tensor(&tensor, order)?;
            Ok(Value::Num(norm))
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum NormOrder {
    Default,
    One,
    Two,
    Inf,
    NegInf,
    Zero,
    Fro,
    Nuc,
    P(f64),
}

#[derive(Debug, Clone, Copy)]
enum TensorKind {
    Vector,
    Matrix { rows: usize, cols: usize },
}

async fn norm_gpu(handle: GpuTensorHandle, order: NormOrder) -> BuiltinResult<Value> {
    let maybe_provider = runmat_accelerate_api::provider();

    if let Some(provider) = maybe_provider {
        let provider_order = ProviderNormOrder::from(order);
        if let Ok(result) = provider.norm(&handle, provider_order).await {
            return Ok(Value::GpuTensor(result));
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(map_control_flow)?;
    let norm = norm_real_tensor(&tensor, order)?;

    if let Some(provider) = maybe_provider {
        match upload_scalar(provider, norm) {
            Ok(uploaded) => return Ok(Value::GpuTensor(uploaded)),
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(build_runtime_error("interaction pending...")
                        .with_builtin(NAME)
                        .build());
                }
            }
        }
    }

    Ok(Value::Num(norm))
}

fn norm_real_tensor(tensor: &Tensor, order: NormOrder) -> BuiltinResult<f64> {
    norm_real_tensor_impl(tensor, order)
}

fn norm_complex_tensor(tensor: &ComplexTensor, order: NormOrder) -> BuiltinResult<f64> {
    norm_complex_tensor_impl(tensor, order)
}

fn norm_real_tensor_impl(tensor: &Tensor, order: NormOrder) -> BuiltinResult<f64> {
    let kind = classify_tensor(&tensor.shape)?;
    let resolved = match order {
        NormOrder::Default => NormOrder::Two,
        other => other,
    };
    match kind {
        TensorKind::Vector => {
            let magnitudes: Vec<f64> = tensor.data.iter().map(|&v| v.abs()).collect();
            vector_norm_from_magnitudes(&magnitudes, resolved)
        }
        TensorKind::Matrix { rows, cols } => matrix_norm_real(tensor, rows, cols, resolved),
    }
}

fn norm_complex_tensor_impl(tensor: &ComplexTensor, order: NormOrder) -> BuiltinResult<f64> {
    let kind = classify_tensor(&tensor.shape)?;
    let resolved = match order {
        NormOrder::Default => NormOrder::Two,
        other => other,
    };
    match kind {
        TensorKind::Vector => {
            let magnitudes: Vec<f64> = tensor.data.iter().map(|&(re, im)| re.hypot(im)).collect();
            vector_norm_from_magnitudes(&magnitudes, resolved)
        }
        TensorKind::Matrix { rows, cols } => matrix_norm_complex(tensor, rows, cols, resolved),
    }
}

fn classify_tensor(shape: &[usize]) -> BuiltinResult<TensorKind> {
    if shape.is_empty() {
        return Ok(TensorKind::Vector);
    }

    if shape.len() > 2 && shape.iter().skip(2).any(|&d| d > 1) {
        return Err(builtin_error(format!(
            "{NAME}: input must be a vector or 2-D matrix."
        )));
    }

    let rows = shape.first().copied().unwrap_or(0);
    let cols = shape.get(1).copied().unwrap_or(1);

    if shape.len() == 1 || rows <= 1 || cols <= 1 {
        Ok(TensorKind::Vector)
    } else {
        Ok(TensorKind::Matrix { rows, cols })
    }
}

fn vector_norm_from_magnitudes(magnitudes: &[f64], order: NormOrder) -> BuiltinResult<f64> {
    if magnitudes.iter().any(|v| v.is_nan()) {
        return Ok(f64::NAN);
    }
    match order {
        NormOrder::Default => unreachable!("resolved in caller"),
        NormOrder::One => Ok(magnitudes.iter().sum()),
        NormOrder::Two | NormOrder::Fro => Ok(root_sum_of_squares(magnitudes)),
        NormOrder::Inf => Ok(magnitudes
            .iter()
            .fold(0.0, |acc, &v| if v > acc { v } else { acc })),
        NormOrder::NegInf => {
            if magnitudes.is_empty() {
                Ok(0.0)
            } else {
                let mut min_val = f64::INFINITY;
                for &v in magnitudes {
                    if v < min_val {
                        min_val = v;
                    }
                }
                if min_val == f64::INFINITY {
                    Ok(0.0)
                } else {
                    Ok(min_val)
                }
            }
        }
        NormOrder::Zero => {
            let mut count = 0.0;
            for &v in magnitudes {
                if v.is_nan() {
                    return Ok(f64::NAN);
                }
                if v != 0.0 {
                    count += 1.0;
                }
            }
            Ok(count)
        }
        NormOrder::Nuc => Err(builtin_error(format!(
            "{NAME}: nuclear norm is only defined for matrices."
        ))),
        NormOrder::P(p) => {
            if !p.is_finite() {
                return Err(builtin_error(format!("{NAME}: invalid norm order {p}")));
            }
            if p < 1.0 {
                return Err(builtin_error(format!(
                    "{NAME}: vector norm order {p} must satisfy p >= 1 (or use 0, Inf, or -Inf)."
                )));
            }
            if magnitudes.is_empty() {
                return Ok(0.0);
            }
            let sum: f64 = magnitudes.iter().map(|&v| v.powf(p)).sum();
            Ok(sum.powf(1.0 / p))
        }
    }
}

fn root_sum_of_squares(values: &[f64]) -> f64 {
    let mut scale = 0.0f64;
    let mut sumsq = 1.0f64;
    let mut count = 0usize;
    for &value in values {
        if value == 0.0 {
            continue;
        }
        let abs = value.abs();
        if abs.is_nan() {
            return f64::NAN;
        }
        if scale < abs {
            let ratio = if scale == 0.0 { 0.0 } else { scale / abs };
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = abs;
        } else {
            let ratio = abs / scale;
            sumsq += ratio * ratio;
        }
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        scale * sumsq.sqrt()
    }
}

fn matrix_norm_real(
    tensor: &Tensor,
    rows: usize,
    cols: usize,
    order: NormOrder,
) -> BuiltinResult<f64> {
    if tensor.data.iter().any(|v| v.is_nan()) {
        return Ok(f64::NAN);
    }
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    match order {
        NormOrder::Default => unreachable!("resolved in caller"),
        NormOrder::One => {
            let magnitudes: Vec<f64> = tensor.data.iter().map(|&v| v.abs()).collect();
            Ok(max_column_sum(&magnitudes, rows, cols))
        }
        NormOrder::Two => spectral_norm_real(tensor, rows, cols),
        NormOrder::Inf => {
            let magnitudes: Vec<f64> = tensor.data.iter().map(|&v| v.abs()).collect();
            Ok(max_row_sum(&magnitudes, rows, cols))
        }
        NormOrder::Fro => {
            let magnitudes: Vec<f64> = tensor.data.iter().map(|&v| v.abs()).collect();
            Ok(root_sum_of_squares(&magnitudes))
        }
        NormOrder::Nuc => nuclear_norm_real(tensor, rows, cols),
        NormOrder::Zero => Err(builtin_error(format!(
            "{NAME}: matrix norm order 0 is not supported; use 1, 2, Inf, 'fro', or 'nuc'."
        ))),
        NormOrder::NegInf => Err(builtin_error(format!(
            "{NAME}: matrix norm order -Inf is not supported; use 1, 2, Inf, 'fro', or 'nuc'."
        ))),
        NormOrder::P(p) => Err(builtin_error(format!(
            "{NAME}: matrix norm order {p} is not supported; use 1, 2, Inf, 'fro', or 'nuc'."
        ))),
    }
}

fn matrix_norm_complex(
    tensor: &ComplexTensor,
    rows: usize,
    cols: usize,
    order: NormOrder,
) -> BuiltinResult<f64> {
    if tensor
        .data
        .iter()
        .any(|&(re, im)| re.is_nan() || im.is_nan())
    {
        return Ok(f64::NAN);
    }
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    match order {
        NormOrder::Default => unreachable!("resolved in caller"),
        NormOrder::One => {
            let magnitudes: Vec<f64> = tensor.data.iter().map(|&(re, im)| re.hypot(im)).collect();
            Ok(max_column_sum(&magnitudes, rows, cols))
        }
        NormOrder::Two => spectral_norm_complex(tensor, rows, cols),
        NormOrder::Inf => {
            let magnitudes: Vec<f64> = tensor.data.iter().map(|&(re, im)| re.hypot(im)).collect();
            Ok(max_row_sum(&magnitudes, rows, cols))
        }
        NormOrder::Fro => {
            let magnitudes: Vec<f64> = tensor.data.iter().map(|&(re, im)| re.hypot(im)).collect();
            Ok(root_sum_of_squares(&magnitudes))
        }
        NormOrder::Nuc => nuclear_norm_complex(tensor, rows, cols),
        NormOrder::Zero => Err(builtin_error(format!(
            "{NAME}: matrix norm order 0 is not supported for complex inputs; use 1, 2, Inf, 'fro', or 'nuc'."
        ))),
        NormOrder::NegInf => Err(builtin_error(format!(
            "{NAME}: matrix norm order -Inf is not supported for complex inputs; use 1, 2, Inf, 'fro', or 'nuc'."
        ))),
        NormOrder::P(p) => Err(builtin_error(format!(
            "{NAME}: matrix norm order {p} is not supported for complex inputs; use 1, 2, Inf, 'fro', or 'nuc'."
        ))),
    }
}

fn max_column_sum(magnitudes: &[f64], rows: usize, cols: usize) -> f64 {
    let mut max_sum = 0.0;
    for c in 0..cols {
        let mut sum = 0.0;
        for r in 0..rows {
            sum += magnitudes[r + c * rows];
        }
        if sum.is_nan() {
            return f64::NAN;
        }
        if sum > max_sum {
            max_sum = sum;
        }
    }
    max_sum
}

fn max_row_sum(magnitudes: &[f64], rows: usize, cols: usize) -> f64 {
    let mut max_sum = 0.0;
    for r in 0..rows {
        let mut sum = 0.0;
        for c in 0..cols {
            sum += magnitudes[r + c * rows];
        }
        if sum.is_nan() {
            return f64::NAN;
        }
        if sum > max_sum {
            max_sum = sum;
        }
    }
    max_sum
}

fn spectral_norm_real(tensor: &Tensor, rows: usize, cols: usize) -> BuiltinResult<f64> {
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    let matrix = DMatrix::from_column_slice(rows, cols, &tensor.data);
    let svd = SVD::new(matrix, false, false);
    Ok(svd
        .singular_values
        .iter()
        .fold(0.0, |acc, &value| if value > acc { value } else { acc }))
}

fn spectral_norm_complex(tensor: &ComplexTensor, rows: usize, cols: usize) -> BuiltinResult<f64> {
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    let data: Vec<Complex64> = tensor
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let matrix = DMatrix::from_column_slice(rows, cols, &data);
    let svd = SVD::new(matrix, false, false);
    Ok(svd
        .singular_values
        .iter()
        .fold(0.0, |acc, &value| if value > acc { value } else { acc }))
}

fn nuclear_norm_real(tensor: &Tensor, rows: usize, cols: usize) -> BuiltinResult<f64> {
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    let matrix = DMatrix::from_column_slice(rows, cols, &tensor.data);
    let svd = SVD::new(matrix, false, false);
    Ok(svd.singular_values.iter().sum())
}

fn nuclear_norm_complex(tensor: &ComplexTensor, rows: usize, cols: usize) -> BuiltinResult<f64> {
    if rows == 0 || cols == 0 {
        return Ok(0.0);
    }
    let data: Vec<Complex64> = tensor
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let matrix = DMatrix::from_column_slice(rows, cols, &data);
    let svd = SVD::new(matrix, false, false);
    Ok(svd.singular_values.iter().sum())
}

fn parse_order(args: &[Value]) -> BuiltinResult<NormOrder> {
    match args.len() {
        0 => Ok(NormOrder::Default),
        1 => parse_order_value(&args[0]),
        _ => Err(builtin_error(format!(
            "{NAME}: expected a single optional norm order argument."
        ))),
    }
}

fn parse_order_value(value: &Value) -> BuiltinResult<NormOrder> {
    match value {
        Value::Num(n) => parse_numeric(*n),
        Value::Int(i) => parse_numeric(i.to_f64()),
        Value::Bool(b) => parse_numeric(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                parse_numeric(t.data[0])
            } else {
                Err(builtin_error(format!(
                    "{NAME}: norm order must be a scalar."
                )))
            }
        }
        Value::LogicalArray(l) => {
            if l.len() == 1 {
                let val = if l.data[0] != 0 { 1.0 } else { 0.0 };
                parse_numeric(val)
            } else {
                Err(builtin_error(format!(
                    "{NAME}: norm order must be a scalar logical value."
                )))
            }
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(builtin_error(format!(
            "{NAME}: norm order must be real-valued."
        ))),
        Value::GpuTensor(_) => Err(builtin_error(format!(
            "{NAME}: norm order cannot be a GPU-resident tensor."
        ))),
        _ => {
            if let Some(text) = tensor::value_to_string(value) {
                parse_order_string(&text)
            } else {
                Err(builtin_error(format!(
                    "{NAME}: unsupported norm order argument {value:?}"
                )))
            }
        }
    }
}

fn parse_numeric(raw: f64) -> BuiltinResult<NormOrder> {
    if raw.is_nan() {
        return Err(builtin_error(format!(
            "{NAME}: norm order must be a real scalar."
        )));
    }
    if raw.is_infinite() {
        return Ok(if raw.is_sign_positive() {
            NormOrder::Inf
        } else {
            NormOrder::NegInf
        });
    }
    if approx_eq(raw, 0.0) {
        return Ok(NormOrder::Zero);
    }
    if approx_eq(raw, 1.0) {
        return Ok(NormOrder::One);
    }
    if approx_eq(raw, 2.0) {
        return Ok(NormOrder::Two);
    }
    Ok(NormOrder::P(raw))
}

fn parse_order_string(raw: &str) -> BuiltinResult<NormOrder> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(builtin_error(format!(
            "{NAME}: norm order string cannot be empty."
        )));
    }
    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "fro" => Ok(NormOrder::Fro),
        "nuc" | "nuclear" => Ok(NormOrder::Nuc),
        "inf" => Ok(NormOrder::Inf),
        "-inf" => Ok(NormOrder::NegInf),
        _ => {
            if let Ok(value) = trimmed.parse::<f64>() {
                parse_numeric(value)
            } else {
                Err(builtin_error(format!(
                    "{NAME}: unrecognised norm order '{trimmed}'."
                )))
            }
        }
    }
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= f64::EPSILON * (a.abs() + b.abs() + 1.0)
}

fn upload_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: f64,
) -> BuiltinResult<GpuTensorHandle> {
    let data = [value];
    let shape = [1usize, 1usize];
    provider
        .upload(&HostTensorView {
            data: &data,
            shape: &shape,
        })
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

impl From<ProviderNormOrder> for NormOrder {
    fn from(value: ProviderNormOrder) -> Self {
        match value {
            ProviderNormOrder::Two => NormOrder::Two,
            ProviderNormOrder::One => NormOrder::One,
            ProviderNormOrder::Inf => NormOrder::Inf,
            ProviderNormOrder::NegInf => NormOrder::NegInf,
            ProviderNormOrder::Zero => NormOrder::Zero,
            ProviderNormOrder::Fro => NormOrder::Fro,
            ProviderNormOrder::Nuc => NormOrder::Nuc,
            ProviderNormOrder::P(p) => NormOrder::P(p),
        }
    }
}

impl From<NormOrder> for ProviderNormOrder {
    fn from(value: NormOrder) -> Self {
        match value {
            NormOrder::Default | NormOrder::Two => ProviderNormOrder::Two,
            NormOrder::One => ProviderNormOrder::One,
            NormOrder::Inf => ProviderNormOrder::Inf,
            NormOrder::NegInf => ProviderNormOrder::NegInf,
            NormOrder::Zero => ProviderNormOrder::Zero,
            NormOrder::Fro => ProviderNormOrder::Fro,
            NormOrder::Nuc => ProviderNormOrder::Nuc,
            NormOrder::P(p) => ProviderNormOrder::P(p),
        }
    }
}

/// Helper for provider backends that reuse the host implementation.
pub fn norm_host_real_for_provider(
    tensor: &Tensor,
    order: ProviderNormOrder,
) -> BuiltinResult<f64> {
    let resolved = NormOrder::from(order);
    norm_real_tensor_impl(tensor, resolved)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, ResolveContext, Type};
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    fn assert_close(actual: f64, expected: f64) {
        if actual.is_nan() && expected.is_nan() {
            return;
        }
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-10,
            "expected {expected}, got {actual} (diff {diff})"
        );
    }

    #[test]
    fn norm_type_returns_scalar() {
        let out = numeric_scalar_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_vector_default_two() {
        let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let value = norm_builtin(Value::Tensor(tensor), Vec::new()).expect("norm");
        match value {
            Value::Num(v) => assert_close(v, 5.0),
            other => panic!("expected scalar value, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_vector_fractional_p() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let value = norm_builtin(Value::Tensor(tensor), vec![Value::Num(1.5)]).expect("norm");
        match value {
            Value::Num(v) => {
                let expected = (1f64.powf(1.5) + 2f64.powf(1.5) + 3f64.powf(1.5)).powf(1.0 / 1.5);
                assert_close(v, expected);
            }
            other => panic!("expected scalar value, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_vector_infinity_and_negative_infinity() {
        let tensor = Tensor::new(vec![2.0, -7.0, 4.0], vec![3, 1]).unwrap();
        let inf = norm_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Num(f64::INFINITY)],
        )
        .expect("norm inf");
        let neg_inf = norm_builtin(Value::Tensor(tensor), vec![Value::Num(f64::NEG_INFINITY)])
            .expect("norm -inf");
        assert_close(
            match inf {
                Value::Num(v) => v,
                _ => panic!("expected scalar"),
            },
            7.0,
        );
        assert_close(
            match neg_inf {
                Value::Num(v) => v,
                _ => panic!("expected scalar"),
            },
            2.0,
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_vector_zero_norm_counts_nonzeros() {
        let tensor = Tensor::new(vec![0.0, 0.0, 5.0, 0.0], vec![4, 1]).unwrap();
        let value = norm_builtin(Value::Tensor(tensor), vec![Value::Num(0.0)]).expect("norm");
        match value {
            Value::Num(v) => assert_close(v, 1.0),
            _ => panic!("expected scalar"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_vector_p_less_than_one_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err =
            unwrap_error(norm_builtin(Value::Tensor(tensor), vec![Value::Num(0.5)]).unwrap_err());
        assert!(
            err.message().contains("p >= 1"),
            "expected p >= 1 error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_vector_nuclear_norm_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = unwrap_error(
            norm_builtin(Value::Tensor(tensor), vec![Value::from("nuc")]).unwrap_err(),
        );
        assert!(
            err.message().contains("only defined for matrices"),
            "expected matrix-only message, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_matrix_fro_and_nuclear() {
        let tensor = Tensor::new(vec![2.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let fro =
            norm_builtin(Value::Tensor(tensor.clone()), vec![Value::from("fro")]).expect("fro");
        let nuc =
            norm_builtin(Value::Tensor(tensor.clone()), vec![Value::from("nuc")]).expect("nuc");
        assert_close(
            match fro {
                Value::Num(v) => v,
                _ => panic!("expected scalar"),
            },
            (2.0f64.powi(2) + 1.0).sqrt(),
        );
        assert_close(
            match nuc {
                Value::Num(v) => v,
                _ => panic!("expected scalar"),
            },
            3.0,
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_matrix_two_matches_spectral_radius() {
        let tensor = Tensor::new(vec![3.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let value = norm_builtin(Value::Tensor(tensor), Vec::new()).expect("norm");
        match value {
            Value::Num(v) => assert_close(v, 3.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_matrix_invalid_order_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err =
            unwrap_error(norm_builtin(Value::Tensor(tensor), vec![Value::Num(3.0)]).unwrap_err());
        assert!(
            err.message().contains("not supported"),
            "expected unsupported message, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_order_accepts_boolean_scalar() {
        let tensor = Tensor::new(vec![2.0, -3.0], vec![2, 1]).unwrap();
        let value = norm_builtin(Value::Tensor(tensor), vec![Value::Bool(true)]).expect("norm");
        match value {
            Value::Num(v) => assert_close(v, 5.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_order_logical_scalar_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let logical = runmat_builtins::LogicalArray::new(vec![1], vec![1]).expect("logical scalar");
        let value =
            norm_builtin(Value::Tensor(tensor), vec![Value::LogicalArray(logical)]).expect("norm");
        match value {
            Value::Num(v) => assert_close(v, 3.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_order_char_array_inf() {
        let tensor = Tensor::new(vec![2.0, -7.0, 4.0], vec![3, 1]).unwrap();
        let chars = CharArray::new("Inf".chars().collect(), 1, 3).unwrap();
        let value =
            norm_builtin(Value::Tensor(tensor), vec![Value::CharArray(chars)]).expect("norm");
        match value {
            Value::Num(v) => assert_close(v, 7.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_order_tensor_non_scalar_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let order = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err = unwrap_error(
            norm_builtin(Value::Tensor(tensor), vec![Value::Tensor(order)]).unwrap_err(),
        );
        assert!(
            err.message().contains("scalar"),
            "expected scalar error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_higher_dimensional_tensor_errors() {
        let data: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        let tensor = Tensor::new(data, vec![2, 2, 2]).unwrap();
        let err = unwrap_error(norm_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err());
        assert!(
            err.message().contains("vector or 2-D matrix"),
            "expected dimensionality error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_complex_vector() {
        let tensor = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![2, 1]).unwrap();
        let value = norm_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("norm");
        match value {
            Value::Num(v) => assert_close(v, (1.0f64 + 4.0 + 9.0 + 16.0).sqrt()),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_complex_matrix_nuclear() {
        let tensor = ComplexTensor::new(
            vec![(2.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let value =
            norm_builtin(Value::ComplexTensor(tensor), vec![Value::from("nuc")]).expect("norm");
        match value {
            Value::Num(v) => assert_close(v, 3.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_empty_returns_zero() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let value = norm_builtin(Value::Tensor(tensor), Vec::new()).expect("norm");
        match value {
            Value::Num(v) => assert_eq!(v, 0.0),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = norm_builtin(Value::GpuTensor(handle), Vec::new()).expect("norm");
            let gathered = test_support::gather(result).expect("gather");
            assert_close(gathered.data[0], 5.0);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn norm_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![3.0, 0.0, 0.0, 4.0], vec![2, 2]).unwrap();
        let cpu = norm_real_tensor(&tensor, NormOrder::Default).expect("cpu norm");

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");

        let result = norm_builtin(Value::GpuTensor(handle), Vec::new()).expect("norm");
        let gathered = test_support::gather(result).expect("gather");
        assert_close(gathered.data[0], cpu);
    }

    fn norm_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::norm_builtin(value, rest))
    }
}
