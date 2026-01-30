//! MATLAB-compatible `eq` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor, type_shapes::logical_binary_type};
use crate::{build_runtime_error, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::rel::eq")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "eq",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_eq",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Prefers provider elem_eq kernels when available; otherwise inputs gather to host tensors automatically.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::rel::eq")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "eq",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let lhs = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let rhs = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0", "1.0"),
                ScalarType::F64 => ("f64(0.0)", "f64(1.0)"),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            Ok(format!("select({zero}, {one}, ({lhs} == {rhs}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits comparison kernels that write 0 or 1; providers may override with specialised shaders.",
};

const BUILTIN_NAME: &str = "eq";
const IDENT_INVALID_INPUT: &str = "MATLAB:eq:InvalidInput";
const IDENT_SIZE_MISMATCH: &str = "MATLAB:eq:SizeMismatch";

fn eq_error(message: impl Into<String>, identifier: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(identifier)
        .build()
}

#[runtime_builtin(
    name = "eq",
    category = "logical/rel",
    summary = "Element-wise equality comparison for scalars, arrays, and gpuArray inputs.",
    keywords = "eq,equality,comparison,logical,gpu",
    accel = "elementwise",
    type_resolver(logical_binary_type),
    builtin_path = "crate::builtins::logical::rel::eq"
)]
async fn eq_builtin(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if let (Value::GpuTensor(ref a), Value::GpuTensor(ref b)) = (&lhs, &rhs) {
        if let Some(result) = try_eq_gpu(a, b).await {
            return result;
        }
    }
    eq_host(lhs, rhs).await
}

async fn try_eq_gpu(
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
) -> Option<crate::BuiltinResult<Value>> {
    let provider = runmat_accelerate_api::provider()?;
    match provider.elem_eq(a, b).await {
        Ok(handle) => Some(Ok(gpu_helpers::logical_gpu_value(handle))),
        Err(err) => {
            drop(err);
            None
        }
    }
}

async fn eq_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if let Some(value) = eq_identity(&lhs, &rhs) {
        return Ok(value);
    }

    let (lhs, rhs) = normalize_char_string(lhs, rhs);

    if let Some(result) = scalar_eq_value(&lhs, &rhs) {
        return result;
    }

    let left = EqOperand::from_value(lhs).await?;
    let right = EqOperand::from_value(rhs).await?;

    match (left, right) {
        (EqOperand::Numeric(a), EqOperand::Numeric(b)) => {
            let (data, shape) = numeric_eq(&a, &b)?;
            logical_result(data, shape)
        }
        (EqOperand::Numeric(a), EqOperand::Complex(b)) => {
            let promoted = promote_numeric_to_complex(&a);
            let (data, shape) = complex_eq(&promoted, &b)?;
            logical_result(data, shape)
        }
        (EqOperand::Complex(a), EqOperand::Numeric(b)) => {
            let promoted = promote_numeric_to_complex(&b);
            let (data, shape) = complex_eq(&a, &promoted)?;
            logical_result(data, shape)
        }
        (EqOperand::Complex(a), EqOperand::Complex(b)) => {
            let (data, shape) = complex_eq(&a, &b)?;
            logical_result(data, shape)
        }
        (EqOperand::String(a), EqOperand::String(b)) => {
            let (data, shape) = string_eq(&a, &b)?;
            logical_result(data, shape)
        }
        (EqOperand::Numeric(_), EqOperand::String(_))
        | (EqOperand::Complex(_), EqOperand::String(_))
        | (EqOperand::String(_), EqOperand::Numeric(_))
        | (EqOperand::String(_), EqOperand::Complex(_)) => Err(eq_error(
            "eq: mixing numeric and string inputs is not supported",
            IDENT_INVALID_INPUT,
        )),
    }
}

fn scalar_numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => t.data.first().copied(),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            Some(ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        _ => None,
    }
}

fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(re, im) => Some((*re, *im)),
        Value::ComplexTensor(ct) if ct.data.len() == 1 => ct.data.first().copied(),
        _ => None,
    }
}

fn scalar_string_value(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data.first().cloned(),
        _ => None,
    }
}

fn scalar_eq_value(lhs: &Value, rhs: &Value) -> Option<crate::BuiltinResult<Value>> {
    let left_string = scalar_string_value(lhs);
    let right_string = scalar_string_value(rhs);
    if left_string.is_some() || right_string.is_some() {
        let left = left_string?;
        let right = right_string?;
        return Some(Ok(Value::Bool(left == right)));
    }

    let left = scalar_complex_value(lhs).or_else(|| scalar_numeric_value(lhs).map(|v| (v, 0.0)))?;
    let right =
        scalar_complex_value(rhs).or_else(|| scalar_numeric_value(rhs).map(|v| (v, 0.0)))?;
    let (ar, ai) = left;
    let (br, bi) = right;
    Some(Ok(Value::Bool(ar == br && ai == bi)))
}

fn eq_identity(lhs: &Value, rhs: &Value) -> Option<Value> {
    match (handle_ptr(lhs), handle_ptr(rhs)) {
        (Some(a), Some(b)) => Some(Value::Bool(a == b)),
        (Some(_), None) | (None, Some(_)) => Some(Value::Bool(false)),
        (None, None) => None,
    }
}

fn handle_ptr(value: &Value) -> Option<usize> {
    match value {
        Value::HandleObject(handle) => Some(unsafe { handle.target.as_raw() } as usize),
        Value::Listener(listener) => Some(unsafe { listener.target.as_raw() } as usize),
        _ => None,
    }
}

fn normalize_char_string(lhs: Value, rhs: Value) -> (Value, Value) {
    match (lhs, rhs) {
        (Value::CharArray(ca), Value::String(s)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::String(text), Value::String(s))
        }
        (Value::String(s), Value::CharArray(ca)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::String(s), Value::String(text))
        }
        (Value::CharArray(ca), Value::StringArray(sa)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::String(text), Value::StringArray(sa))
        }
        (Value::StringArray(sa), Value::CharArray(ca)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::StringArray(sa), Value::String(text))
        }
        (lhs, rhs) => (lhs, rhs),
    }
}

fn logical_result(data: Vec<u8>, shape: Vec<usize>) -> crate::BuiltinResult<Value> {
    if tensor::element_count(&shape) <= 1 && data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| eq_error(format!("eq: {e}"), IDENT_INVALID_INPUT))
    }
}

#[derive(Debug)]
struct NumericBuffer {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl NumericBuffer {
    fn scalar(value: f64) -> Self {
        Self {
            data: vec![value],
            shape: vec![1, 1],
        }
    }

    fn from_tensor(tensor: Tensor) -> Self {
        Self {
            data: tensor.data,
            shape: tensor.shape,
        }
    }

    fn from_logical(array: LogicalArray) -> Self {
        let shape = array.shape.clone();
        let data = array
            .data
            .into_iter()
            .map(|b| if b != 0 { 1.0 } else { 0.0 })
            .collect();
        Self { data, shape }
    }

    fn from_char_array(array: CharArray) -> Self {
        let rows = array.rows;
        let cols = array.cols;
        if rows == 0 || cols == 0 {
            return Self {
                data: Vec::new(),
                shape: vec![rows, cols],
            };
        }
        let mut data = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for r in 0..rows {
                let idx = r * cols + c;
                let ch = array.data[idx];
                data.push(ch as u32 as f64);
            }
        }
        Self {
            data,
            shape: vec![rows, cols],
        }
    }
}

#[derive(Debug)]
struct ComplexBuffer {
    data: Vec<(f64, f64)>,
    shape: Vec<usize>,
}

impl ComplexBuffer {
    fn scalar(re: f64, im: f64) -> Self {
        Self {
            data: vec![(re, im)],
            shape: vec![1, 1],
        }
    }

    fn from_tensor(tensor: ComplexTensor) -> Self {
        Self {
            data: tensor.data,
            shape: tensor.shape,
        }
    }
}

#[derive(Debug)]
struct StringBuffer {
    data: Vec<String>,
    shape: Vec<usize>,
}

impl StringBuffer {
    fn scalar(value: String) -> Self {
        Self {
            data: vec![value],
            shape: vec![1, 1],
        }
    }

    fn from_array(array: StringArray) -> Self {
        let StringArray { data, shape, .. } = array;
        Self { data, shape }
    }
}

enum EqOperand {
    Numeric(NumericBuffer),
    Complex(ComplexBuffer),
    String(StringBuffer),
}

impl EqOperand {
    async fn from_value(value: Value) -> crate::BuiltinResult<Self> {
        match value {
            Value::Num(n) => Ok(EqOperand::Numeric(NumericBuffer::scalar(n))),
            Value::Bool(flag) => Ok(EqOperand::Numeric(NumericBuffer::scalar(if flag {
                1.0
            } else {
                0.0
            }))),
            Value::Int(i) => Ok(EqOperand::Numeric(NumericBuffer::scalar(i.to_f64()))),
            Value::Tensor(tensor) => Ok(EqOperand::Numeric(NumericBuffer::from_tensor(tensor))),
            Value::LogicalArray(array) => {
                Ok(EqOperand::Numeric(NumericBuffer::from_logical(array)))
            }
            Value::Complex(re, im) => Ok(EqOperand::Complex(ComplexBuffer::scalar(re, im))),
            Value::ComplexTensor(tensor) => {
                Ok(EqOperand::Complex(ComplexBuffer::from_tensor(tensor)))
            }
            Value::String(s) => Ok(EqOperand::String(StringBuffer::scalar(s))),
            Value::StringArray(sa) => Ok(EqOperand::String(StringBuffer::from_array(sa))),
            Value::CharArray(array) => {
                Ok(EqOperand::Numeric(NumericBuffer::from_char_array(array)))
            }
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|err| {
                        eq_error(format!("{BUILTIN_NAME}: {err}"), IDENT_INVALID_INPUT)
                    })?;
                Ok(EqOperand::Numeric(NumericBuffer::from_tensor(tensor)))
            }
            unsupported => Err(eq_error(
                format!("eq: unsupported input type {unsupported:?}"),
                IDENT_INVALID_INPUT,
            )),
        }
    }
}

fn numeric_eq(
    lhs: &NumericBuffer,
    rhs: &NumericBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|err| eq_error(err, IDENT_SIZE_MISMATCH))?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return Ok((Vec::new(), shape));
    }
    let strides_l = compute_strides(&lhs.shape);
    let strides_r = compute_strides(&rhs.shape);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let lhs_val = if lhs.data.is_empty() {
            0.0
        } else {
            let offset = broadcast_index(idx, &shape, &lhs.shape, &strides_l);
            lhs.data[offset]
        };
        let rhs_val = if rhs.data.is_empty() {
            0.0
        } else {
            let offset = broadcast_index(idx, &shape, &rhs.shape, &strides_r);
            rhs.data[offset]
        };
        out.push(if lhs_val == rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

fn complex_eq(
    lhs: &ComplexBuffer,
    rhs: &ComplexBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|err| eq_error(err, IDENT_SIZE_MISMATCH))?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return Ok((Vec::new(), shape));
    }
    let strides_l = compute_strides(&lhs.shape);
    let strides_r = compute_strides(&rhs.shape);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let lhs_val = if lhs.data.is_empty() {
            (0.0, 0.0)
        } else {
            let offset = broadcast_index(idx, &shape, &lhs.shape, &strides_l);
            lhs.data[offset]
        };
        let rhs_val = if rhs.data.is_empty() {
            (0.0, 0.0)
        } else {
            let offset = broadcast_index(idx, &shape, &rhs.shape, &strides_r);
            rhs.data[offset]
        };
        out.push(if lhs_val.0 == rhs_val.0 && lhs_val.1 == rhs_val.1 {
            1
        } else {
            0
        });
    }
    Ok((out, shape))
}

fn string_eq(
    lhs: &StringBuffer,
    rhs: &StringBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|err| eq_error(err, IDENT_SIZE_MISMATCH))?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return Ok((Vec::new(), shape));
    }
    let strides_l = compute_strides(&lhs.shape);
    let strides_r = compute_strides(&rhs.shape);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let lhs_val = if lhs.data.is_empty() {
            ""
        } else {
            let offset = broadcast_index(idx, &shape, &lhs.shape, &strides_l);
            lhs.data[offset].as_str()
        };
        let rhs_val = if rhs.data.is_empty() {
            ""
        } else {
            let offset = broadcast_index(idx, &shape, &rhs.shape, &strides_r);
            rhs.data[offset].as_str()
        };
        out.push(if lhs_val == rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

fn promote_numeric_to_complex(buffer: &NumericBuffer) -> ComplexBuffer {
    ComplexBuffer {
        data: buffer.data.iter().map(|&v| (v, 0.0)).collect(),
        shape: buffer.shape.clone(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_builtins::HandleRef;

    fn run_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
        block_on(super::eq_builtin(lhs, rhs))
    }

    #[cfg(feature = "wgpu")]
    fn run_eq_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
        block_on(eq_host(lhs, rhs))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eq_scalar_true() {
        let result = run_eq(Value::Num(5.0), Value::Num(5.0)).expect("eq");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eq_scalar_false() {
        let result = run_eq(Value::Num(5.0), Value::Num(4.0)).expect("eq");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eq_vector_broadcast() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 2.0], vec![1, 4]).unwrap();
        let result = run_eq(Value::Tensor(tensor), Value::Num(2.0)).expect("eq");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 4]);
                assert_eq!(array.data, vec![0, 1, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eq_char_array_against_numeric() {
        let char_array = CharArray::new(vec!['A', 'B', 'A'], 1, 3).unwrap();
        let tensor = Tensor::new(vec![65.0, 66.0, 65.0], vec![1, 3]).unwrap();
        let result = run_eq(Value::CharArray(char_array), Value::Tensor(tensor)).expect("eq");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(array.data, vec![1, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eq_string_array_broadcast() {
        let sa = StringArray::new(vec!["red".into(), "blue".into()], vec![1, 2]).unwrap();
        let result = run_eq(Value::StringArray(sa), Value::String("red".into())).expect("eq");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 2]);
                assert_eq!(array.data, vec![1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eq_handle_identity() {
        unsafe {
            let raw = Box::into_raw(Box::new(Value::Num(1.0)));
            let ptr = runmat_gc_api::GcPtr::from_raw(raw);
            let handle = HandleRef {
                class_name: "Dummy".to_string(),
                target: ptr,
                valid: true,
            };
            let a = Value::HandleObject(handle.clone());
            let b = Value::HandleObject(handle.clone());
            assert_eq!(run_eq(a.clone(), b.clone()).unwrap(), Value::Bool(true));

            let other_raw = Box::into_raw(Box::new(Value::Num(2.0)));
            let other_ptr = runmat_gc_api::GcPtr::from_raw(other_raw);
            let other_handle = HandleRef {
                class_name: "Dummy".to_string(),
                target: other_ptr,
                valid: true,
            };
            let other = Value::HandleObject(other_handle);
            assert_eq!(run_eq(a, other).unwrap(), Value::Bool(false));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eq_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let a = provider.upload(&view).expect("upload");
            let b = provider.upload(&view).expect("upload");
            let result = run_eq(Value::GpuTensor(a), Value::GpuTensor(b)).expect("gpu eq succeeds");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            assert_eq!(gathered.data, vec![1.0, 1.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eq_numeric_and_string_error() {
        let err = run_eq(Value::Num(1.0), Value::String("a".into())).unwrap_err();
        assert!(err.message().contains("mixing numeric and string inputs"));
        assert_eq!(err.identifier(), Some(IDENT_INVALID_INPUT));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eq_complex_and_numeric() {
        let complex = Value::Complex(2.0, 0.0);
        let numeric = Value::Num(2.0);
        assert_eq!(run_eq(complex, numeric).unwrap(), Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn eq_wgpu_matches_host() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let cpu =
            run_eq_host(Value::Tensor(tensor.clone()), Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let a = provider.upload(&view).unwrap();
        let b = provider.upload(&view).unwrap();
        let gpu = run_eq(Value::GpuTensor(a), Value::GpuTensor(b)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::LogicalArray(expected), actual) => {
                assert_eq!(actual.shape, expected.shape);
                let tol = match provider.precision() {
                    ProviderPrecision::F64 => 1e-12,
                    ProviderPrecision::F32 => 1e-5,
                };
                for (idx, value) in actual.data.iter().enumerate() {
                    let expected_val = expected.data[idx] as f64;
                    assert!((value - expected_val).abs() <= tol);
                }
            }
            (Value::Bool(flag), actual) => {
                assert_eq!(tensor::element_count(&actual.shape), 1);
                assert_eq!(actual.data[0] != 0.0, flag);
            }
            other => panic!("unexpected comparison result {other:?}"),
        }
    }
}
