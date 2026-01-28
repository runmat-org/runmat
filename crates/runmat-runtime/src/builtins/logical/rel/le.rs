//! MATLAB-compatible `le` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::rel::le")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "le",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_le",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Prefers provider elem_le kernels when available; otherwise inputs gather to host tensors automatically.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::rel::le")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "le",
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
            Ok(format!("select({zero}, {one}, ({lhs} <= {rhs}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits comparison kernels that write 1 when the left operand is less than or equal to the right.",
};

const BUILTIN_NAME: &str = "le";
const IDENT_INVALID_INPUT: &str = "MATLAB:le:InvalidInput";
const IDENT_SIZE_MISMATCH: &str = "MATLAB:le:SizeMismatch";
const IDENT_COMPLEX_UNSUPPORTED: &str = "MATLAB:le:ComplexNotSupported";

fn le_error(message: impl Into<String>, identifier: &'static str) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(identifier)
        .build()
}

#[runtime_builtin(
    name = "le",
    category = "logical/rel",
    summary = "Element-wise less-than-or-equal comparison for scalars, arrays, and gpuArray inputs.",
    keywords = "le,less equal,comparison,logical,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::rel::le"
)]
async fn le_builtin(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if let (Value::GpuTensor(ref a), Value::GpuTensor(ref b)) = (&lhs, &rhs) {
        if let Some(result) = try_le_gpu(a, b).await {
            return result;
        }
    }
    le_host(lhs, rhs).await
}

async fn try_le_gpu(
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
) -> Option<crate::BuiltinResult<Value>> {
    let provider = runmat_accelerate_api::provider()?;
    match provider.elem_le(a, b).await {
        Ok(handle) => Some(Ok(gpu_helpers::logical_gpu_value(handle))),
        Err(err) => {
            drop(err);
            None
        }
    }
}

async fn le_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let (lhs, rhs) = normalize_char_string(lhs, rhs);

    let left = LeOperand::from_value(lhs).await?;
    let right = LeOperand::from_value(rhs).await?;

    match (left, right) {
        (LeOperand::Numeric(a), LeOperand::Numeric(b)) => {
            let (data, shape) = numeric_le(&a, &b)?;
            logical_result(data, shape)
        }
        (LeOperand::String(a), LeOperand::String(b)) => {
            let (data, shape) = string_le(&a, &b)?;
            logical_result(data, shape)
        }
        (LeOperand::Numeric(_), LeOperand::String(_))
        | (LeOperand::String(_), LeOperand::Numeric(_)) => Err(le_error(
            "le: mixing numeric and string inputs is not supported",
            IDENT_INVALID_INPUT,
        )),
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
            .map_err(|e| le_error(format!("le: {e}"), IDENT_INVALID_INPUT))
    }
}

enum LeOperand {
    Numeric(NumericBuffer),
    String(StringBuffer),
}

impl LeOperand {
    async fn from_value(value: Value) -> crate::BuiltinResult<Self> {
        match value {
            Value::Num(n) => Ok(LeOperand::Numeric(NumericBuffer::scalar(n))),
            Value::Bool(flag) => Ok(LeOperand::Numeric(NumericBuffer::scalar(if flag {
                1.0
            } else {
                0.0
            }))),
            Value::Int(i) => Ok(LeOperand::Numeric(NumericBuffer::scalar(i.to_f64()))),
            Value::Tensor(tensor) => Ok(LeOperand::Numeric(NumericBuffer::from_tensor(tensor))),
            Value::LogicalArray(array) => {
                Ok(LeOperand::Numeric(NumericBuffer::from_logical(array)))
            }
            Value::CharArray(array) => {
                Ok(LeOperand::Numeric(NumericBuffer::from_char_array(array)))
            }
            Value::String(s) => Ok(LeOperand::String(StringBuffer::scalar(s))),
            Value::StringArray(sa) => Ok(LeOperand::String(StringBuffer::from_array(sa))),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|err| {
                        le_error(format!("{BUILTIN_NAME}: {err}"), IDENT_INVALID_INPUT)
                    })?;
                Ok(LeOperand::Numeric(NumericBuffer::from_tensor(tensor)))
            }
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(le_error(
                "le: complex inputs are not supported",
                IDENT_COMPLEX_UNSUPPORTED,
            )),
            unsupported => Err(le_error(
                format!("le: unsupported input type {unsupported:?}"),
                IDENT_INVALID_INPUT,
            )),
        }
    }
}

fn numeric_le(
    lhs: &NumericBuffer,
    rhs: &NumericBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|err| le_error(err, IDENT_SIZE_MISMATCH))?;
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
        out.push(if lhs_val <= rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

fn string_le(
    lhs: &StringBuffer,
    rhs: &StringBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|err| le_error(err, IDENT_SIZE_MISMATCH))?;
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
        out.push(if lhs_val <= rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
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
        Self {
            data: array.data,
            shape: array.shape,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;

    fn run_le(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
        block_on(super::le_builtin(lhs, rhs))
    }

    #[cfg(feature = "wgpu")]
    fn run_le_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
        block_on(le_host(lhs, rhs))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn le_scalar_true_for_equal_values() {
        let result = run_le(Value::Num(4.0), Value::Num(4.0)).expect("le");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn le_scalar_false() {
        let result = run_le(Value::Num(5.0), Value::Num(3.0)).expect("le");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn le_vector_broadcast() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![1, 4]).unwrap();
        let result = run_le(Value::Tensor(tensor), Value::Num(3.0)).expect("le");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 4]);
                assert_eq!(array.data, vec![1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn le_char_array_against_numeric() {
        let chars = CharArray::new(vec!['A', 'B', 'C'], 1, 3).unwrap();
        let tensor = Tensor::new(vec![65.0, 66.0, 67.0], vec![1, 3]).unwrap();
        let result = run_le(Value::CharArray(chars), Value::Tensor(tensor)).expect("le");
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
    fn le_string_array_against_scalar() {
        let array = StringArray::new(vec!["apple".into(), "carrot".into()], vec![1, 2]).unwrap();
        let result = run_le(Value::StringArray(array), Value::String("banana".into())).expect("le");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 2]);
                assert_eq!(mask.data, vec![1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn le_string_numeric_error() {
        let err =
            run_le(Value::String("apple".into()), Value::Num(3.0)).expect_err("expected error");
        assert!(err.message().contains("mixing numeric and string"));
        assert_eq!(err.identifier(), Some(IDENT_INVALID_INPUT));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn le_complex_error() {
        let err = run_le(Value::Complex(1.0, 1.0), Value::Num(0.0)).expect_err("le");
        assert!(err.message().contains("complex"));
        assert_eq!(err.identifier(), Some(IDENT_COMPLEX_UNSUPPORTED));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn le_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
            let rhs = Tensor::new(vec![2.0, 4.0, 8.0], vec![1, 3]).unwrap();
            let view_l = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let view_r = HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            };
            let handle_l = provider.upload(&view_l).expect("upload lhs");
            let handle_r = provider.upload(&view_r).expect("upload rhs");
            let result =
                run_le(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).expect("le");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.data, vec![1.0, 1.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn le_wgpu_matches_host() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![0.0, 2.0, 5.0, 7.0], vec![4, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = run_le_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();

        let view_l = HostTensorView {
            data: &lhs.data,
            shape: &lhs.shape,
        };
        let view_r = HostTensorView {
            data: &rhs.data,
            shape: &rhs.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle_l = provider.upload(&view_l).expect("upload lhs");
        let handle_r = provider.upload(&view_r).expect("upload rhs");
        let gpu = run_le(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");

        match (cpu, gathered) {
            (Value::LogicalArray(host), tensor) => {
                assert_eq!(tensor.shape, host.shape);
                let expected: Vec<f64> = host
                    .data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect();
                assert_eq!(tensor.data, expected);
            }
            (Value::Bool(host_flag), tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                let expected = if host_flag { 1.0 } else { 0.0 };
                assert_eq!(tensor.data, vec![expected]);
            }
            other => panic!("unexpected output combination: {other:?}"),
        }
    }
}
