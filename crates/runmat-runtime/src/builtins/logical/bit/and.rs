//! MATLAB-compatible logical `and` builtin with GPU support.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor, type_shapes::logical_binary_type};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::bit::and")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "and",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "logical_and",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Falls back to host execution when the provider does not implement logical_and; non-zero (including NaN) inputs map to true.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::bit::and")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "and",
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
            let cond = format!("(({lhs} != {zero}) && ({rhs} != {zero}))");
            Ok(format!("select({zero}, {one}, {cond})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion generates WGSL kernels that treat non-zero inputs as true and write 0/1 outputs.",
};

#[runtime_builtin(
    name = "and",
    category = "logical/bit",
    summary = "Element-wise logical AND for scalars, arrays, and gpuArray values.",
    keywords = "logical,and,elementwise,boolean,gpu",
    accel = "elementwise",
    type_resolver(logical_binary_type),
    builtin_path = "crate::builtins::logical::bit::and"
)]
async fn and_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let (Value::GpuTensor(ref a), Value::GpuTensor(ref b)) = (&lhs, &rhs) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.logical_and(a, b) {
                return Ok(gpu_helpers::logical_gpu_value(handle));
            }
        }
    }
    and_host(lhs, rhs).await
}

async fn and_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    let left = logical_buffer_from("and", lhs).await?;
    let right = logical_buffer_from("and", rhs).await?;
    let shape = broadcast_shapes("and", &left.shape, &right.shape)
        .map_err(|err| builtin_error("and", err))?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return logical_value("and", Vec::new(), shape);
    }

    let strides_left = compute_strides(&left.shape);
    let strides_right = compute_strides(&right.shape);

    let mut data = Vec::with_capacity(total);
    for linear in 0..total {
        let lhs_bit = if left.data.is_empty() {
            0
        } else {
            let idx = broadcast_index(linear, &shape, &left.shape, &strides_left);
            *left.data.get(idx).unwrap_or(&0)
        };
        let rhs_bit = if right.data.is_empty() {
            0
        } else {
            let idx = broadcast_index(linear, &shape, &right.shape, &strides_right);
            *right.data.get(idx).unwrap_or(&0)
        };
        data.push(if lhs_bit != 0 && rhs_bit != 0 { 1 } else { 0 });
    }

    logical_value("and", data, shape)
}

fn builtin_error(fn_name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(fn_name).build()
}

fn logical_value(fn_name: &str, data: Vec<u8>, shape: Vec<usize>) -> BuiltinResult<Value> {
    if data.len() == 1 && tensor::element_count(&shape) == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| builtin_error(fn_name, format!("{fn_name}: {e}")))
    }
}

struct LogicalBuffer {
    data: Vec<u8>,
    shape: Vec<usize>,
}

async fn logical_buffer_from(name: &str, value: Value) -> BuiltinResult<LogicalBuffer> {
    match value {
        Value::LogicalArray(array) => {
            let LogicalArray { data, shape } = array;
            Ok(LogicalBuffer { data, shape })
        }
        Value::Bool(flag) => Ok(LogicalBuffer {
            data: vec![if flag { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Num(n) => Ok(LogicalBuffer {
            data: vec![logical_from_f64(n)],
            shape: vec![1, 1],
        }),
        Value::Int(i) => Ok(LogicalBuffer {
            data: vec![if i.to_i64() != 0 { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Complex(re, im) => Ok(LogicalBuffer {
            data: vec![logical_from_complex(re, im)],
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor_to_logical_buffer(tensor),
        Value::ComplexTensor(tensor) => complex_tensor_to_logical_buffer(tensor),
        Value::CharArray(array) => char_array_to_logical_buffer(array),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|err| builtin_error(name, format!("{name}: {err}")))?;
            tensor_to_logical_buffer(tensor)
        }
        other => Err(builtin_error(
            name,
            format!(
                "{name}: unsupported input type {:?}; expected logical, numeric, complex, or character data",
                other
            ),
        )),
    }
}

fn tensor_to_logical_buffer(tensor: Tensor) -> BuiltinResult<LogicalBuffer> {
    let Tensor { data, shape, .. } = tensor;
    let mapped = data.into_iter().map(logical_from_f64).collect();
    Ok(LogicalBuffer {
        data: mapped,
        shape,
    })
}

fn complex_tensor_to_logical_buffer(tensor: ComplexTensor) -> BuiltinResult<LogicalBuffer> {
    let ComplexTensor { data, shape, .. } = tensor;
    let mapped = data
        .into_iter()
        .map(|(re, im)| logical_from_complex(re, im))
        .collect();
    Ok(LogicalBuffer {
        data: mapped,
        shape,
    })
}

fn char_array_to_logical_buffer(array: CharArray) -> BuiltinResult<LogicalBuffer> {
    let CharArray { data, rows, cols } = array;
    let mapped = data
        .into_iter()
        .map(|ch| if ch == '\0' { 0 } else { 1 })
        .collect();
    Ok(LogicalBuffer {
        data: mapped,
        shape: vec![rows, cols],
    })
}

#[inline]
fn logical_from_f64(value: f64) -> u8 {
    if value != 0.0 {
        1
    } else {
        0
    }
}

#[inline]
fn logical_from_complex(re: f64, im: f64) -> u8 {
    if re != 0.0 || im != 0.0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;

    fn assert_error_contains(err: RuntimeError, expected: &str) {
        assert!(
            err.message().contains(expected),
            "unexpected error: {}",
            err.message()
        );
    }

    fn run_and(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(super::and_builtin(lhs, rhs))
    }

    #[cfg(feature = "wgpu")]
    fn run_and_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(and_host(lhs, rhs))
    }
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_builtins::IntValue;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_of_booleans() {
        assert_eq!(
            run_and(Value::Bool(true), Value::Bool(false)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_and(Value::Bool(true), Value::Bool(true)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_numeric_arrays() {
        let a = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = run_and(Value::Tensor(a), Value::Tensor(b)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(array.data, vec![1, 0, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_scalar_broadcasts() {
        let tensor = Tensor::new(vec![1.0, 0.0, 3.0, 0.0], vec![4, 1]).unwrap();
        let result = run_and(Value::Tensor(tensor), Value::Int(IntValue::I32(1))).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![4, 1]);
                assert_eq!(array.data, vec![1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_char_arrays() {
        let lhs = CharArray::new("Run".chars().collect(), 1, 3).unwrap();
        let rhs = CharArray::new(vec!['R', 'u', '\0'], 1, 3).unwrap();
        let result =
            run_and(Value::CharArray(lhs), Value::CharArray(rhs)).expect("and char arrays");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![1, 3]);
                assert_eq!(arr.data, vec![1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_treats_nan_as_true() {
        let result = run_and(Value::Num(f64::NAN), Value::Num(1.0)).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_complex_inputs() {
        let result = run_and(Value::Complex(0.0, 0.0), Value::Complex(0.0, 2.0)).unwrap();
        assert_eq!(result, Value::Bool(false));

        let result = run_and(Value::Complex(1.0, 0.0), Value::Complex(0.0, 2.0)).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_size_mismatch_errors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
        let err = run_and(Value::Tensor(lhs), Value::Tensor(rhs)).unwrap_err();
        assert_error_contains(err, "size mismatch");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_rejects_unsupported_types() {
        let err = run_and(Value::String("runmat".into()), Value::Bool(true)).unwrap_err();
        assert_error_contains(err, "unsupported input type");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 2.0, 0.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let a = provider.upload(&view).unwrap();
            let b = provider.upload(&view).unwrap();
            let result = run_and(Value::GpuTensor(a), Value::GpuTensor(b)).unwrap();
            let gathered = test_support::gather(result).unwrap();
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![0.0, 1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn and_gpu_supports_broadcast() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![0.0, 2.0, 0.0, 4.0], vec![4, 1]).unwrap();
            let rhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();

            let lhs_view = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let rhs_view = HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            };

            let gpu_lhs = provider.upload(&lhs_view).expect("upload lhs");
            let gpu_rhs = provider.upload(&rhs_view).expect("upload rhs");

            let result =
                run_and(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs)).expect("and");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![0.0, 1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn and_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");

        let lhs = Tensor::new(vec![0.0, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![1.0, 0.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let cpu_value =
            run_and_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).expect("host and");
        let (expected_data, expected_shape) = match cpu_value {
            Value::LogicalArray(arr) => (arr.data.clone(), arr.shape.clone()),
            other => panic!("expected logical array, got {other:?}"),
        };

        let view_lhs = HostTensorView {
            data: &lhs.data,
            shape: &lhs.shape,
        };
        let view_rhs = HostTensorView {
            data: &rhs.data,
            shape: &rhs.shape,
        };
        let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
        let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");

        let gpu_value =
            run_and(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs)).expect("gpu and");
        let gathered = test_support::gather(gpu_value).expect("gather gpu result");

        assert_eq!(gathered.shape, expected_shape);
        let tol = match provider.precision() {
            ProviderPrecision::F64 => 1e-12,
            ProviderPrecision::F32 => 1e-5,
        };
        for (idx, (actual, expected)) in gathered.data.iter().zip(expected_data.iter()).enumerate()
        {
            let expected_f = if *expected != 0 { 1.0 } else { 0.0 };
            assert!(
                (actual - expected_f).abs() <= tol,
                "mismatch at index {idx}: got {actual}, expected {expected_f}"
            );
        }
    }
}
