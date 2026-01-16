//! MATLAB-compatible logical `not` builtin with GPU support.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::bit::not")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "not",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_not",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dispatches to the provider `logical_not` hook when available; otherwise the runtime gathers to host and performs the negation on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::bit::not")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "not",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0".to_string(), "1.0".to_string()),
                ScalarType::F64 => ("f64(0.0)".to_string(), "f64(1.0)".to_string()),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            let cond = format!("({input} != {zero})");
            Ok(format!("select({one}, {zero}, {cond})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion kernels treat any non-zero input as true and write 0/1 outputs, matching MATLAB logical semantics.",
};

#[runtime_builtin(
    name = "not",
    category = "logical/bit",
    summary = "Element-wise logical negation for scalars, arrays, and gpuArray values.",
    keywords = "logical,not,boolean,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::bit::not"
)]
fn not_builtin(value: Value) -> Result<Value, String> {
    if let Value::GpuTensor(ref handle) = value {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(device_out) = provider.logical_not(handle) {
                return Ok(gpu_helpers::logical_gpu_value(device_out));
            }
        }
    }
    not_host(value)
}

fn not_host(value: Value) -> Result<Value, String> {
    let buffer = logical_buffer_from("not", value)?;
    let LogicalBuffer { data, shape } = buffer;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return logical_value("not", Vec::new(), shape);
    }
    let mapped = data
        .into_iter()
        .map(|bit| if bit == 0 { 1 } else { 0 })
        .collect::<Vec<_>>();
    logical_value("not", mapped, shape)
}

fn logical_value(fn_name: &str, data: Vec<u8>, shape: Vec<usize>) -> Result<Value, String> {
    if data.len() == 1 && tensor::element_count(&shape) == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| format!("{fn_name}: {e}"))
    }
}

struct LogicalBuffer {
    data: Vec<u8>,
    shape: Vec<usize>,
}

fn logical_buffer_from(name: &str, value: Value) -> Result<LogicalBuffer, String> {
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
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            tensor_to_logical_buffer(tensor)
        }
        other => Err(format!(
            "{name}: unsupported input type {:?}; expected logical, numeric, complex, or character data",
            other
        )),
    }
}

fn tensor_to_logical_buffer(tensor: Tensor) -> Result<LogicalBuffer, String> {
    let Tensor { data, shape, .. } = tensor;
    let mapped = data
        .into_iter()
        .map(|v| if v != 0.0 { 1 } else { 0 })
        .collect();
    Ok(LogicalBuffer {
        data: mapped,
        shape,
    })
}

fn complex_tensor_to_logical_buffer(tensor: ComplexTensor) -> Result<LogicalBuffer, String> {
    let ComplexTensor { data, shape, .. } = tensor;
    let mapped = data
        .into_iter()
        .map(|(re, im)| if re != 0.0 || im != 0.0 { 1 } else { 0 })
        .collect();
    Ok(LogicalBuffer {
        data: mapped,
        shape,
    })
}

fn char_array_to_logical_buffer(array: CharArray) -> Result<LogicalBuffer, String> {
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
    #[cfg(feature = "wgpu")]
    use crate::builtins::common::tensor;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_of_booleans() {
        assert_eq!(not_builtin(Value::Bool(true)).unwrap(), Value::Bool(false));
        assert_eq!(not_builtin(Value::Bool(false)).unwrap(), Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_numeric_array() {
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let result = not_builtin(Value::Tensor(tensor)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(array.data, vec![1, 0, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_complex_scalar() {
        let result =
            not_builtin(Value::Complex(0.0, 0.0)).expect("not complex zero should succeed");
        assert_eq!(result, Value::Bool(true));

        let result =
            not_builtin(Value::Complex(1.0, 0.0)).expect("not complex nonzero should succeed");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_nan_yields_false() {
        let result = not_builtin(Value::Num(f64::NAN)).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_char_array() {
        let chars = CharArray::new_row("A\0C");
        let result = not_builtin(Value::CharArray(chars)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(array.data, vec![0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = not_builtin(Value::GpuTensor(handle)).expect("not on gpu");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 1.0, 0.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_accepts_int_inputs() {
        let value = Value::Int(IntValue::I32(0));
        assert_eq!(not_builtin(value).unwrap(), Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_tensor_scalar_returns_bool() {
        let tensor = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        assert_eq!(
            not_builtin(Value::Tensor(tensor)).unwrap(),
            Value::Bool(false)
        );

        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        assert_eq!(
            not_builtin(Value::Tensor(tensor)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = not_builtin(Value::Tensor(tensor)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![0, 3]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_complex_tensor() {
        let tensor =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 0.0), (0.0, -2.0)], vec![3, 1]).unwrap();
        let result = not_builtin(Value::ComplexTensor(tensor)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![3, 1]);
                assert_eq!(array.data, vec![1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_logical_array_flips_bits() {
        let array = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = not_builtin(Value::LogicalArray(array)).unwrap();
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0, 1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn not_rejects_string_input() {
        let err = not_builtin(Value::String("abc".into())).unwrap_err();
        assert!(
            err.contains("unsupported input type"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn not_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 3.0, 0.0, -1.0], vec![2, 2]).unwrap();
        let cpu = not_host(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = not_builtin(Value::GpuTensor(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = tensor::value_to_tensor(&cpu).expect("cpu tensor");
        assert_eq!(gathered.shape, cpu_tensor.shape);
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            ProviderPrecision::F64 => 1e-12,
            ProviderPrecision::F32 => 1e-5,
        };
        for (expected, actual) in cpu_tensor.data.iter().zip(gathered.data.iter()) {
            assert!((*expected - *actual).abs() < tol, "{expected} vs {actual}");
        }
    }
}
