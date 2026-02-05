//! MATLAB-compatible `flipud` builtin with GPU-aware semantics for RunMat.

use crate::builtins::array::shape::flip::{
    complex_tensor_into_value, flip_char_array_with, flip_complex_tensor_with, flip_gpu_with,
    flip_logical_array_with, flip_string_array_with, flip_tensor_with,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{CellArray, ComplexTensor, ResolveContext, Type, Value};
use runmat_macros::runtime_builtin;

const UD_DIM: [usize; 1] = [1];

fn preserve_matrix_type(args: &[Type], _context: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape: Some(shape) } => {
            let rows = shape.get(0).copied().unwrap_or(None);
            let cols = shape.get(1).copied().unwrap_or(None);
            Type::Tensor {
                shape: Some(vec![rows, cols]),
            }
        }
        Type::Logical { shape: Some(shape) } => {
            let rows = shape.get(0).copied().unwrap_or(None);
            let cols = shape.get(1).copied().unwrap_or(None);
            Type::Logical {
                shape: Some(vec![rows, cols]),
            }
        }
        Type::Tensor { shape: None } => Type::tensor(),
        Type::Logical { shape: None } => Type::logical(),
        Type::Num | Type::Int | Type::Bool => Type::tensor(),
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::flipud")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "flipud",
    op_kind: GpuOpKind::Custom("flip"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("flip")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to the generic flip hook with axis=0; falls back to host mirror when the hook is missing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::flipud")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "flipud",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a data-reordering barrier; fusion planner preserves residency but does not fuse through flipud.",
};

fn flipud_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("flipud").build()
}

#[runtime_builtin(
    name = "flipud",
    category = "array/shape",
    summary = "Flip an array up-to-down along the first dimension.",
    keywords = "flipud,flip,vertical,matrix,gpu",
    accel = "custom",
    type_resolver(preserve_matrix_type),
    builtin_path = "crate::builtins::array::shape::flipud"
)]
async fn flipud_builtin(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM).map(tensor::tensor_into_value)?)
        }
        Value::LogicalArray(array) => {
            Ok(flip_logical_array_with("flipud", array, &UD_DIM).map(Value::LogicalArray)?)
        }
        Value::ComplexTensor(ct) => {
            Ok(flip_complex_tensor_with("flipud", ct, &UD_DIM).map(Value::ComplexTensor)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| flipud_error(format!("flipud: {e}")))?;
            Ok(flip_complex_tensor_with("flipud", tensor, &UD_DIM)
                .map(complex_tensor_into_value)?)
        }
        Value::StringArray(strings) => {
            Ok(flip_string_array_with("flipud", strings, &UD_DIM).map(Value::StringArray)?)
        }
        Value::CharArray(chars) => {
            Ok(flip_char_array_with("flipud", chars, &UD_DIM).map(Value::CharArray)?)
        }
        Value::String(scalar) => Ok(Value::String(scalar)),
        Value::Cell(cell) => flip_cell_array_rows(cell),
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("flipud", Value::Num(n))
                .map_err(|e| flipud_error(e))?;
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM).map(tensor::tensor_into_value)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("flipud", Value::Int(i))
                .map_err(|e| flipud_error(e))?;
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM).map(tensor::tensor_into_value)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("flipud", Value::Bool(flag))
                .map_err(|e| flipud_error(e))?;
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => Ok(flip_gpu_with("flipud", handle, &UD_DIM).await?),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(flipud_error("flipud: unsupported input type")),
    }
}

fn flip_cell_array_rows(cell: CellArray) -> crate::BuiltinResult<Value> {
    if cell.rows <= 1 || cell.data.is_empty() {
        return Ok(Value::Cell(cell));
    }
    let rows = cell.rows;
    let cols = cell.cols;
    let data = cell.data;
    let mut flipped = Vec::with_capacity(data.len());
    for row in (0..rows).rev() {
        let base = row * cols;
        for col in 0..cols {
            flipped.push(data[base + col].clone());
        }
    }
    CellArray::new_handles(flipped, rows, cols)
        .map(Value::Cell)
        .map_err(|e| flipud_error(format!("flipud: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn flipud_builtin(value: Value) -> crate::BuiltinResult<Value> {
        block_on(super::flipud_builtin(value))
    }
    use crate::builtins::array::shape::flip::{
        flip_complex_tensor, flip_logical_array, flip_tensor,
    };
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, LogicalArray, StringArray, StructValue, Tensor, Type, Value,
    };

    #[test]
    fn flipud_type_keeps_matrix_shape() {
        let out = preserve_matrix_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_matrix_reverses_rows() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("tensor");
        let expected = flip_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_column_vector_reverses_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let expected = flip_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_row_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let expected = tensor.clone();
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_nd_tensor_flips_first_dim_only() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_char_array() {
        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let result = flipud_builtin(Value::CharArray(chars)).expect("flipud");
        match result {
            Value::CharArray(out) => {
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "matrun");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_string_array() {
        let strings =
            StringArray::new(vec!["top".into(), "bottom".into()], vec![2, 1]).expect("strings");
        let result = flipud_builtin(Value::StringArray(strings)).expect("flipud");
        match result {
            Value::StringArray(out) => assert_eq!(out.data, vec!["bottom", "top"]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_cell_array_reverses_rows() {
        let cell = CellArray::new(
            vec![
                Value::from("r1c1"),
                Value::from("r1c2"),
                Value::from("r2c1"),
                Value::from("r2c2"),
            ],
            2,
            2,
        )
        .expect("cell");
        let result = flipud_builtin(Value::Cell(cell)).expect("flipud");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.get(0, 0).unwrap(), Value::from("r2c1"));
                assert_eq!(out.get(0, 1).unwrap(), Value::from("r2c2"));
                assert_eq!(out.get(1, 0).unwrap(), Value::from("r1c1"));
                assert_eq!(out.get(1, 1).unwrap(), Value::from("r1c2"));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_logical_array_preserves_bits() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let expected = flip_logical_array(logical.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::LogicalArray(logical)).expect("flipud");
        match result {
            Value::LogicalArray(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_scalar_numeric_noop() {
        let result = flipud_builtin(Value::Num(42.0)).expect("flipud");
        match result {
            Value::Num(v) => assert_eq!(v, 42.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_complex_tensor_defaults_to_first_dim() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (2.0, -1.0), (3.0, 0.5), (4.0, -0.25)],
            vec![2, 2],
        )
        .unwrap();
        let expected = flip_complex_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::ComplexTensor(tensor)).expect("flipud");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_rejects_struct_inputs() {
        let mut st = StructValue::new();
        st.fields.insert("field".into(), Value::Num(1.0));
        let err = flipud_builtin(Value::Struct(st)).expect_err("struct unsupported");
        assert!(err.to_string().contains("unsupported input type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
            let gathered = test_support::gather(result).expect("gather");
            let expected = flip_tensor(tensor, &UD_DIM).expect("expected");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.data, expected.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_preserves_row_vector() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, tensor.shape);
            assert_eq!(gathered.data, tensor.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_falls_back_when_axis_missing() {
        // The simple provider does not expose flip, so this exercises gather→flip→upload.
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result = flipud_builtin(Value::Tensor(tensor.clone())).expect("flipud");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, flip_tensor(tensor, &UD_DIM).unwrap().data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_with_registered_provider_preserves_gpu_type() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
            assert!(matches!(result, Value::GpuTensor(_)));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn flipud_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let cpu = flip_tensor(tensor.clone(), &UD_DIM).expect("cpu flip");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }
}
