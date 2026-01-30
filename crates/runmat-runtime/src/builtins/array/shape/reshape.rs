//! MATLAB-compatible `reshape` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::shape::value_numel;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::common::type_shapes::element_count_if_known;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::reshape")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "reshape",
    op_kind: GpuOpKind::Custom("reshape"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("reshape")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers update residency metadata via custom reshape hook; no kernel launches required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::reshape")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "reshape",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Reshape influences fusion layout but emits no kernels; fusion planner treats it as a metadata op.",
};

fn reshape_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("reshape").build()
}

fn reshape_rank_from_args(args: &[Type]) -> Option<usize> {
    if args.len() < 2 {
        return None;
    }
    if args.len() > 2 {
        return Some(args.len() - 1);
    }
    match &args[1] {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape)
        }
        Type::Num | Type::Int | Type::Bool => Some(1),
        _ => None,
    }
}

fn reshape_type(args: &[Type]) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    let rank = reshape_rank_from_args(args);
    let shape = rank.map(crate::builtins::common::type_shapes::unknown_shape);
    match input {
        Type::Tensor { .. } => Type::Tensor { shape },
        Type::Logical { .. } => Type::Logical { shape },
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Num | Type::Int | Type::Bool => Type::Tensor { shape },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runtime_builtin(
    name = "reshape",
    category = "array/shape",
    summary = "Rearrange the dimensions of an array without changing its data.",
    keywords = "reshape,resize,dimensions,gpu,auto",
    accel = "shape",
    type_resolver(reshape_type),
    builtin_path = "crate::builtins::array::shape::reshape"
)]
async fn reshape_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(reshape_error("reshape: size information missing"));
    }
    let tokens = parse_size_arguments(&rest).await?;
    let numel = value_numel(&value).await?;
    let dims = finalize_dimensions(tokens, numel)?;
    reshape_value(value, &dims)
}

fn reshape_value(value: Value, dims: &[usize]) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            let Tensor { data, .. } = tensor;
            Tensor::new(data, dims.to_vec())
                .map(Value::Tensor)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::ComplexTensor(ct) => {
            let ComplexTensor { data, .. } = ct;
            ComplexTensor::new(data, dims.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::LogicalArray(logical) => {
            let LogicalArray { data, .. } = logical;
            LogicalArray::new(data, dims.to_vec())
                .map(Value::LogicalArray)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::String(s) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::String(s))
            } else {
                StringArray::new(vec![s], dims.to_vec())
                    .map(Value::StringArray)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::StringArray(strings) => {
            let StringArray { data, .. } = strings;
            StringArray::new(data, dims.to_vec())
                .map(Value::StringArray)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::CharArray(chars) => reshape_char_array(chars, dims),
        Value::Cell(cell) => reshape_cell_array(cell, dims),
        Value::GpuTensor(handle) => reshape_gpu_tensor(handle, dims),
        Value::Num(n) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Num(n))
            } else {
                Tensor::new(vec![n], dims.to_vec())
                    .map(Value::Tensor)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::Int(i) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Int(i))
            } else {
                Tensor::new(vec![i.to_f64()], dims.to_vec())
                    .map(Value::Tensor)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::Bool(b) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Bool(b))
            } else {
                let fill = if b { 1u8 } else { 0u8 };
                let total: usize = dims.iter().product();
                LogicalArray::new(vec![fill; total], dims.to_vec())
                    .map(Value::LogicalArray)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::Complex(re, im) => reshape_complex_scalar(re, im, dims),
        other => Err(reshape_error(format!(
            "reshape: unsupported input type {:?}; expected numeric, logical, char, string, cell, or gpu array",
            other
        ))),
    }
}

fn reshape_complex_scalar(re: f64, im: f64, dims: &[usize]) -> crate::BuiltinResult<Value> {
    let total: usize = dims.iter().copied().product();
    if total != 1 {
        return Err(reshape_error(format!(
            "reshape: product of dimensions ({total}) must equal numel(A) (1)"
        )));
    }

    if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
        Ok(Value::Complex(re, im))
    } else {
        ComplexTensor::new(vec![(re, im)], dims.to_vec())
            .map(Value::ComplexTensor)
            .map_err(|e| reshape_error(format!("reshape: {e}")))
    }
}

fn reshape_char_array(ca: CharArray, dims: &[usize]) -> crate::BuiltinResult<Value> {
    let (rows, cols) = match dims.len() {
        0 => {
            return Err(reshape_error(
                "reshape: size vector must contain at least one element",
            ))
        }
        1 => (dims[0], 1),
        2 => (dims[0], dims[1]),
        _ => {
            return Err(reshape_error(
                "reshape: char arrays currently support at most two dimensions",
            ))
        }
    };
    CharArray::new(ca.data, rows, cols)
        .map(Value::CharArray)
        .map_err(|e| reshape_error(format!("reshape: {e}")))
}

fn reshape_cell_array(ca: CellArray, dims: &[usize]) -> crate::BuiltinResult<Value> {
    let (rows, cols) = match dims.len() {
        0 => {
            return Err(reshape_error(
                "reshape: size vector must contain at least one element",
            ))
        }
        1 => (dims[0], 1),
        2 => (dims[0], dims[1]),
        _ => {
            return Err(reshape_error(
                "reshape: cell arrays currently support at most two dimensions",
            ))
        }
    };
    CellArray::new_handles(ca.data, rows, cols)
        .map(Value::Cell)
        .map_err(|e| reshape_error(format!("reshape: {e}")))
}

fn reshape_gpu_tensor(
    handle: runmat_accelerate_api::GpuTensorHandle,
    dims: &[usize],
) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        provider
            .reshape(&handle, dims)
            .map(Value::GpuTensor)
            .map_err(|e| reshape_error(format!("reshape: {e}")))
    } else {
        let mut updated = handle;
        updated.shape = dims.to_vec();
        Ok(Value::GpuTensor(updated))
    }
}

#[derive(Clone, Copy, Debug)]
enum DimToken {
    Known(usize),
    Auto,
}

async fn parse_size_arguments(args: &[Value]) -> crate::BuiltinResult<Vec<DimToken>> {
    if args.len() == 1 {
        let value = &args[0];
        match value {
            Value::Tensor(t) => {
                if t.data.is_empty() {
                    return Err(reshape_error(
                        "reshape: size vector must contain at least one element",
                    ));
                }
                let dims = tensor::dims_from_value_async(value)
                    .await
                    .map_err(|e| reshape_error(format!("reshape: {e}")))?;
                let Some(dims) = dims else {
                    return Err(reshape_error(
                        "reshape: size vector must be a row or column vector",
                    ));
                };
                Ok(dims.into_iter().map(DimToken::Known).collect())
            }
            Value::LogicalArray(la) => {
                if la.data.is_empty() {
                    return Err(reshape_error(
                        "reshape: size vector must contain at least one element",
                    ));
                }
                let dims = tensor::dims_from_value_async(value)
                    .await
                    .map_err(|e| reshape_error(format!("reshape: {e}")))?;
                let Some(dims) = dims else {
                    return Err(reshape_error(
                        "reshape: size vector must be a row or column vector",
                    ));
                };
                Ok(dims.into_iter().map(DimToken::Known).collect())
            }
            Value::GpuTensor(_) => {
                let dims = tensor::dims_from_value_async(value)
                    .await
                    .map_err(|e| reshape_error(format!("reshape: {e}")))?;
                let Some(dims) = dims else {
                    return Err(reshape_error(
                        "reshape: size vector must be a row or column vector",
                    ));
                };
                Ok(dims.into_iter().map(DimToken::Known).collect())
            }
            Value::Int(_) | Value::Num(_) | Value::Bool(_) => {
                Ok(vec![parse_size_scalar(value).await?])
            }
            other => Err(reshape_error(format!(
                "reshape: size vector must be numeric, got {:?}",
                other
            ))),
        }
    } else {
        let mut tokens = Vec::with_capacity(args.len());
        for value in args {
            tokens.push(parse_size_scalar(value).await?);
        }
        Ok(tokens)
    }
}
async fn parse_size_scalar(value: &Value) -> crate::BuiltinResult<DimToken> {
    match value {
        Value::Tensor(t) => {
            if t.data.is_empty() {
                return Ok(DimToken::Auto);
            }
            if t.data.len() != 1 {
                return Err(reshape_error("reshape: size arguments must be scalars"));
            }
        }
        Value::LogicalArray(la) => {
            if la.data.is_empty() {
                return Ok(DimToken::Auto);
            }
            if la.data.len() != 1 {
                return Err(reshape_error("reshape: size arguments must be scalars"));
            }
        }
        _ => {}
    }

    let Some(dim) = tensor::dimension_from_value_async(value, "reshape", true)
        .await
        .map_err(|e| reshape_error(format!("reshape: {e}")))?
    else {
        return Err(reshape_error(format!(
            "reshape: size arguments must be numeric scalars, got {:?}",
            value
        )));
    };
    Ok(DimToken::Known(dim))
}

fn finalize_dimensions(tokens: Vec<DimToken>, numel: usize) -> crate::BuiltinResult<Vec<usize>> {
    if tokens.is_empty() {
        return Err(reshape_error(
            "reshape: size vector must contain at least one element",
        ));
    }

    let mut dims = Vec::with_capacity(tokens.len());
    let mut known_product: usize = 1;
    let mut auto_index: Option<usize> = None;

    for (idx, token) in tokens.iter().enumerate() {
        match token {
            DimToken::Known(value) => {
                if *value == 0 {
                    known_product = 0;
                } else if known_product != 0 {
                    known_product = known_product.checked_mul(*value).ok_or_else(|| {
                        reshape_error("reshape: product of dimensions exceeds usize range")
                    })?;
                }
                dims.push(*value);
            }
            DimToken::Auto => {
                if auto_index.is_some() {
                    return Err(reshape_error(
                        "reshape: can only specify a single [] dimension",
                    ));
                }
                auto_index = Some(idx);
                dims.push(1); // placeholder
            }
        }
    }

    if let Some(auto) = auto_index {
        if known_product == 0 {
            if numel != 0 {
                return Err(reshape_error(format!(
                    "reshape: product of dimensions (0) must equal numel(A) ({numel})"
                )));
            }
            dims[auto] = 0;
        } else if !numel.is_multiple_of(known_product) {
            return Err(reshape_error(format!(
                "reshape: product of dimensions ({}) must equal numel(A) ({numel})",
                known_product
            )));
        } else {
            dims[auto] = numel / known_product;
        }
    } else if known_product != numel {
        return Err(reshape_error(format!(
            "reshape: product of dimensions ({known_product}) must equal numel(A) ({numel})"
        )));
    }

    Ok(dims)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::dispatcher::download_handle_async;
    use futures::executor::block_on;

    fn reshape_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::reshape_builtin(value, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, LogicalArray};

    fn tensor_from_slice(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_vector_to_matrix() {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[12, 1]);
        let result = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(tensor_from_slice(&[3.0, 4.0], &[1, 2]))],
        )
        .expect("reshape");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 4]);
                assert_eq!(out.data, (1..=12).map(|v| v as f64).collect::<Vec<_>>());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_with_auto_dimension() {
        let data: Vec<f64> = (1..=18).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[18, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = vec![Value::from(3.0), Value::Tensor(empty)];
        let result = reshape_builtin(Value::Tensor(tensor), args).expect("reshape");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 6]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_logical_array_preserves_type() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0, 1, 0], vec![6, 1]).expect("logical");
        let result = reshape_builtin(
            Value::LogicalArray(logical),
            vec![Value::from(2.0), Value::from(3.0)],
        )
        .expect("reshape");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![1, 0, 1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_char_array_single_dimension_becomes_column() {
        let chars = CharArray::new("abcd".chars().collect(), 1, 4).expect("char array");
        let result =
            reshape_builtin(Value::CharArray(chars), vec![Value::from(4.0)]).expect("reshape");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 4);
                assert_eq!(out.cols, 1);
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "abcd");
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_cell_array_two_dimensional() {
        let cell = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).expect("cell");
        let result = reshape_builtin(Value::Cell(cell), vec![Value::from(2.0), Value::from(1.0)])
            .expect("reshape");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 1);
                let first = out.get(0, 0).expect("first cell");
                let second = out.get(1, 0).expect("second cell");
                assert!(matches!(first, Value::Num(f) if (f - 1.0).abs() < 1e-12));
                assert!(matches!(second, Value::Num(f) if (f - 2.0).abs() < 1e-12));
            }
            other => panic!("expected CellArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_string_scalar_high_rank() {
        let result = reshape_builtin(
            Value::String("runmat".to_string()),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1, 1]);
                assert_eq!(sa.data, vec!["runmat".to_string()]);
            }
            other => panic!("expected StringArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_gpu_preserves_handle_shape() {
        test_support::with_test_provider(|provider| {
            let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
            let tensor = tensor_from_slice(&data, &[3, 4]);
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = reshape_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::from(2.0), Value::from(6.0)],
            )
            .expect("reshape");
            match result {
                Value::GpuTensor(out) => {
                    assert_eq!(out.shape, vec![2, 6]);
                    assert_eq!(out.buffer_id, handle.buffer_id);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn reshape_wgpu_updates_provider_shape() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        let _ = register_wgpu_provider(WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[3, 4]);
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = reshape_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::from(2.0), Value::from(6.0)],
        )
        .expect("reshape");
        let Value::GpuTensor(reshaped) = result else {
            panic!("expected gpu tensor");
        };
        assert_eq!(reshaped.shape, vec![2, 6]);
        let host = block_on(download_handle_async(provider, &reshaped)).expect("download");
        assert_eq!(host.shape, vec![2, 6]);
        assert_eq!(host.data, tensor.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_mismatched_elements_errors() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[6, 1]);
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(4.0), Value::from(4.0)],
        )
        .expect_err("should fail");
        assert!(err.to_string().contains("product of dimensions"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_multiple_auto_errors() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[6, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(empty.clone()), Value::Tensor(empty)],
        )
        .expect_err("should fail");
        assert!(err.to_string().contains("single []"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_accepts_zero_sized_dimension() {
        let tensor = tensor_from_slice(&[], &[0, 1]);
        let result = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(0.0), Value::from(3.0)],
        )
        .expect("reshape zero");
        match result {
            Value::Tensor(out) => assert_eq!(out.shape, vec![0, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_int_scalar_to_vector() {
        let value = Value::Int(IntValue::I32(5));
        let err = reshape_builtin(value.clone(), vec![Value::from(1.0), Value::from(5.0)])
            .expect_err("should fail because numel mismatch");
        assert!(err.to_string().contains("numel"));
        let ok = reshape_builtin(value, vec![Value::from(1.0), Value::from(1.0)])
            .expect("reshape scalar");
        assert!(matches!(ok, Value::Int(_)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_complex_scalar_high_rank() {
        let result = reshape_builtin(
            Value::Complex(1.0, 2.0),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 1, 1]);
                assert_eq!(ct.data, vec![(1.0, 2.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_auto_dimension_mismatch_reports_product() {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[12, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(5.0), Value::Tensor(empty)],
        )
        .expect_err("should fail");
        assert!(
            err.to_string().contains("5"),
            "expected product to appear in error message, got {err}"
        );
    }
}
