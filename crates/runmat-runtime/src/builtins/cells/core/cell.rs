//! MATLAB-compatible `cell` builtin implemented for the modern RunMat runtime.

use runmat_builtins::{
    CharArray, ComplexTensor, IntValue, LogicalArray, StringArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{keyword_of, shape_from_value};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{gather_if_needed, make_cell_with_shape};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::cells::core::cell")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cell",
    op_kind: GpuOpKind::Custom("container"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Cell arrays are allocated on the host heap; providers currently gather any GPU inputs and rely on host execution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::cells::core::cell")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cell",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Cell creation acts as a fusion sink and terminates GPU fusion plans.",
};

#[runtime_builtin(
    name = "cell",
    category = "cells/core",
    summary = "Create empty MATLAB cell arrays.",
    keywords = "cell,cell array,container,empty",
    accel = "array_construct",
    sink = true,
    builtin_path = "crate::builtins::cells::core::cell"
)]
fn cell_builtin(args: Vec<Value>) -> Result<Value, String> {
    let parsed = ParsedCell::parse(args)?;
    build_cell(parsed)
}

struct ParsedCell {
    shape: Vec<usize>,
    prototype: Option<Value>,
}

impl ParsedCell {
    fn parse(args: Vec<Value>) -> Result<Self, String> {
        let mut dims: Vec<Value> = Vec::new();
        let mut prototype: Option<Value> = None;
        let mut idx = 0;

        while idx < args.len() {
            let value = &args[idx];
            if let Some(keyword) = keyword_of(value) {
                match keyword.as_str() {
                    "like" => {
                        if prototype.is_some() {
                            return Err("cell: multiple 'like' specifications are not supported"
                                .to_string());
                        }
                        let Some(proto) = args.get(idx + 1) else {
                            return Err("cell: expected prototype after 'like'".to_string());
                        };
                        prototype = Some(proto.clone());
                        idx += 2;
                        continue;
                    }
                    other => {
                        return Err(format!("cell: unrecognised option '{other}'"));
                    }
                }
            }

            dims.push(args[idx].clone());
            idx += 1;
        }

        let shape = parse_shape_arguments(&dims, prototype.as_ref())?;
        Ok(Self { shape, prototype })
    }
}

fn build_cell(parsed: ParsedCell) -> Result<Value, String> {
    let shape = ensure_min_rank(parsed.shape);
    let total = if shape.is_empty() {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| "cell: requested size exceeds platform limits".to_string())?
    };

    if total == 0 {
        return make_cell_with_shape(Vec::new(), shape).map_err(|e| format!("cell: {e}"));
    }

    let default_value = empty_value_like(parsed.prototype.as_ref())?;
    let mut values = Vec::with_capacity(total);
    values.resize(total, default_value);
    make_cell_with_shape(values, shape).map_err(|e| format!("cell: {e}"))
}

fn ensure_min_rank(dims: Vec<usize>) -> Vec<usize> {
    match dims.len() {
        0 => vec![0, 0],
        1 => vec![dims[0], 1],
        _ => dims,
    }
}

fn parse_shape_arguments(args: &[Value], prototype: Option<&Value>) -> Result<Vec<usize>, String> {
    if args.is_empty() {
        if let Some(proto) = prototype {
            return shape_from_value(proto, "cell");
        }
        return Ok(vec![0, 0]);
    }

    if args.len() == 1 {
        let host = gather_if_needed(&args[0]).map_err(|e| format!("cell: {e}"))?;
        return parse_single_argument(&host);
    }

    let mut dims = Vec::with_capacity(args.len());
    for value in args {
        let host = gather_if_needed(value).map_err(|e| format!("cell: {e}"))?;
        dims.push(parse_size_scalar(&host, "cell")?);
    }
    Ok(dims)
}

fn parse_single_argument(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) => {
            let n = parse_size_scalar(value, "cell")?;
            Ok(vec![n, n])
        }
        Value::Tensor(t) => parse_size_tensor(t),
        Value::LogicalArray(arr) => parse_size_logical_array(arr),
        other => Err(format!(
            "cell: size arguments must be numeric scalars or vectors, got {other:?}"
        )),
    }
}

fn parse_size_scalar(value: &Value, context: &str) -> Result<usize, String> {
    match value {
        Value::Int(iv) => parse_intvalue(iv, context),
        Value::Num(n) => parse_numeric(*n, context),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            if t.data.len() != 1 {
                return Err(format!("{context}: size inputs must be scalar"));
            }
            parse_numeric(t.data[0], context)
        }
        Value::LogicalArray(arr) => {
            if arr.data.len() != 1 {
                return Err(format!("{context}: size inputs must be scalar"));
            }
            let numeric = if arr.data[0] != 0 { 1.0 } else { 0.0 };
            parse_numeric(numeric, context)
        }
        other => Err(format!(
            "{context}: size inputs must be numeric scalars, got {other:?}"
        )),
    }
}

fn parse_size_tensor(t: &Tensor) -> Result<Vec<usize>, String> {
    if t.data.is_empty() {
        return Ok(vec![0, 0]);
    }
    if !is_vector_shape(&t.shape) {
        return Err("cell: size vector must be 1-D".to_string());
    }
    let dims = t
        .data
        .iter()
        .map(|&value| parse_numeric(value, "cell"))
        .collect::<Result<Vec<_>, _>>()?;
    if dims.len() == 1 {
        Ok(vec![dims[0], 1])
    } else {
        Ok(dims)
    }
}

fn parse_size_logical_array(arr: &LogicalArray) -> Result<Vec<usize>, String> {
    if arr.data.is_empty() {
        return Ok(vec![0, 0]);
    }
    if !is_vector_shape(&arr.shape) {
        return Err("cell: size vector must be 1-D".to_string());
    }
    let dims = arr
        .data
        .iter()
        .map(|&value| {
            let numeric = if value != 0 { 1.0 } else { 0.0 };
            parse_numeric(numeric, "cell")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dims.len() == 1 {
        Ok(vec![dims[0], 1])
    } else {
        Ok(dims)
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => false,
    }
}

fn empty_value_like(proto: Option<&Value>) -> Result<Value, String> {
    match proto {
        Some(value) => match value {
            Value::LogicalArray(_) | Value::Bool(_) => LogicalArray::new(Vec::new(), vec![0, 0])
                .map(Value::LogicalArray)
                .map_err(|e| format!("cell: {e}")),
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                ComplexTensor::new(Vec::new(), vec![0, 0])
                    .map(Value::ComplexTensor)
                    .map_err(|e| format!("cell: {e}"))
            }
            Value::String(_) => Ok(Value::String(String::new())),
            Value::StringArray(_) => StringArray::new(Vec::new(), vec![0, 0])
                .map(Value::StringArray)
                .map_err(|e| format!("cell: {e}")),
            Value::CharArray(_) => CharArray::new(Vec::new(), 0, 0)
                .map(Value::CharArray)
                .map_err(|e| format!("cell: {e}")),
            Value::Cell(_) => {
                make_cell_with_shape(Vec::new(), vec![0, 0]).map_err(|e| format!("cell: {e}"))
            }
            Value::Struct(_) => Ok(Value::Struct(StructValue::new())),
            Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::GpuTensor(_) => {
                default_empty_double()
            }
            Value::Object(_)
            | Value::HandleObject(_)
            | Value::Listener(_)
            | Value::FunctionHandle(_)
            | Value::Closure(_)
            | Value::ClassRef(_)
            | Value::MException(_) => default_empty_double(),
        },
        None => default_empty_double(),
    }
}

fn default_empty_double() -> Result<Value, String> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|e| format!("cell: {e}"))
}

fn parse_intvalue(value: &IntValue, context: &str) -> Result<usize, String> {
    let raw = match value {
        IntValue::I8(v) => *v as i128,
        IntValue::I16(v) => *v as i128,
        IntValue::I32(v) => *v as i128,
        IntValue::I64(v) => *v as i128,
        IntValue::U8(v) => *v as i128,
        IntValue::U16(v) => *v as i128,
        IntValue::U32(v) => *v as i128,
        IntValue::U64(v) => *v as i128,
    };
    if raw < 0 {
        return Err(format!(
            "{context}: size inputs must be non-negative integers"
        ));
    }
    if raw as u128 > usize::MAX as u128 {
        return Err("cell: requested size exceeds platform limits".to_string());
    }
    Ok(raw as usize)
}

fn parse_numeric(value: f64, context: &str) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(format!("{context}: size inputs must be finite"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(format!("{context}: size inputs must be integers"));
    }
    if rounded < 0.0 {
        return Err(format!(
            "{context}: size inputs must be non-negative integers"
        ));
    }
    if rounded > (1u64 << 53) as f64 {
        return Err("cell: size inputs larger than 2^53 are not supported".to_string());
    }
    if rounded > usize::MAX as f64 {
        return Err("cell: requested size exceeds platform limits".to_string());
    }
    Ok(rounded as usize)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    fn expect_cell_with<F>(value: Value, expected_shape: &[usize], mut check: F)
    where
        F: FnMut(&Value),
    {
        match value {
            Value::Cell(cell) => {
                assert_eq!(cell.shape, expected_shape, "shape mismatch");
                let expected_rows = expected_shape.first().copied().unwrap_or(0);
                let expected_cols = match expected_shape.len() {
                    0 => 0,
                    1 => 1,
                    _ => expected_shape[1],
                };
                assert_eq!(cell.rows, expected_rows, "rows mismatch");
                assert_eq!(cell.cols, expected_cols, "cols mismatch");
                let expected_total = expected_shape
                    .iter()
                    .fold(1usize, |acc, &dim| acc.saturating_mul(dim));
                let expected_total = if expected_shape.is_empty() {
                    0
                } else {
                    expected_total
                };
                assert_eq!(cell.data.len(), expected_total, "element count mismatch");
                for handle in cell.data {
                    let element = unsafe { &*handle.as_raw() };
                    check(element);
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    fn expect_cell(value: Value, expected_shape: &[usize]) {
        expect_cell_with(value, expected_shape, |element| match element {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty double array, found {other:?}"),
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_no_arguments_returns_empty() {
        let result = cell_builtin(Vec::new()).expect("cell()");
        expect_cell(result, &[0, 0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_scalar_creates_square() {
        let result = cell_builtin(vec![Value::Num(3.0)]).expect("cell(3)");
        expect_cell(result, &[3, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_two_sizes() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = cell_builtin(args).expect("cell(2,4)");
        expect_cell(result, &[2, 4]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_size_vector() {
        let tensor = Tensor::new(vec![2.0, 5.0], vec![1, 2]).unwrap();
        let result = cell_builtin(vec![Value::Tensor(tensor)]).expect("cell([2 5])");
        expect_cell(result, &[2, 5]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_column_size_vector() {
        let tensor = Tensor::new(vec![4.0, 1.0], vec![2, 1]).unwrap();
        let result = cell_builtin(vec![Value::Tensor(tensor)]).expect("cell([4; 1])");
        expect_cell(result, &[4, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_accepts_gpu_size_vector() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload size vector");
            let result = cell_builtin(vec![Value::GpuTensor(handle)]).expect("cell(gpu size)");
            expect_cell(result, &[3, 2]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cell_wgpu_size_vector_and_like() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![2.0, 3.0, 1.0], vec![1, 3]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload size vector");
        let result = cell_builtin(vec![Value::GpuTensor(handle)]).expect("cell(wgpu size)");
        expect_cell(result, &[2, 3, 1]);

        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let proto_view = runmat_accelerate_api::HostTensorView {
            data: &proto.data,
            shape: &proto.shape,
        };
        let proto_handle = provider.upload(&proto_view).expect("upload prototype");
        let like_result = cell_builtin(vec![Value::from("like"), Value::GpuTensor(proto_handle)])
            .expect("cell('like', gpu prototype)");
        expect_cell(like_result, &[2, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_multi_dimensional_vector() {
        let tensor = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let result = cell_builtin(vec![Value::Tensor(tensor)]).expect("cell([2 3 4])");
        expect_cell(result, &[2, 3, 4]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_variadic_dimensions() {
        let args = vec![Value::Num(2.0), Value::Num(3.0), Value::Num(5.0)];
        let result = cell_builtin(args).expect("cell(2,3,5)");
        expect_cell(result, &[2, 3, 5]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_single_element_vector_is_column() {
        let tensor = Tensor::new(vec![4.0], vec![1, 1]).unwrap();
        let result = cell_builtin(vec![Value::Tensor(tensor)]).expect("cell([4])");
        expect_cell(result, &[4, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_rejects_negative() {
        let err = cell_builtin(vec![Value::Num(-1.0)]).unwrap_err();
        assert!(err.contains("non-negative"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_rejects_fractional() {
        let err = cell_builtin(vec![Value::Num(2.5)]).unwrap_err();
        assert!(err.contains("integers"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_infers_shape_from_prototype() {
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = cell_builtin(args).expect("cell('like', tensor)");
        expect_cell(result, &[2, 2]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_logical_uses_logical_empty() {
        let logical = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let args = vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::LogicalArray(logical),
        ];
        let result = cell_builtin(args).expect("cell(___, 'like', logical)");
        expect_cell_with(result, &[2, 2], |element| match element {
            Value::LogicalArray(arr) => {
                assert!(arr.data.is_empty());
                assert_eq!(arr.shape, vec![0, 0]);
            }
            other => panic!("expected logical empty, got {other:?}"),
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_cell_prototype_produces_empty_cell_elements() {
        let proto = crate::make_cell_with_shape(Vec::new(), vec![0, 0]).unwrap();
        let args = vec![Value::Num(1.0), Value::from("like"), proto.clone()];
        let result = cell_builtin(args).expect("cell(1,'like',cell)");
        expect_cell_with(result, &[1, 1], |element| match element {
            Value::Cell(inner) => {
                assert_eq!(inner.shape, vec![0, 0]);
                assert_eq!(inner.data.len(), 0);
            }
            other => panic!("expected nested empty cell, got {other:?}"),
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_is_case_insensitive() {
        let proto = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let result = cell_builtin(vec![Value::from("LIKE"), Value::Tensor(proto)])
            .expect("cell('LIKE', ...)");
        expect_cell(result, &[1, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_requires_prototype() {
        let err = cell_builtin(vec![Value::from("like")]).unwrap_err();
        assert!(
            err.contains("expected prototype"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_rejects_multiple_keywords() {
        let proto = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = cell_builtin(vec![
            Value::Num(1.0),
            Value::from("like"),
            Value::Tensor(proto.clone()),
            Value::from("like"),
            Value::Tensor(proto),
        ])
        .unwrap_err();
        assert!(err.contains("multiple 'like'"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_gpu_prototype_falls_back_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload prototype");
            let result = cell_builtin(vec![Value::from("like"), Value::GpuTensor(handle)])
                .expect("cell('like', gpu)");
            expect_cell(result, &[2, 1]);
        });
    }
}
