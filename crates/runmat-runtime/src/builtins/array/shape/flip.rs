//! MATLAB-compatible `flip` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the `flip` function, mirroring MathWorks MATLAB
//! behaviour for numeric tensors, logical masks, string arrays, complex data,
//! character arrays, and gpuArray handles. It honours dimension vectors,
//! direction keywords such as `'horizontal'`, and gracefully falls back to the
//! host when a registered acceleration provider does not expose a native flip
//! kernel.

use crate::builtins::common::arg_tokens::{tokens_from_values, ArgToken};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    CharArray, ComplexTensor, LogicalArray, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::flip")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "flip",
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
    notes: "Providers may implement flip directly; the runtime falls back to gather→flip→upload when the hook is missing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::flip")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "flip",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Flip is a data-reordering boundary; fusion planner treats it as a residency-preserving barrier.",
};

fn preserve_array_type(args: &[Type], _context: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { shape } => Type::Tensor {
            shape: shape.clone(),
        },
        Type::Logical { shape } => Type::Logical {
            shape: shape.clone(),
        },
        Type::Num | Type::Int | Type::Bool => Type::tensor(),
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn flip_error_for(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(builtin).build()
}

#[runtime_builtin(
    name = "flip",
    category = "array/shape",
    summary = "Reverse the order of elements along specific dimensions.",
    keywords = "flip,reverse,dimension,gpu,horizontal,vertical",
    accel = "custom",
    type_resolver(preserve_array_type),
    builtin_path = "crate::builtins::array::shape::flip"
)]
async fn flip_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(flip_error_for("flip", "flip: too many input arguments"));
    }
    let spec = parse_flip_spec(&rest)?;
    match value {
        Value::Tensor(tensor) => {
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::LogicalArray(array) => {
            let dims = resolve_dims(&spec, &array.shape);
            Ok(flip_logical_array(array, &dims).map(Value::LogicalArray)?)
        }
        Value::ComplexTensor(ct) => {
            let dims = resolve_dims(&spec, &ct.shape);
            Ok(flip_complex_tensor(ct, &dims).map(Value::ComplexTensor)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| flip_error_for("flip", format!("flip: {e}")))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_complex_tensor(tensor, &dims).map(complex_tensor_into_value)?)
        }
        Value::StringArray(strings) => {
            let dims = resolve_dims(&spec, &strings.shape);
            Ok(flip_string_array(strings, &dims).map(Value::StringArray)?)
        }
        Value::CharArray(chars) => {
            let dims = resolve_dims(&spec, &[chars.rows, chars.cols]);
            Ok(flip_char_array(chars, &dims).map(Value::CharArray)?)
        }
        Value::String(scalar) => Ok(Value::String(scalar)),
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("flip", Value::Num(n))
                .map_err(|e| flip_error_for("flip", e))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("flip", Value::Int(i))
                .map_err(|e| flip_error_for("flip", e))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("flip", Value::Bool(flag))
                .map_err(|e| flip_error_for("flip", e))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            Ok(flip_tensor(tensor, &dims).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => {
            let dims = resolve_dims(&spec, &handle.shape);
            Ok(flip_gpu(handle, &dims).await?)
        }
        Value::Cell(_) => Err(flip_error_for(
            "flip",
            "flip: cell arrays are not yet supported",
        )),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(flip_error_for("flip", "flip: unsupported input type")),
    }
}

#[derive(Clone, Debug)]
enum FlipSpec {
    Default,
    Dims(Vec<usize>),
}

fn parse_flip_spec(args: &[Value]) -> crate::BuiltinResult<FlipSpec> {
    match args.len() {
        0 => Ok(FlipSpec::Default),
        1 => {
            let tokens = tokens_from_values(args);
            if let Some(token) = tokens.first() {
                if let Some(direction_dims) = parse_direction_token(token)? {
                    return Ok(FlipSpec::Dims(direction_dims));
                }
            }
            if let Some(direction_dims) = parse_direction(&args[0])? {
                return Ok(FlipSpec::Dims(direction_dims));
            }
            let dims = parse_dims_value(&args[0])?;
            if dims.is_empty() {
                Ok(FlipSpec::Default)
            } else {
                Ok(FlipSpec::Dims(dims))
            }
        }
        _ => unreachable!(),
    }
}

fn parse_direction_token(token: &ArgToken) -> crate::BuiltinResult<Option<Vec<usize>>> {
    let ArgToken::String(text) = token else {
        return Ok(None);
    };
    let dims = match text.as_str() {
        "horizontal" | "left-right" | "leftright" | "lr" | "right-left" | "righthoriz" => {
            vec![2]
        }
        "vertical" | "up-down" | "updown" | "ud" | "down-up" => vec![1],
        "both" => vec![1, 2],
        other => {
            return Err(flip_error_for(
                "flip",
                format!("flip: unknown direction '{other}'"),
            ));
        }
    };
    Ok(Some(dims))
}

fn parse_direction(value: &Value) -> crate::BuiltinResult<Option<Vec<usize>>> {
    let text_opt = match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            tensor::value_to_string(value)
        }
        _ => None,
    };
    if let Some(text) = text_opt {
        let lowered = text.trim().to_ascii_lowercase();
        let dims = match lowered.as_str() {
            "horizontal" | "left-right" | "leftright" | "lr" | "right-left" | "righthoriz" => {
                vec![2]
            }
            "vertical" | "up-down" | "updown" | "ud" | "down-up" => vec![1],
            "both" => vec![1, 2],
            other => {
                return Err(flip_error_for(
                    "flip",
                    format!("flip: unknown direction '{other}'"),
                ));
            }
        };
        return Ok(Some(dims));
    }
    Ok(None)
}

fn parse_dims_value(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => parse_dims_tensor(t),
        Value::LogicalArray(la) => {
            let tensor = tensor::logical_to_tensor(la).map_err(|e| {
                flip_error_for(
                    "flip",
                    format!("flip: unable to parse dimension vector: {e}"),
                )
            })?;
            parse_dims_tensor(&tensor)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let dim =
                tensor::parse_dimension(value, "flip").map_err(|e| flip_error_for("flip", e))?;
            Ok(vec![dim])
        }
        Value::GpuTensor(_) => Err(flip_error_for(
            "flip",
            "flip: dimension argument must be specified on the host (numeric or string)",
        )),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                let tmp = Value::StringArray(sa.clone());
                parse_direction(&tmp)?
                    .ok_or_else(|| flip_error_for("flip", "flip: dimension vector must be numeric"))
            } else {
                Err(flip_error_for(
                    "flip",
                    "flip: dimension vector must be numeric",
                ))
            }
        }
        Value::String(_) | Value::CharArray(_) => parse_direction(value)?
            .ok_or_else(|| flip_error_for("flip", "flip: unknown direction string")),
        _ => Err(flip_error_for(
            "flip",
            "flip: dimension vector must be numeric or a direction string",
        )),
    }
}

fn parse_dims_tensor(tensor: &Tensor) -> crate::BuiltinResult<Vec<usize>> {
    if !is_vector(&tensor.shape) {
        return Err(flip_error_for(
            "flip",
            "flip: dimension vector must be a row or column vector",
        ));
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for entry in &tensor.data {
        if !entry.is_finite() {
            return Err(flip_error_for(
                "flip",
                "flip: dimension indices must be finite",
            ));
        }
        let rounded = entry.round();
        if (rounded - entry).abs() > f64::EPSILON {
            return Err(flip_error_for(
                "flip",
                "flip: dimension indices must be integers",
            ));
        }
        if rounded < 1.0 {
            return Err(flip_error_for(
                "flip",
                "flip: dimension indices must be >= 1",
            ));
        }
        dims.push(rounded as usize);
    }
    Ok(dims)
}

fn is_vector(shape: &[usize]) -> bool {
    let mut non_singleton = 0usize;
    for &dim in shape {
        if dim > 1 {
            non_singleton += 1;
        }
        if non_singleton > 1 {
            return false;
        }
    }
    true
}

fn resolve_dims(spec: &FlipSpec, shape: &[usize]) -> Vec<usize> {
    match spec {
        FlipSpec::Default => vec![default_flip_dim(shape)],
        FlipSpec::Dims(dims) => dims.clone(),
    }
}

fn default_flip_dim(shape: &[usize]) -> usize {
    for (idx, &extent) in shape.iter().enumerate() {
        if extent > 1 {
            return idx + 1;
        }
    }
    1
}

pub(crate) fn flip_tensor(tensor: Tensor, dims: &[usize]) -> crate::BuiltinResult<Tensor> {
    flip_tensor_with("flip", tensor, dims)
}

pub(crate) fn flip_tensor_with(
    builtin: &'static str,
    tensor: Tensor,
    dims: &[usize],
) -> crate::BuiltinResult<Tensor> {
    if tensor.data.is_empty() || dims.is_empty() {
        return Ok(tensor);
    }
    let data = flip_generic(&tensor.data, &tensor.shape, dims, builtin)?;
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

pub(crate) fn flip_complex_tensor(
    tensor: ComplexTensor,
    dims: &[usize],
) -> crate::BuiltinResult<ComplexTensor> {
    flip_complex_tensor_with("flip", tensor, dims)
}

pub(crate) fn flip_complex_tensor_with(
    builtin: &'static str,
    tensor: ComplexTensor,
    dims: &[usize],
) -> crate::BuiltinResult<ComplexTensor> {
    if tensor.data.is_empty() || dims.is_empty() {
        return Ok(tensor);
    }
    let data = flip_generic(&tensor.data, &tensor.shape, dims, builtin)?;
    ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

pub(crate) fn flip_logical_array(
    array: LogicalArray,
    dims: &[usize],
) -> crate::BuiltinResult<LogicalArray> {
    flip_logical_array_with("flip", array, dims)
}

pub(crate) fn flip_logical_array_with(
    builtin: &'static str,
    array: LogicalArray,
    dims: &[usize],
) -> crate::BuiltinResult<LogicalArray> {
    if array.data.is_empty() || dims.is_empty() {
        return Ok(array);
    }
    let data = flip_generic(&array.data, &array.shape, dims, builtin)?;
    LogicalArray::new(data, array.shape.clone())
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

pub(crate) fn flip_string_array(
    array: StringArray,
    dims: &[usize],
) -> crate::BuiltinResult<StringArray> {
    flip_string_array_with("flip", array, dims)
}

pub(crate) fn flip_string_array_with(
    builtin: &'static str,
    array: StringArray,
    dims: &[usize],
) -> crate::BuiltinResult<StringArray> {
    if array.data.is_empty() || dims.is_empty() {
        return Ok(array);
    }
    let data = flip_generic(&array.data, &array.shape, dims, builtin)?;
    StringArray::new(data, array.shape.clone())
        .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

pub(crate) fn flip_char_array(array: CharArray, dims: &[usize]) -> crate::BuiltinResult<CharArray> {
    flip_char_array_with("flip", array, dims)
}

pub(crate) fn flip_char_array_with(
    builtin: &'static str,
    array: CharArray,
    dims: &[usize],
) -> crate::BuiltinResult<CharArray> {
    if array.data.is_empty() || dims.is_empty() {
        return Ok(array);
    }
    let rows = array.rows;
    let cols = array.cols;
    let mut flip_rows = false;
    let mut flip_cols = false;
    for &dim in dims {
        if dim == 0 {
            return Err(flip_error_for(
                builtin,
                format!("{builtin}: dimension must be >= 1"),
            ));
        }
        match dim {
            1 => flip_rows = !flip_rows,
            2 => flip_cols = !flip_cols,
            _ => {}
        }
    }
    if !flip_rows && !flip_cols {
        return Ok(array);
    }
    let mut out = vec!['\0'; array.data.len()];
    for row in 0..rows {
        for col in 0..cols {
            let dest_idx = row * cols + col;
            let src_row = if flip_rows { rows - 1 - row } else { row };
            let src_col = if flip_cols { cols - 1 - col } else { col };
            let src_idx = src_row * cols + src_col;
            out[dest_idx] = array.data[src_idx];
        }
    }
    CharArray::new(out, rows, cols).map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
}

pub(crate) async fn flip_gpu(
    handle: GpuTensorHandle,
    dims: &[usize],
) -> crate::BuiltinResult<Value> {
    flip_gpu_with("flip", handle, dims).await
}

pub(crate) async fn flip_gpu_with(
    builtin: &'static str,
    handle: GpuTensorHandle,
    dims: &[usize],
) -> crate::BuiltinResult<Value> {
    if dims.is_empty() {
        return Ok(Value::GpuTensor(handle));
    }
    if dims.contains(&0) {
        return Err(flip_error_for(
            builtin,
            format!("{builtin}: dimension indices must be >= 1"),
        ));
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        let zero_based: Vec<usize> = dims.iter().map(|&d| d - 1).collect();
        if let Ok(out) = provider.flip(&handle, &zero_based) {
            return Ok(Value::GpuTensor(out));
        }
        let host_tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        let flipped = flip_tensor_with(builtin, host_tensor, dims)?;
        let view = HostTensorView {
            data: &flipped.data,
            shape: &flipped.shape,
        };
        provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| flip_error_for(builtin, format!("{builtin}: {e}")))
    } else {
        let host_tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        flip_tensor_with(builtin, host_tensor, dims).map(tensor::tensor_into_value)
    }
}

fn flip_generic<T: Clone>(
    data: &[T],
    shape: &[usize],
    dims: &[usize],
    builtin: &'static str,
) -> crate::BuiltinResult<Vec<T>> {
    if dims.contains(&0) {
        return Err(flip_error_for(
            builtin,
            format!("{builtin}: dimension indices must be >= 1"),
        ));
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let max_dim = dims.iter().copied().max().unwrap_or(0);
    let mut ext_shape = shape.to_vec();
    if max_dim > ext_shape.len() {
        ext_shape.extend(std::iter::repeat_n(1, max_dim - ext_shape.len()));
    }
    let total: usize = ext_shape.iter().product();
    if total != data.len() {
        return Err(flip_error_for(
            builtin,
            format!("{builtin}: shape does not match data length"),
        ));
    }
    let mut flip_flags = vec![false; ext_shape.len()];
    for &dim in dims {
        let axis = dim - 1;
        if axis >= flip_flags.len() {
            continue;
        }
        flip_flags[axis] = !flip_flags[axis];
    }
    if !flip_flags.iter().any(|&flag| flag) {
        return Ok(data.to_vec());
    }
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let mut coords = unravel_index(idx, &ext_shape);
        for (axis, flag) in flip_flags.iter().enumerate() {
            if *flag && ext_shape[axis] > 1 {
                coords[axis] = ext_shape[axis] - 1 - coords[axis];
            }
        }
        let src_idx = ravel_index(&coords, &ext_shape);
        out.push(data[src_idx].clone());
    }
    Ok(out)
}

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for &extent in shape {
        if extent == 0 {
            coords.push(0);
        } else {
            coords.push(index % extent);
            index /= extent;
        }
    }
    coords
}

fn ravel_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut index = 0usize;
    let mut stride = 1usize;
    for (coord, extent) in coords.iter().zip(shape.iter()) {
        if *extent > 0 {
            index += coord * stride;
            stride *= extent;
        }
    }
    index
}

pub(crate) fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn flip_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::flip_builtin(value, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, StringArray, Tensor};

    #[test]
    fn flip_type_preserves_logical_shape() {
        let out = preserve_array_type(
            &[Type::Logical {
                shape: Some(vec![Some(2), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Logical {
                shape: Some(vec![Some(2), Some(1)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_vector_defaults_to_first_non_singleton_dim() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let value =
            flip_builtin(Value::Tensor(tensor), Vec::new()).expect("flip row vector default");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, vec![4.0, 3.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_matrix_vertical_default() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).expect("tensor");
        let value = flip_builtin(Value::Tensor(tensor), Vec::new()).expect("flip matrix");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.data, vec![2.0, 4.0, 1.0, 6.0, 3.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_horizontal_keyword() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).expect("tensor");
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::from("horizontal")])
            .expect("flip horizontal");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, vec![5.0, 3.0, 6.0, 1.0, 4.0, 2.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_multiple_dimensions() {
        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let value = flip_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap(),
            )],
        )
        .expect("flip dims");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2, 2]);
                assert_eq!(t.data, vec![6.0, 5.0, 8.0, 7.0, 2.0, 1.0, 4.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_both_direction_keyword() {
        let tensor = Tensor::new((1..=6).map(|v| v as f64).collect(), vec![3, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &[1, 2]).expect("cpu flip");
        let value =
            flip_builtin(Value::Tensor(tensor), vec![Value::from("both")]).expect("flip both");
        match value {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_char_array_horizontal() {
        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let value =
            flip_builtin(Value::CharArray(chars), vec![Value::from("horizontal")]).expect("flip");
        match value {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 3);
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "nurtam");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_direction_accepts_char_array_keyword() {
        let keyword = CharArray::new_row("vertical");
        let tensor = Tensor::new((1..=4).map(|v| v as f64).collect(), vec![2, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &[1]).expect("cpu flip");
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::CharArray(keyword)])
            .expect("flip via char");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_logical_array_preserves_type() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let expected = flip_logical_array(logical.clone(), &[2]).expect("cpu logical flip");
        let value = flip_builtin(
            Value::LogicalArray(logical),
            vec![Value::from("horizontal")],
        )
        .expect("flip logical");
        match value {
            Value::LogicalArray(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_complex_tensor_defaults_to_first_dim() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (2.0, -1.0), (3.0, 0.5), (4.0, -0.25)],
            vec![2, 2],
        )
        .unwrap();
        let expected = flip_complex_tensor(tensor.clone(), &[1]).expect("cpu complex flip");
        let value = flip_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("flip complex");
        match value {
            Value::ComplexTensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_string_array_vertical() {
        let strings =
            StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).expect("string array");
        let value =
            flip_builtin(Value::StringArray(strings), vec![Value::from("vertical")]).expect("flip");
        match value {
            Value::StringArray(out) => {
                assert_eq!(out.data, vec!["b".to_string(), "a".to_string()])
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_accepts_dimension_vector_tensor() {
        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let value =
            flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("flip dims");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_dimension_tensor_must_be_vector() {
        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let dims = Tensor::new((1..=4).map(|v| v as f64).collect(), vec![2, 2]).unwrap();
        let err =
            flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect_err("flip fail");
        assert!(err.to_string().contains("row or column vector"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_dimensions_beyond_rank_are_noops() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let original = tensor.clone();
        let dims = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("flip");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, original.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_rejects_zero_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = flip_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
            .expect_err("flip should fail");
        assert!(err.to_string().contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flip_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cpu =
                flip_tensor(tensor.clone(), &[default_flip_dim(&tensor.shape)]).expect("cpu flip");
            let value = flip_builtin(Value::GpuTensor(handle), Vec::new()).expect("flip gpu");
            let gathered = test_support::gather(value).expect("gather gpu result");
            assert_eq!(gathered.data, cpu.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn flip_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).expect("tensor");
        let cpu = flip_tensor(tensor.clone(), &[1, 3]).expect("cpu flip");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_value = flip_builtin(
            Value::GpuTensor(handle),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap(),
            )],
        )
        .expect("flip gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.data, cpu.data);
    }
}
