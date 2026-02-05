//! MATLAB-compatible `cat` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::arg_tokens::{ArgToken, tokens_from_context, tokens_from_values};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::concatenation::char_array_from_f64_with_prefix;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, ResolveContext, StringArray, Tensor, Type,
    Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::cat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cat",
    op_kind: GpuOpKind::Custom("cat"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("cat")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Falls back to gather + upload when providers lack a native concatenation kernel.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::cat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cat",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Concatenation is a sink and terminates fusion pipelines.",
};

fn cat_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("cat").build()
}

#[derive(Clone, Copy, Debug)]
struct ParsedCatTokens {
    dim: Option<usize>,
    like_index: Option<usize>,
}

fn parse_cat_tokens(tokens: &[ArgToken]) -> ParsedCatTokens {
    let dim = match tokens.first() {
        Some(ArgToken::Number(value)) => coerce_positive_dim(*value),
        _ => None,
    };
    let like_index = if tokens.len() >= 3 {
        match &tokens[tokens.len() - 2] {
            ArgToken::String(text) if text == "like" => Some(tokens.len() - 2),
            _ => None,
        }
    } else {
        None
    };
    ParsedCatTokens { dim, like_index }
}

fn coerce_positive_dim(value: f64) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return None;
    }
    if rounded < 1.0 {
        return None;
    }
    Some(rounded as usize)
}

fn cell_element_type(inputs: &[Type]) -> Option<Box<Type>> {
    let mut element: Option<Type> = None;
    for ty in inputs {
        let Type::Cell { element_type, .. } = ty else {
            return None;
        };
        match (&element, element_type.as_deref()) {
            (None, Some(current)) => element = Some(current.clone()),
            (Some(existing), Some(current)) if existing == current => {}
            (Some(_), Some(_)) => return None,
            _ => {}
        }
    }
    element.map(Box::new)
}

fn cat_input_shape(ty: &Type) -> Option<Vec<Option<usize>>> {
    match ty {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            Some(shape.clone())
        }
        Type::Num | Type::Int | Type::Bool => Some(vec![Some(1), Some(1)]),
        _ => None,
    }
}

fn cat_concat_shape(inputs: &[Type], dim_1based: usize) -> Option<Vec<Option<usize>>> {
    if inputs.is_empty() || dim_1based == 0 {
        return None;
    }
    let mut shapes = Vec::with_capacity(inputs.len());
    for ty in inputs {
        shapes.push(cat_input_shape(ty)?);
    }
    let rank = shapes
        .iter()
        .map(|shape| shape.len())
        .max()?
        .max(dim_1based);
    let mut padded = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let mut current = shape;
        while current.len() < rank {
            current.push(Some(1));
        }
        padded.push(current);
    }
    let mut output = vec![None; rank];
    let dim_zero = dim_1based - 1;
    for axis in 0..rank {
        if axis == dim_zero {
            let mut total: Option<usize> = Some(0);
            for shape in &padded {
                match (total, shape[axis]) {
                    (Some(acc), Some(value)) => total = acc.checked_add(value),
                    _ => {
                        total = None;
                        break;
                    }
                }
            }
            output[axis] = total;
        } else {
            let mut shared: Option<usize> = None;
            let mut mismatch = false;
            for shape in &padded {
                match (shared, shape[axis]) {
                    (None, value) => shared = value,
                    (Some(current), Some(value)) if current == value => {}
                    (Some(_), Some(_)) => {
                        mismatch = true;
                        break;
                    }
                    _ => {
                        shared = None;
                        break;
                    }
                }
            }
            output[axis] = if mismatch { None } else { shared };
        }
    }
    let min_len = dim_1based.max(2).min(output.len());
    while output.len() > min_len && matches!(output.last(), Some(Some(1))) {
        output.pop();
    }
    Some(output)
}

fn cat_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.len() < 3 {
        return Type::Unknown;
    }
    let parsed = parse_cat_tokens(&tokens_from_context(ctx));
    let inputs = &args[1..];
    let all_cells = inputs.iter().all(|arg| matches!(arg, Type::Cell { .. }));
    if all_cells {
        return Type::Cell {
            element_type: cell_element_type(inputs),
            length: None,
        };
    }
    let all_strings = inputs.iter().all(|arg| matches!(arg, Type::String));
    if all_strings {
        return Type::cell_of(Type::String);
    }
    let has_numeric = inputs
        .iter()
        .any(|arg| matches!(arg, Type::Tensor { .. } | Type::Num | Type::Int));
    let has_logical = inputs
        .iter()
        .any(|arg| matches!(arg, Type::Logical { .. } | Type::Bool));
    let dim = match parsed.dim {
        Some(value) if value > 0 => value,
        Some(_) => return Type::Unknown,
        None => {
            if has_numeric {
                return Type::tensor();
            }
            if has_logical {
                return Type::logical();
            }
            return Type::Unknown;
        }
    };
    let shape = cat_concat_shape(inputs, dim);
    if has_numeric {
        return Type::Tensor { shape };
    }
    if has_logical {
        return Type::Logical { shape };
    }
    Type::Unknown
}

fn cat_err(message: impl Into<String>) -> RuntimeError {
    cat_error(message)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CatCategory {
    Numeric,
    Logical,
    Complex,
    Char,
    String,
    Cell,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LikeDevice {
    Host,
    Gpu,
}

#[derive(Clone, Debug)]
struct LikeSpec {
    device: LikeDevice,
    category_hint: Option<CatCategory>,
}

impl Default for LikeSpec {
    fn default() -> Self {
        Self {
            device: LikeDevice::Host,
            category_hint: None,
        }
    }
}

impl LikeSpec {
    fn from_prototype(proto: Value) -> BuiltinResult<Self> {
        match proto {
            Value::GpuTensor(_) => Ok(Self {
                device: LikeDevice::Gpu,
                category_hint: Some(CatCategory::Numeric),
            }),
            Value::Tensor(_) | Value::Num(_) | Value::Int(_) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Numeric),
            }),
            Value::LogicalArray(_) | Value::Bool(_) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Logical),
            }),
            Value::ComplexTensor(_) | Value::Complex(_, _) => Ok(Self {
                device: LikeDevice::Host,
                category_hint: Some(CatCategory::Complex),
            }),
            other => Err(cat_err(format!(
                "cat: unsupported prototype for 'like' ({other:?}); provide a numeric or gpuArray prototype"
            ))),
        }
    }

    fn ensure_device(&self, category: CatCategory) -> BuiltinResult<()> {
        if matches!(self.device, LikeDevice::Gpu) && !matches!(category, CatCategory::Numeric) {
            return Err(cat_err(
                "cat: GPU 'like' prototypes are only supported for numeric inputs",
            ));
        }
        Ok(())
    }
}

fn extract_like(mut inputs: Vec<Value>) -> BuiltinResult<(Vec<Value>, LikeSpec)> {
    if inputs.len() >= 2 {
        let tokens = tokens_from_values(&inputs);
        let parsed = parse_cat_tokens(&tokens);
        if parsed.like_index == Some(inputs.len() - 2) {
            let prototype = inputs.last().cloned().unwrap();
            if matches!(
                prototype,
                Value::CharArray(_) | Value::String(_) | Value::StringArray(_) | Value::Cell(_)
            ) {
                // Treat as data to avoid colliding with textual concatenation cases.
            } else if inputs.len() < 4 {
                // Removing the pair would leave fewer than two inputs; treat as data.
            } else {
                let spec = LikeSpec::from_prototype(prototype)?;
                inputs.pop();
                inputs.pop();
                return Ok((inputs, spec));
            }
        }
    }
    Ok((inputs, LikeSpec::default()))
}

#[runtime_builtin(
    name = "cat",
    category = "array/shape",
    summary = "Concatenate arrays along a specified dimension while preserving MATLAB semantics.",
    keywords = "cat,concatenate,array,dimension,gpu",
    accel = "array_construct",
    type_resolver(cat_type),
    builtin_path = "crate::builtins::array::shape::cat"
)]
async fn cat_builtin(dim: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() < 2 {
        return Err(cat_err("cat: at least two input arrays are required"));
    }
    let dim_index = match dim {
        Value::Int(_) | Value::Num(_) | Value::GpuTensor(_) => {
            match tensor::dimension_from_value_async(&dim, "cat", false)
                .await
                .map_err(cat_err)?
            {
                Some(index) => index,
                None => {
                    return Err(cat_err(format!(
                        "cat: dimension must be numeric, got {:?}",
                        dim
                    )))
                }
            }
        }
        _ => {
            return Err(cat_err(format!(
                "cat: dimension must be numeric, got {:?}",
                dim
            )))
        }
    };
    let dim_zero = dim_index - 1;

    let (inputs, like) = extract_like(rest)?;
    if inputs.len() < 2 {
        return Err(cat_err("cat: at least two input arrays are required"));
    }

    if inputs.iter().any(|v| matches!(v, Value::GpuTensor(_))) {
        if !inputs.iter().all(|v| matches!(v, Value::GpuTensor(_))) {
            return Err(cat_err(
                "cat: cannot mix gpuArray inputs with host arrays; convert them first",
            ));
        }
        return cat_gpu_tensors(dim_zero, inputs, &like).await;
    }

    let category = determine_category(&inputs, &like)?;
    match category {
        CatCategory::String => cat_string_arrays(dim_zero, inputs),
        CatCategory::Char => cat_char_arrays(dim_zero, inputs),
        CatCategory::Cell => cat_cell_arrays(dim_zero, inputs),
        CatCategory::Logical => cat_logical_arrays(dim_zero, inputs, &like),
        CatCategory::Complex => cat_complex_arrays(dim_zero, inputs, &like),
        CatCategory::Numeric => cat_numeric_tensors(dim_zero, inputs, &like),
    }
}

fn determine_category(inputs: &[Value], like: &LikeSpec) -> BuiltinResult<CatCategory> {
    let mut category = infer_category(inputs)?;
    if let Some(hint) = like.category_hint {
        category = match hint {
            CatCategory::Numeric => {
                if matches!(
                    category,
                    CatCategory::String
                        | CatCategory::Char
                        | CatCategory::Cell
                        | CatCategory::Complex
                ) {
                    return Err(cat_err(
                        "cat: 'like' prototype class does not match the input classes",
                    ));
                }
                CatCategory::Numeric
            }
            CatCategory::Logical => {
                if !matches!(category, CatCategory::Logical) {
                    return Err(cat_err(
                        "cat: 'like' logical prototypes require logical inputs",
                    ));
                }
                CatCategory::Logical
            }
            CatCategory::Complex => {
                if matches!(
                    category,
                    CatCategory::String | CatCategory::Char | CatCategory::Cell
                ) {
                    return Err(cat_err(
                        "cat: 'like' complex prototypes require numeric or complex inputs",
                    ));
                }
                CatCategory::Complex
            }
            CatCategory::Char | CatCategory::String | CatCategory::Cell => {
                return Err(cat_err(
                    "cat: 'like' prototypes for char, string, or cell arrays are not supported",
                ));
            }
        };
    }
    like.ensure_device(category)?;
    Ok(category)
}

fn infer_category(inputs: &[Value]) -> BuiltinResult<CatCategory> {
    let mut has_string = false;
    let mut has_char = false;
    let mut has_cell = false;
    let mut has_complex = false;
    let mut has_numeric = false;
    let mut all_logical = true;

    for value in inputs {
        match value {
            Value::Tensor(_) | Value::Num(_) | Value::Int(_) => {
                has_numeric = true;
                all_logical = false;
            }
            Value::LogicalArray(_) | Value::Bool(_) => {
                has_numeric = true;
            }
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                has_complex = true;
                has_numeric = true;
                all_logical = false;
            }
            Value::String(_) | Value::StringArray(_) => {
                has_string = true;
                all_logical = false;
            }
            Value::CharArray(_) => {
                has_char = true;
                all_logical = false;
            }
            Value::Cell(_) => {
                has_cell = true;
                all_logical = false;
            }
            Value::GpuTensor(_) => {
                return Err(cat_err(
                    "cat: gpuArray inputs must be concatenated using the GPU path",
                ));
            }
            other => {
                return Err(cat_err(format!(
                    "cat: unsupported input type for concatenation: {other:?}"
                )));
            }
        }
        if !matches!(value, Value::LogicalArray(_) | Value::Bool(_)) {
            all_logical = false;
        }
    }

    if has_string && (has_char || has_cell || has_complex || (has_numeric && !all_logical)) {
        return Err(cat_err("cat: cannot mix string arrays with other classes"));
    }
    if has_char && (has_cell || has_complex || has_string) {
        return Err(cat_err("cat: cannot mix char arrays with other classes"));
    }
    if has_cell && (has_complex || (has_numeric && !all_logical) || has_string || has_char) {
        return Err(cat_err("cat: cannot mix cell arrays with other classes"));
    }
    if has_complex && (has_string || has_char || has_cell) {
        return Err(cat_err(
            "cat: cannot mix complex arrays with textual or cell arrays",
        ));
    }

    if has_string {
        Ok(CatCategory::String)
    } else if has_char {
        Ok(CatCategory::Char)
    } else if has_cell {
        Ok(CatCategory::Cell)
    } else if has_complex {
        Ok(CatCategory::Complex)
    } else if all_logical {
        Ok(CatCategory::Logical)
    } else {
        Ok(CatCategory::Numeric)
    }
}

fn finalize_numeric_output(tensor: Tensor, like: &LikeSpec) -> BuiltinResult<Value> {
    like.ensure_device(CatCategory::Numeric)?;
    match like.device {
        LikeDevice::Host => Ok(tensor::tensor_into_value(tensor)),
        LikeDevice::Gpu => {
            let provider = runmat_accelerate_api::provider().ok_or_else(|| {
                cat_err(
                    "cat: GPU output requested via 'like' but no acceleration provider is active",
                )
            })?;
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).map_err(|err| {
                cat_err(format!("cat: failed to upload concatenated tensor: {err}"))
            })?;
            Ok(Value::GpuTensor(handle))
        }
    }
}

fn cat_numeric_tensors(
    dim_zero: usize,
    values: Vec<Value>,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    let mut tensors = Vec::with_capacity(values.len());
    for value in values {
        let tensor = tensor::value_into_tensor_for("cat", value).map_err(cat_err)?;
        tensors.push(tensor);
    }

    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).collect();
    let data_refs: Vec<&[f64]> = tensors.iter().map(|t| t.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let tensor = Tensor::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    finalize_numeric_output(tensor, like)
}

fn cat_logical_arrays(
    dim_zero: usize,
    values: Vec<Value>,
    _like: &LikeSpec,
) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(value_into_logical(value)?);
    }
    let shapes: Vec<Vec<usize>> = arrays.iter().map(|a| a.shape.clone()).collect();
    let data_refs: Vec<&[u8]> = arrays.iter().map(|a| a.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let logical = LogicalArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::LogicalArray(logical))
}

fn cat_complex_arrays(
    dim_zero: usize,
    values: Vec<Value>,
    _like: &LikeSpec,
) -> BuiltinResult<Value> {
    if values.iter().any(|v| matches!(v, Value::GpuTensor(_))) {
        return Err(cat_err(
            "cat: complex concatenation requires host arrays; convert gpuArray values first",
        ));
    }

    let mut tensors = Vec::with_capacity(values.len());
    for value in values {
        let tensor = match value {
            Value::ComplexTensor(ct) => ct,
            Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cat_err(format!("cat: {e}")))?,
            other => {
                let real = tensor::value_into_tensor_for("cat", other).map_err(cat_err)?;
                tensor_to_complex(real)?
            }
        };
        tensors.push(tensor);
    }

    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).collect();
    let data_refs: Vec<&[(f64, f64)]> = tensors.iter().map(|t| t.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let tensor = ComplexTensor::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn cat_char_arrays(dim_zero: usize, values: Vec<Value>) -> BuiltinResult<Value> {
    if dim_zero > 1 {
        return Err(cat_err("cat: char arrays only support dimensions 1 or 2"));
    }
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(char_array_from_value(value)?);
    }
    match dim_zero {
        0 => concat_char_rows(arrays),
        _ => concat_char_cols(arrays),
    }
}

fn char_array_from_value(value: Value) -> BuiltinResult<CharArray> {
    match value {
        Value::CharArray(ca) => Ok(ca),
        Value::Num(n) => char_array_from_f64_with_prefix(n, "cat"),
        Value::Int(i) => char_array_from_f64_with_prefix(i.to_f64(), "cat"),
        Value::Bool(flag) => char_array_from_f64_with_prefix(if flag { 1.0 } else { 0.0 }, "cat"),
        other => Err(cat_err(format!(
            "cat: expected char arrays or scalar code points, got {other:?}"
        ))),
    }
}

fn concat_char_rows(arrays: Vec<CharArray>) -> BuiltinResult<Value> {
    let cols = arrays.first().map(|a| a.cols).unwrap_or(0);
    for (idx, arr) in arrays.iter().enumerate() {
        if arr.cols != cols {
            return Err(cat_err(format!(
                "cat: dimension 2 mismatch between input 1 (size {}) and input {} (size {})",
                cols,
                idx + 1,
                arr.cols
            )));
        }
    }
    let total_rows = arrays.iter().map(|a| a.rows).sum();
    if total_rows == 0 || cols == 0 {
        let data = Vec::new();
        let result =
            CharArray::new(data, total_rows, cols).map_err(|e| cat_err(format!("cat: {e}")))?;
        return Ok(Value::CharArray(result));
    }
    let mut data = Vec::with_capacity(total_rows * cols);
    for arr in arrays {
        data.extend_from_slice(&arr.data);
    }
    let result =
        CharArray::new(data, total_rows, cols).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::CharArray(result))
}

fn concat_char_cols(arrays: Vec<CharArray>) -> BuiltinResult<Value> {
    let rows = arrays.first().map(|a| a.rows).unwrap_or(0);
    for (idx, arr) in arrays.iter().enumerate() {
        if arr.rows != rows {
            return Err(cat_err(format!(
                "cat: dimension 1 mismatch between input 1 (size {}) and input {} (size {})",
                rows,
                idx + 1,
                arr.rows
            )));
        }
    }
    let total_cols = arrays.iter().map(|a| a.cols).sum();
    if total_cols == 0 || rows == 0 {
        let data = Vec::new();
        let result =
            CharArray::new(data, rows, total_cols).map_err(|e| cat_err(format!("cat: {e}")))?;
        return Ok(Value::CharArray(result));
    }
    let mut data = Vec::with_capacity(rows * total_cols);
    for row in 0..rows {
        for arr in &arrays {
            for col in 0..arr.cols {
                let idx = row * arr.cols + col;
                data.push(arr.data[idx]);
            }
        }
    }
    let result =
        CharArray::new(data, rows, total_cols).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::CharArray(result))
}

fn cat_string_arrays(dim_zero: usize, values: Vec<Value>) -> BuiltinResult<Value> {
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        arrays.push(value_into_string_array(value)?);
    }
    let shapes: Vec<Vec<usize>> = arrays.iter().map(|a| a.shape.clone()).collect();
    let data_refs: Vec<&[String]> = arrays.iter().map(|a| a.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let array = StringArray::new(data, shape).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::StringArray(array))
}

fn cat_cell_arrays(dim_zero: usize, values: Vec<Value>) -> BuiltinResult<Value> {
    if dim_zero > 1 {
        return Err(cat_err(
            "cat: cell arrays only support concatenation along dimensions 1 or 2",
        ));
    }
    let mut arrays = Vec::with_capacity(values.len());
    for value in values {
        if let Value::Cell(cell) = value {
            arrays.push(cell);
        } else {
            return Err(cat_err("cat: expected cell arrays"));
        }
    }
    match dim_zero {
        0 => concat_cell_rows(arrays),
        _ => concat_cell_cols(arrays),
    }
}

fn concat_cell_rows(arrays: Vec<CellArray>) -> BuiltinResult<Value> {
    let cols = arrays.first().map(|a| a.cols).unwrap_or(0);
    for (idx, arr) in arrays.iter().enumerate() {
        if arr.cols != cols {
            return Err(cat_err(format!(
                "cat: dimension 2 mismatch between input 1 (size {}) and input {} (size {})",
                cols,
                idx + 1,
                arr.cols
            )));
        }
    }
    let total_rows = arrays.iter().map(|a| a.rows).sum();
    if total_rows == 0 || cols == 0 {
        let cell = CellArray::new(Vec::new(), total_rows, cols)
            .map_err(|e| cat_err(format!("cat: {e}")))?;
        return Ok(Value::Cell(cell));
    }
    let mut values = Vec::with_capacity(total_rows * cols);
    for arr in arrays {
        for handle in arr.data {
            let value = unsafe { &*handle.as_raw() }.clone();
            values.push(value);
        }
    }
    let cell =
        CellArray::new(values, total_rows, cols).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::Cell(cell))
}

fn concat_cell_cols(arrays: Vec<CellArray>) -> BuiltinResult<Value> {
    let rows = arrays.first().map(|a| a.rows).unwrap_or(0);
    for (idx, arr) in arrays.iter().enumerate() {
        if arr.rows != rows {
            return Err(cat_err(format!(
                "cat: dimension 1 mismatch between input 1 (size {}) and input {} (size {})",
                rows,
                idx + 1,
                arr.rows
            )));
        }
    }
    let total_cols = arrays.iter().map(|a| a.cols).sum();
    if total_cols == 0 || rows == 0 {
        let cell = CellArray::new(Vec::new(), rows, total_cols)
            .map_err(|e| cat_err(format!("cat: {e}")))?;
        return Ok(Value::Cell(cell));
    }
    let mut values = Vec::with_capacity(rows * total_cols);
    for row in 0..rows {
        for arr in &arrays {
            for col in 0..arr.cols {
                let idx = row * arr.cols + col;
                let value = unsafe { &*arr.data[idx].as_raw() }.clone();
                values.push(value);
            }
        }
    }
    let cell =
        CellArray::new(values, rows, total_cols).map_err(|e| cat_err(format!("cat: {e}")))?;
    Ok(Value::Cell(cell))
}

async fn cat_gpu_tensors(
    dim_zero: usize,
    values: Vec<Value>,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    if let Some(hint) = like.category_hint {
        if !matches!(hint, CatCategory::Numeric) {
            return Err(cat_err(
                "cat: 'like' prototype class does not match gpuArray inputs",
            ));
        }
    }
    like.ensure_device(CatCategory::Numeric)?;

    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| cat_err("cat: no acceleration provider is registered"))?;

    let mut handles = Vec::with_capacity(values.len());
    for value in values {
        if let Value::GpuTensor(handle) = value {
            handles.push(handle);
        }
    }

    // Native provider hook
    if let Ok(result) = provider.cat(dim_zero + 1, &handles) {
        return finalize_gpu_value(result, like).await;
    }

    let mut tensors = Vec::with_capacity(handles.len());
    for handle in &handles {
        let tensor = gpu_helpers::gather_tensor_async(handle).await?;
        tensors.push(tensor);
    }

    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape.clone()).collect();
    let data_refs: Vec<&[f64]> = tensors.iter().map(|t| t.data.as_slice()).collect();
    let (data, shape) = concat_column_major(dim_zero, &shapes, &data_refs, "cat")?;
    let tensor = Tensor::new(data, shape.clone()).map_err(|e| cat_err(format!("cat: {e}")))?;
    if matches!(like.device, LikeDevice::Host) {
        return Ok(tensor::tensor_into_value(tensor));
    }

    let view = HostTensorView {
        data: &tensor.data,
        shape: &shape,
    };
    match provider.upload(&view) {
        Ok(handle) => Ok(Value::GpuTensor(handle)),
        Err(_) => Ok(tensor::tensor_into_value(tensor)),
    }
}

fn concat_column_major<T: Clone>(
    dim_zero: usize,
    shapes: &[Vec<usize>],
    data: &[&[T]],
    context: &str,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    if shapes.is_empty() {
        return Err(cat_err(format!("{context}: no inputs to concatenate")));
    }
    let rank = shapes
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(1)
        .max(dim_zero + 1);

    let mut padded = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let mut current = shape.clone();
        while current.len() < rank {
            current.push(1);
        }
        padded.push(current);
    }

    for (idx, (shape, slice)) in padded.iter().zip(data.iter()).enumerate() {
        let expected = checked_product(shape)
            .ok_or_else(|| cat_err(format!("{context}: input {} exceeds maximum size", idx + 1)))?;
        if expected != slice.len() {
            return Err(cat_err(format!(
                "{context}: input {} has {} elements but the shape multiplies to {}",
                idx + 1,
                slice.len(),
                expected
            )));
        }
    }

    for axis in 0..rank {
        if axis == dim_zero {
            continue;
        }
        let reference = padded[0][axis];
        for (idx, shape) in padded.iter().enumerate().skip(1) {
            if shape[axis] != reference {
                return Err(cat_err(format!(
                    "{context}: dimension {} mismatch between input 1 (size {}) and input {} (size {})",
                    axis + 1,
                    reference,
                    idx + 1,
                    shape[axis]
                )));
            }
        }
    }

    let mut output_shape = padded[0].clone();
    let mut concat_dim = 0usize;
    for shape in &padded {
        concat_dim = concat_dim.checked_add(shape[dim_zero]).ok_or_else(|| {
            cat_err(format!(
                "{context}: concatenated dimension exceeds maximum size"
            ))
        })?;
    }
    output_shape[dim_zero] = concat_dim;

    let total = match checked_product(&output_shape) {
        Some(total) => total,
        None => {
            return Err(cat_err(format!(
                "{context}: resulting array exceeds maximum size"
            )))
        }
    };
    if total == 0 {
        return Ok((Vec::new(), normalize_shape(output_shape, dim_zero)));
    }

    let inner = if dim_zero == 0 {
        1
    } else {
        output_shape[..dim_zero].iter().product()
    };
    let outer = if dim_zero + 1 >= rank {
        1
    } else {
        output_shape[dim_zero + 1..].iter().product()
    };

    let mut output = Vec::with_capacity(total);
    for outer_idx in 0..outer {
        for (shape, slice) in padded.iter().zip(data.iter()) {
            let mid = shape[dim_zero];
            let chunk = mid * inner;
            if chunk == 0 {
                continue;
            }
            let offset = outer_idx * chunk;
            output.extend_from_slice(&slice[offset..offset + chunk]);
        }
    }

    Ok((output, normalize_shape(output_shape, dim_zero)))
}

fn normalize_shape(mut shape: Vec<usize>, dim_zero: usize) -> Vec<usize> {
    let min_len = (dim_zero + 1).max(2).min(shape.len());
    while shape.len() > min_len && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn checked_product(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn value_into_logical(value: Value) -> BuiltinResult<LogicalArray> {
    match value {
        Value::LogicalArray(array) => Ok(array),
        Value::Bool(flag) => LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
            .map_err(|e| cat_err(format!("cat: {e}"))),
        other => Err(cat_err(format!(
            "cat: expected logical inputs, got {:?}",
            other
        ))),
    }
}

fn value_into_string_array(value: Value) -> BuiltinResult<StringArray> {
    match value {
        Value::StringArray(array) => Ok(array),
        Value::String(text) => {
            StringArray::new(vec![text], vec![1, 1]).map_err(|e| cat_err(format!("cat: {e}")))
        }
        other => Err(cat_err(format!(
            "cat: expected string arrays, got {:?}",
            other
        ))),
    }
}

fn tensor_to_complex(tensor: Tensor) -> BuiltinResult<ComplexTensor> {
    let data = tensor.data.into_iter().map(|re| (re, 0.0)).collect();
    ComplexTensor::new(data, tensor.shape).map_err(|e| cat_err(format!("cat: {e}")))
}

async fn finalize_gpu_value(
    handle: runmat_accelerate_api::GpuTensorHandle,
    like: &LikeSpec,
) -> BuiltinResult<Value> {
    if matches!(like.device, LikeDevice::Host) {
        let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
        return Ok(tensor::tensor_into_value(tensor));
    }
    Ok(Value::GpuTensor(handle))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn cat_builtin(dim: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::cat_builtin(dim, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor};

    #[test]
    fn cat_type_prefers_cell() {
        let out = cat_type(
            &[
            Type::Int,
            Type::Cell {
                element_type: Some(Box::new(Type::Num)),
                length: None,
            },
            Type::Cell {
                element_type: Some(Box::new(Type::Num)),
                length: None,
            },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert!(matches!(out, Type::Cell { .. }));
    }

    #[test]
    fn cat_type_numeric_falls_back_tensor() {
        let out = cat_type(
            &[Type::Int, Type::Num, Type::Num],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::tensor());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_numeric_rows() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .expect("cat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 2]);
                assert_eq!(t.data, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_dimension_mismatch_errors() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3, 1]).unwrap();
        let err = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(a), Value::Tensor(b)],
        )
        .unwrap_err();
        assert!(err.message().contains("dimension 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_char_columns() {
        let left = CharArray::new("Run".chars().collect(), 1, 3).unwrap();
        let right = CharArray::new("Mat".chars().collect(), 1, 3).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::CharArray(left), Value::CharArray(right)],
        )
        .expect("cat");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 6);
                let text: String = arr.data.iter().collect();
                assert_eq!(text, "RunMat");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_logical_preserves_type() {
        let top = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let bottom = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let result = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![Value::LogicalArray(top), Value::LogicalArray(bottom)],
        )
        .expect("cat");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![2, 3]);
                assert_eq!(arr.data, vec![1, 0, 0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let view_a = HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.data,
                shape: &b.shape,
            };
            let ha = provider.upload(&view_a).expect("upload a");
            let hb = provider.upload(&view_b).expect("upload b");
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![Value::GpuTensor(ha), Value::GpuTensor(hb)],
            )
            .expect("cat");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_gpu_from_host_inputs() {
        test_support::with_test_provider(|provider| {
            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &proto.data,
                shape: &proto.shape,
            };
            let proto_handle = provider.upload(&proto_view).expect("upload proto");

            let a = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let b = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let result = cat_builtin(
                Value::Int(IntValue::I32(1)),
                vec![
                    Value::Tensor(a),
                    Value::Tensor(b),
                    Value::from("like"),
                    Value::GpuTensor(proto_handle),
                ],
            )
            .expect("cat with like");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cat_like_logical_mismatch_errors() {
        let proto = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let err = cat_builtin(
            Value::Int(IntValue::I32(1)),
            vec![
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
                Value::from("like"),
                Value::LogicalArray(proto),
            ],
        )
        .unwrap_err();
        assert!(err.message().contains("logical"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cat_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

        let cpu_result = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())],
        )
        .expect("cat cpu");
        let expected = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let view_a = HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload a");
        let hb = provider.upload(&view_b).expect("upload b");
        let gpu_value = cat_builtin(
            Value::Int(IntValue::I32(2)),
            vec![Value::GpuTensor(ha), Value::GpuTensor(hb)],
        )
        .expect("cat gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }
}
