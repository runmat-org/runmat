//! MATLAB-compatible `diag` builtin.

use crate::builtins::common::{
    spec::{
        BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
        ReductionNaN, ResidencyPolicy, ShapeRequirements,
    },
    tensor,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{HostTensorView, ProviderPrecision};
use runmat_builtins::{
    CharArray, ComplexTensor, LiteralValue, LogicalArray, NumericDType, ResolveContext, Tensor,
    Type, Value,
};
use runmat_macros::runtime_builtin;

const MESSAGE_ID_INVALID_INPUT: &str = "MATLAB:diag:InvalidInput";
const MESSAGE_ID_INVALID_OFFSET: &str = "MATLAB:diag:InvalidOffset";

fn diag_type(args: &[Type], context: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };

    let vector_len = vector_len_from_type(input);
    let vector_mode = literal_keyword_at(context, 1).as_deref() == Some("vector")
        || literal_keyword_at(context, 2).as_deref() == Some("vector");
    let size_override = literal_size_override(context);
    let offset = literal_offset(context).unwrap_or(0);

    let mut output_is_logical = matches!(input, Type::Logical { .. } | Type::Bool);
    for idx in 1..context.literal_args.len() {
        match literal_keyword_at(context, idx).as_deref() {
            Some("double") => output_is_logical = false,
            Some("logical") => output_is_logical = true,
            _ => {}
        }
    }

    let mk_type = |rows: Option<usize>, cols: Option<usize>| {
        if output_is_logical {
            Type::Logical {
                shape: Some(vec![rows, cols]),
            }
        } else {
            Type::Tensor {
                shape: Some(vec![rows, cols]),
            }
        }
    };

    if vector_mode {
        if let Some(len) = vector_len {
            return mk_type(Some(len), Some(1));
        }
        return if output_is_logical {
            Type::logical()
        } else {
            Type::tensor()
        };
    }

    if let Some((rows, cols)) = size_override {
        if vector_len.is_some() {
            return mk_type(Some(rows), Some(cols));
        }
    }

    if let Some(len) = vector_len {
        let shift = offset.unsigned_abs();
        if let Some(size) = len.checked_add(shift) {
            return mk_type(Some(size), Some(size));
        }
        return if output_is_logical {
            Type::logical()
        } else {
            Type::tensor()
        };
    }

    if output_is_logical {
        Type::logical()
    } else {
        Type::tensor()
    }
}

fn vector_len_from_type(input: &Type) -> Option<usize> {
    match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if shape.len() == 1 {
                return shape[0];
            }
            let rows = shape.first().copied().flatten();
            let cols = shape.get(1).copied().flatten();
            match (rows, cols) {
                (Some(1), Some(c)) => Some(c),
                (Some(r), Some(1)) => Some(r),
                _ => None,
            }
        }
        Type::Num | Type::Int | Type::Bool => Some(1),
        _ => None,
    }
}

fn literal_keyword_at(context: &ResolveContext, idx: usize) -> Option<String> {
    context.literal_string_at(idx)
}

fn literal_size_override(context: &ResolveContext) -> Option<(usize, usize)> {
    parse_literal_size(context.literal_vector_at(1).as_deref())
        .or_else(|| parse_literal_size(context.literal_vector_at(2).as_deref()))
}

fn parse_literal_size(values: Option<&[LiteralValue]>) -> Option<(usize, usize)> {
    let values = values?;
    let dims: Vec<usize> = values
        .iter()
        .map(|value| match value {
            LiteralValue::Number(num) => {
                if !num.is_finite() {
                    return None;
                }
                let rounded = num.round();
                if (rounded - num).abs() > 1e-9 || rounded < 0.0 {
                    return None;
                }
                Some(rounded as usize)
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    match dims.as_slice() {
        [m] => Some((*m, *m)),
        [m, n] => Some((*m, *n)),
        _ => None,
    }
}

fn literal_offset(context: &ResolveContext) -> Option<isize> {
    literal_offset_at(context, 1).or_else(|| literal_offset_at(context, 2))
}

fn literal_offset_at(context: &ResolveContext, idx: usize) -> Option<isize> {
    let literal = context.literal_args.get(idx)?;
    match literal {
        LiteralValue::Number(value) => {
            if !value.is_finite() {
                return None;
            }
            let rounded = value.round();
            if (rounded - value).abs() > f64::EPSILON {
                return None;
            }
            Some(rounded as isize)
        }
        LiteralValue::Bool(flag) => Some(if *flag { 1 } else { 0 }),
        _ => None,
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::diag")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "diag",
    op_kind: GpuOpKind::Custom("diag"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "diag executes on the host and gathers GPU inputs first.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::diag")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diag",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "diag is a host-only shape helper.",
};

fn diag_error(message_id: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(message_id)
        .with_builtin("diag")
        .build()
}

#[derive(Clone)]
enum OutputTemplate {
    Native,
    Logical,
    Double,
    Like(Value),
}

#[derive(Clone, Copy)]
enum ClassOverride {
    Logical,
    Double,
}

struct ParsedDiagArgs {
    offset: isize,
    size_override: Option<(usize, usize)>,
    vector_mode: bool,
    template: OutputTemplate,
}

impl ParsedDiagArgs {
    async fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        let mut offset: Option<isize> = None;
        let mut size_override: Option<(usize, usize)> = None;
        let mut vector_mode = false;
        let mut class_override: Option<ClassOverride> = None;
        let mut like_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = &args[idx];
            if let Some(keyword) = keyword_of(arg) {
                match keyword.as_str() {
                    "vector" => {
                        if vector_mode {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: duplicate 'vector' option",
                            ));
                        }
                        vector_mode = true;
                        idx += 1;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: cannot combine 'like' with 'logical'",
                            ));
                        }
                        class_override = Some(ClassOverride::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: cannot combine 'like' with 'double'",
                            ));
                        }
                        class_override = Some(ClassOverride::Double);
                        idx += 1;
                        continue;
                    }
                    "like" => {
                        if class_override.is_some() {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: cannot combine 'like' with class overrides",
                            ));
                        }
                        if like_proto.is_some() {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: duplicate 'like' option",
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(diag_error(
                                MESSAGE_ID_INVALID_INPUT,
                                "diag: expected prototype after 'like'",
                            ));
                        };
                        like_proto = Some(proto);
                        idx += 2;
                        continue;
                    }
                    other => {
                        return Err(diag_error(
                            MESSAGE_ID_INVALID_INPUT,
                            format!("diag: unrecognised option '{other}'"),
                        ));
                    }
                }
            }

            if offset.is_none() {
                if let Some(parsed_offset) = try_parse_offset(arg).await? {
                    offset = Some(parsed_offset);
                    idx += 1;
                    continue;
                }
            }

            if size_override.is_none() {
                if let Some(size) = try_parse_size_override(arg).await? {
                    size_override = Some(size);
                    idx += 1;
                    continue;
                }
            }

            return Err(diag_error(
                MESSAGE_ID_INVALID_INPUT,
                format!("diag: unrecognised argument {arg:?}"),
            ));
        }

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else {
            match class_override {
                Some(ClassOverride::Logical) => OutputTemplate::Logical,
                Some(ClassOverride::Double) => OutputTemplate::Double,
                None => OutputTemplate::Native,
            }
        };

        Ok(Self {
            offset: offset.unwrap_or(0),
            size_override,
            vector_mode,
            template,
        })
    }
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

async fn try_parse_offset(value: &Value) -> BuiltinResult<Option<isize>> {
    let gathered = gather_if_needed_async(value).await?;
    if !is_scalar_offset_candidate(&gathered) {
        return Ok(None);
    }
    scalar_to_isize(&gathered).map(Some)
}

fn is_scalar_offset_candidate(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) => true,
        Value::Tensor(t) => t.data.len() == 1,
        Value::LogicalArray(array) => array.data.len() == 1,
        _ => false,
    }
}

async fn try_parse_size_override(value: &Value) -> BuiltinResult<Option<(usize, usize)>> {
    let Some(dims) = tensor::dims_from_value_async(value)
        .await
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?
    else {
        return Ok(None);
    };

    match dims.as_slice() {
        [] => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: size vector must contain one or two elements",
        )),
        [m] => Ok(Some((*m, *m))),
        [m, n] => Ok(Some((*m, *n))),
        _ => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: size vector must contain one or two elements",
        )),
    }
}

#[runtime_builtin(
    name = "diag",
    category = "array/shape",
    summary = "Extract or create a diagonal from a vector or matrix.",
    keywords = "diag,diagonal,matrix",
    type_resolver(diag_type),
    builtin_path = "crate::builtins::array::shape::diag"
)]
async fn diag_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let parsed = ParsedDiagArgs::parse(rest).await?;
    let gathered = gather_if_needed_async(&value).await?;
    let input = coerce_diag_input(gathered)?;

    let raw = match input {
        DiagInput::Tensor(tensor) => evaluate_tensor(tensor, &parsed)?,
        DiagInput::Logical(array) => evaluate_logical(array, &parsed)?,
        DiagInput::Complex(tensor) => evaluate_complex(tensor, &parsed)?,
        DiagInput::Char(array) => evaluate_char(array, &parsed)?,
    };

    apply_output_template(raw, &parsed.template).await
}

enum DiagInput {
    Tensor(Tensor),
    Logical(LogicalArray),
    Complex(ComplexTensor),
    Char(CharArray),
}

fn coerce_diag_input(value: Value) -> BuiltinResult<DiagInput> {
    match value {
        Value::Tensor(tensor) => Ok(DiagInput::Tensor(tensor)),
        Value::LogicalArray(array) => Ok(DiagInput::Logical(array)),
        Value::ComplexTensor(tensor) => Ok(DiagInput::Complex(tensor)),
        Value::CharArray(array) => Ok(DiagInput::Char(array)),
        Value::Num(n) => Ok(DiagInput::Tensor(
            Tensor::new(vec![n], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?,
        )),
        Value::Int(i) => Ok(DiagInput::Tensor(
            Tensor::new(vec![i.to_f64()], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?,
        )),
        Value::Bool(flag) => Ok(DiagInput::Logical(
            LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?,
        )),
        Value::Complex(re, im) => Ok(DiagInput::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?,
        )),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("diag: unsupported input {other:?}"),
        )),
    }
}

fn evaluate_tensor(tensor: Tensor, args: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let (data, shape) = evaluate_column_major_diag(&tensor.data, &tensor.shape, args, 0.0)?;
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

fn evaluate_logical(array: LogicalArray, args: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let (data, shape) = evaluate_column_major_diag(&array.data, &array.shape, args, 0u8)?;
    LogicalArray::new(data, shape)
        .map(Value::LogicalArray)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

fn evaluate_complex(tensor: ComplexTensor, args: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let (data, shape) = evaluate_column_major_diag(&tensor.data, &tensor.shape, args, (0.0, 0.0))?;
    ComplexTensor::new(data, shape)
        .map(Value::ComplexTensor)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

fn evaluate_char(array: CharArray, args: &ParsedDiagArgs) -> BuiltinResult<Value> {
    let rows = array.rows;
    let cols = array.cols;
    let is_vector = rows == 1 || cols == 1;

    validate_vector_mode(is_vector, args)?;

    if args.vector_mode {
        let len = rows.max(cols);
        let data = vector_copy(&array.data, len);
        let out = CharArray::new(data, len, 1)
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?;
        return Ok(Value::CharArray(out));
    }

    if is_vector {
        let len = rows.max(cols);
        let (out_rows, out_cols) = vector_output_dims(len, args)?;
        let data = diag_matrix_from_vector_row_major(
            &array.data,
            len,
            args.offset,
            out_rows,
            out_cols,
            ' ',
        )?;
        let out = CharArray::new(data, out_rows, out_cols)
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?;
        return Ok(Value::CharArray(out));
    }

    if args.size_override.is_some() {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: size overrides require vector inputs",
        ));
    }

    let diag = diag_vector_from_matrix_row_major(&array.data, rows, cols, args.offset);
    let len = diag.len();
    let out = CharArray::new(diag, len, 1)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?;
    Ok(Value::CharArray(out))
}

fn evaluate_column_major_diag<T: Copy>(
    data: &[T],
    shape: &[usize],
    args: &ParsedDiagArgs,
    zero: T,
) -> BuiltinResult<(Vec<T>, Vec<usize>)> {
    let (rows, cols) = matrix_dims(shape)?;
    let is_vector = rows == 1 || cols == 1;

    validate_vector_mode(is_vector, args)?;

    if args.vector_mode {
        let len = rows.max(cols);
        return Ok((vector_copy(data, len), vec![len, 1]));
    }

    if is_vector {
        let len = rows.max(cols);
        let (out_rows, out_cols) = vector_output_dims(len, args)?;
        let out =
            diag_matrix_from_vector_col_major(data, len, args.offset, out_rows, out_cols, zero)?;
        return Ok((out, vec![out_rows, out_cols]));
    }

    if args.size_override.is_some() {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: size overrides require vector inputs",
        ));
    }

    let diag = diag_vector_from_matrix_col_major(data, rows, cols, args.offset);
    let len = diag.len();
    Ok((diag, vec![len, 1]))
}

fn validate_vector_mode(is_vector: bool, args: &ParsedDiagArgs) -> BuiltinResult<()> {
    if !args.vector_mode {
        return Ok(());
    }
    if !is_vector {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: 'vector' requires a vector input",
        ));
    }
    if args.offset != 0 || args.size_override.is_some() {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: 'vector' cannot be combined with offsets or size overrides",
        ));
    }
    Ok(())
}

fn vector_output_dims(len: usize, args: &ParsedDiagArgs) -> BuiltinResult<(usize, usize)> {
    if let Some((rows, cols)) = args.size_override {
        return Ok((rows, cols));
    }
    let shift = args.offset.unsigned_abs();
    let size = len.checked_add(shift).ok_or_else(|| {
        diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: result dimensions exceed supported limits",
        )
    })?;
    Ok((size, size))
}

fn vector_copy<T: Copy>(data: &[T], len: usize) -> Vec<T> {
    data.iter().copied().take(len).collect()
}

fn matrix_dims(shape: &[usize]) -> BuiltinResult<(usize, usize)> {
    if shape.len() > 2 && shape[2..].iter().any(|dim| *dim != 1) {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: only vectors and matrices are supported",
        ));
    }
    let rows = *shape.first().unwrap_or(&1);
    let cols = *shape.get(1).unwrap_or(&1);
    Ok((rows, cols))
}

fn allocate_out<T: Copy>(rows: usize, cols: usize, value: T) -> BuiltinResult<Vec<T>> {
    let count = rows.checked_mul(cols).ok_or_else(|| {
        diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: result dimensions exceed supported limits",
        )
    })?;
    Ok(vec![value; count])
}

fn diag_matrix_from_vector_col_major<T: Copy>(
    data: &[T],
    len: usize,
    offset: isize,
    rows: usize,
    cols: usize,
    zero: T,
) -> BuiltinResult<Vec<T>> {
    let mut out = allocate_out(rows, cols, zero)?;
    let shift = offset.unsigned_abs();
    let (start_row, start_col) = if offset >= 0 {
        (0usize, shift)
    } else {
        (shift, 0usize)
    };
    if start_row >= rows || start_col >= cols {
        return Ok(out);
    }

    let max_len = (rows - start_row)
        .min(cols - start_col)
        .min(len)
        .min(data.len());
    for idx in 0..max_len {
        let row = start_row + idx;
        let col = start_col + idx;
        out[row + col * rows] = data[idx];
    }
    Ok(out)
}

fn diag_matrix_from_vector_row_major<T: Copy>(
    data: &[T],
    len: usize,
    offset: isize,
    rows: usize,
    cols: usize,
    zero: T,
) -> BuiltinResult<Vec<T>> {
    let mut out = allocate_out(rows, cols, zero)?;
    let shift = offset.unsigned_abs();
    let (start_row, start_col) = if offset >= 0 {
        (0usize, shift)
    } else {
        (shift, 0usize)
    };
    if start_row >= rows || start_col >= cols {
        return Ok(out);
    }

    let max_len = (rows - start_row)
        .min(cols - start_col)
        .min(len)
        .min(data.len());
    for idx in 0..max_len {
        let row = start_row + idx;
        let col = start_col + idx;
        out[row * cols + col] = data[idx];
    }
    Ok(out)
}

fn diag_vector_from_matrix_col_major<T: Copy>(
    data: &[T],
    rows: usize,
    cols: usize,
    offset: isize,
) -> Vec<T> {
    let shift = offset.unsigned_abs();
    let (start_row, start_col) = if offset >= 0 {
        (0usize, shift)
    } else {
        (shift, 0usize)
    };
    if start_row >= rows || start_col >= cols {
        return Vec::new();
    }
    let max_len = (rows - start_row).min(cols - start_col);
    let mut out = Vec::with_capacity(max_len);
    for idx in 0..max_len {
        let row = start_row + idx;
        let col = start_col + idx;
        out.push(data[row + col * rows]);
    }
    out
}

fn diag_vector_from_matrix_row_major<T: Copy>(
    data: &[T],
    rows: usize,
    cols: usize,
    offset: isize,
) -> Vec<T> {
    let shift = offset.unsigned_abs();
    let (start_row, start_col) = if offset >= 0 {
        (0usize, shift)
    } else {
        (shift, 0usize)
    };
    if start_row >= rows || start_col >= cols {
        return Vec::new();
    }
    let max_len = (rows - start_row).min(cols - start_col);
    let mut out = Vec::with_capacity(max_len);
    for idx in 0..max_len {
        let row = start_row + idx;
        let col = start_col + idx;
        out.push(data[row * cols + col]);
    }
    out
}

fn scalar_to_isize(value: &Value) -> BuiltinResult<isize> {
    match value {
        Value::Int(i) => Ok(i.to_i64() as isize),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(diag_error(
                    MESSAGE_ID_INVALID_OFFSET,
                    "diag: diagonal offset must be finite",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(diag_error(
                    MESSAGE_ID_INVALID_OFFSET,
                    "diag: diagonal offset must be an integer",
                ));
            }
            Ok(rounded as isize)
        }
        Value::Tensor(t) if t.data.len() == 1 => scalar_to_isize(&Value::Num(t.data[0])),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(if array.data[0] != 0 { 1 } else { 0 })
        }
        Value::Bool(flag) => Ok(if *flag { 1 } else { 0 }),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_OFFSET,
            format!("diag: diagonal offset must be a numeric scalar, got {other:?}"),
        )),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Native => Ok(value),
        OutputTemplate::Logical => logical_array_from_value(value).map(Value::LogicalArray),
        OutputTemplate::Double => tensor_from_value(value).map(Value::Tensor),
        OutputTemplate::Like(proto) => apply_like_template(value, proto).await,
    }
}

async fn apply_like_template(value: Value, prototype: &Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = prototype {
        return apply_gpu_like_template(value, handle).await;
    }

    let gathered_proto = gather_if_needed_async(prototype).await?;
    match gathered_proto {
        Value::LogicalArray(_) | Value::Bool(_) => {
            logical_array_from_value(value).map(Value::LogicalArray)
        }
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            complex_tensor_from_value(value).map(Value::ComplexTensor)
        }
        Value::Tensor(proto_tensor) => {
            let tensor = cast_tensor_dtype(tensor_from_value(value)?, proto_tensor.dtype)?;
            Ok(Value::Tensor(tensor))
        }
        Value::Num(_)
        | Value::Int(_)
        | Value::CharArray(_)
        | Value::String(_)
        | Value::StringArray(_) => tensor_from_value(value).map(Value::Tensor),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!(
                "diag: unsupported 'like' prototype {other:?}; expected numeric, logical, complex, or gpuArray"
            ),
        )),
    }
}

async fn apply_gpu_like_template(
    value: Value,
    prototype: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(prototype)
        .or_else(runmat_accelerate_api::provider)
        .ok_or_else(|| {
            diag_error(
                MESSAGE_ID_INVALID_INPUT,
                "diag: no acceleration provider registered for 'like' gpuArray prototype",
            )
        })?;

    let logical_target = runmat_accelerate_api::handle_is_logical(prototype);
    if !logical_target && matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        return Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: complex outputs are not supported for gpuArray 'like' prototypes",
        ));
    }

    let precision =
        runmat_accelerate_api::handle_precision(prototype).unwrap_or(provider.precision());

    let host_tensor = if logical_target {
        let logical = logical_array_from_value(value)?;
        tensor::logical_to_tensor(&logical)
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?
    } else {
        let tensor = tensor_from_value(value)?;
        cast_tensor_dtype(tensor, provider_precision_to_dtype(precision))?
    };

    let view = HostTensorView {
        data: &host_tensor.data,
        shape: &host_tensor.shape,
    };
    let uploaded = provider
        .upload(&view)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))?;

    runmat_accelerate_api::set_handle_precision(&uploaded, precision);
    runmat_accelerate_api::set_handle_logical(&uploaded, logical_target);

    Ok(Value::GpuTensor(uploaded))
}

fn provider_precision_to_dtype(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

fn cast_tensor_dtype(tensor: Tensor, dtype: NumericDType) -> BuiltinResult<Tensor> {
    if tensor.dtype == dtype {
        return Ok(tensor);
    }

    let data = match dtype {
        NumericDType::F32 => tensor
            .data
            .iter()
            .map(|value| (*value as f32) as f64)
            .collect(),
        NumericDType::F64 => tensor.data.clone(),
    };

    Tensor::new_with_dtype(data, tensor.shape.clone(), dtype)
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

fn logical_array_from_value(value: Value) -> BuiltinResult<LogicalArray> {
    match value {
        Value::LogicalArray(array) => Ok(array),
        Value::Tensor(tensor) => {
            let data: Vec<u8> = tensor
                .data
                .iter()
                .map(|value| if *value != 0.0 { 1 } else { 0 })
                .collect();
            LogicalArray::new(data, tensor.shape)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::ComplexTensor(tensor) => {
            let data: Vec<u8> = tensor
                .data
                .iter()
                .map(|(re, im)| if *re != 0.0 || *im != 0.0 { 1 } else { 0 })
                .collect();
            LogicalArray::new(data, tensor.shape)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::CharArray(chars) => {
            let data: Vec<u8> = chars
                .data
                .iter()
                .map(|ch| if (*ch as u32) != 0 { 1 } else { 0 })
                .collect();
            LogicalArray::new(data, vec![chars.rows, chars.cols])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::Num(n) => LogicalArray::new(vec![if n != 0.0 { 1 } else { 0 }], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Int(i) => LogicalArray::new(vec![if i.to_i64() != 0 { 1 } else { 0 }], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Bool(flag) => LogicalArray::new(vec![if flag { 1 } else { 0 }], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Complex(re, im) => {
            let logical = if re != 0.0 || im != 0.0 { 1 } else { 0 };
            LogicalArray::new(vec![logical], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("diag: cannot convert {other:?} to logical output"),
        )),
    }
}

fn tensor_from_value(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) => Ok(tensor),
        Value::LogicalArray(array) => tensor::logical_to_tensor(&array)
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Int(i) => Tensor::new(vec![i.to_f64()], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Bool(flag) => Tensor::new(vec![if flag { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::CharArray(chars) => char_array_to_tensor(&chars),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            "diag: cannot convert complex output to 'double'",
        )),
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("diag: cannot convert {other:?} to double output"),
        )),
    }
}

fn complex_tensor_from_value(value: Value) -> BuiltinResult<ComplexTensor> {
    match value {
        Value::ComplexTensor(tensor) => Ok(tensor),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Tensor(tensor) => {
            let data: Vec<(f64, f64)> = tensor.data.into_iter().map(|re| (re, 0.0)).collect();
            ComplexTensor::new(data, tensor.shape)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::LogicalArray(array) => {
            let data: Vec<(f64, f64)> = array
                .data
                .iter()
                .map(|value| if *value != 0 { (1.0, 0.0) } else { (0.0, 0.0) })
                .collect();
            ComplexTensor::new(data, array.shape)
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::CharArray(chars) => {
            let data: Vec<(f64, f64)> = chars
                .data
                .iter()
                .map(|ch| (*ch as u32 as f64, 0.0))
                .collect();
            ComplexTensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        Value::Num(n) => ComplexTensor::new(vec![(n, 0.0)], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Int(i) => ComplexTensor::new(vec![(i.to_f64(), 0.0)], vec![1, 1])
            .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}"))),
        Value::Bool(flag) => {
            let re = if flag { 1.0 } else { 0.0 };
            ComplexTensor::new(vec![(re, 0.0)], vec![1, 1])
                .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
        }
        other => Err(diag_error(
            MESSAGE_ID_INVALID_INPUT,
            format!("diag: cannot convert {other:?} to complex output"),
        )),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|ch| *ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|err| diag_error(MESSAGE_ID_INVALID_INPUT, format!("diag: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn run_diag(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(diag_builtin(value, rest))
    }

    fn size_vector(rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(vec![rows as f64, cols as f64], vec![1, 2]).unwrap())
    }

    #[test]
    fn diag_type_vector_to_square() {
        let out = diag_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(4)])
            }
        );
    }

    #[test]
    fn diag_type_matrix_falls_back_tensor() {
        let out = diag_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::tensor());
    }

    #[test]
    fn diag_vector_mode_returns_column_vector() {
        let value = Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap());
        let out = run_diag(value, vec![Value::from("vector")]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape, vec![3, 1]);
        assert_eq!(tensor.data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn diag_vector_size_override_rectangular() {
        let value = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let out = run_diag(value, vec![size_vector(2, 4)]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.data, vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn diag_vector_offset_and_size_override() {
        let value = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let out = run_diag(value, vec![Value::Num(1.0), size_vector(3, 4)]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape, vec![3, 4]);
        assert_eq!(
            tensor.data,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn diag_extracts_subdiagonal() {
        let matrix = Tensor::new(
            vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let out = run_diag(Value::Tensor(matrix), vec![Value::Num(-1.0)]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(tensor.data, vec![4.0, 8.0]);
    }

    #[test]
    fn diag_char_rectangular_and_extract() {
        let chars = Value::CharArray(CharArray::new_row("ab"));
        let out = run_diag(chars, vec![size_vector(2, 4)]).expect("diag");
        let Value::CharArray(matrix) = out else {
            panic!("expected char output");
        };
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 4);
        assert_eq!(matrix.data, vec!['a', ' ', ' ', ' ', ' ', 'b', ' ', ' ']);

        let extracted = run_diag(Value::CharArray(matrix), Vec::new()).expect("diag extract");
        let Value::CharArray(vector) = extracted else {
            panic!("expected char output");
        };
        assert_eq!(vector.rows, 2);
        assert_eq!(vector.cols, 1);
        assert_eq!(vector.data, vec!['a', 'b']);
    }

    #[test]
    fn diag_supports_trailing_singleton_dims() {
        let matrix = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2, 1, 1]).unwrap();
        let out = run_diag(Value::Tensor(matrix), Vec::new()).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape, vec![2, 1]);
        assert_eq!(tensor.data, vec![1.0, 2.0]);
    }

    #[test]
    fn diag_rejects_non_singleton_trailing_dims() {
        let matrix = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = run_diag(Value::Tensor(matrix), Vec::new()).expect_err("expected error");
        assert!(
            err.message()
                .contains("only vectors and matrices are supported"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn diag_logical_output_override() {
        let value = Value::Tensor(Tensor::new(vec![1.0, 0.0, 3.0], vec![1, 3]).unwrap());
        let out = run_diag(value, vec![Value::from("logical")]).expect("diag");
        let Value::LogicalArray(array) = out else {
            panic!("expected logical output");
        };
        assert_eq!(array.shape, vec![3, 3]);
        assert_eq!(array.data, vec![1, 0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn diag_double_override_from_logical_input() {
        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let out =
            run_diag(Value::LogicalArray(logical), vec![Value::from("double")]).expect("diag");
        let Value::Tensor(tensor) = out else {
            panic!("expected tensor output");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn diag_like_logical_output() {
        let value = Value::Tensor(Tensor::new(vec![2.0, 0.0], vec![1, 2]).unwrap());
        let out = run_diag(value, vec![Value::from("like"), Value::Bool(true)]).expect("diag");
        let Value::LogicalArray(array) = out else {
            panic!("expected logical output");
        };
        assert_eq!(array.shape, vec![2, 2]);
        assert_eq!(array.data, vec![1, 0, 0, 0]);
    }

    #[test]
    fn diag_like_complex_output() {
        let value = Value::Tensor(Tensor::new(vec![2.0, 0.0], vec![1, 2]).unwrap());
        let out =
            run_diag(value, vec![Value::from("like"), Value::Complex(1.0, 2.0)]).expect("diag");
        let Value::ComplexTensor(tensor) = out else {
            panic!("expected complex tensor output");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(
            tensor.data,
            vec![(2.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        );
    }

    #[test]
    fn diag_like_gpu_uploads_result() {
        test_support::with_test_provider(|provider| {
            let proto_tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &proto_tensor.data,
                shape: &proto_tensor.shape,
            };
            let proto = provider.upload(&view).expect("upload");

            let value = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
            let out =
                run_diag(value, vec![Value::from("like"), Value::GpuTensor(proto)]).expect("diag");
            let Value::GpuTensor(handle) = out else {
                panic!("expected gpu tensor output");
            };
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 0.0, 2.0]);
        });
    }

    #[test]
    fn diag_like_gpu_logical_preserves_logical_flag() {
        test_support::with_test_provider(|provider| {
            let proto_tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &proto_tensor.data,
                shape: &proto_tensor.shape,
            };
            let proto = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_logical(&proto, true);

            let value = Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap());
            let out =
                run_diag(value, vec![Value::from("like"), Value::GpuTensor(proto)]).expect("diag");
            let Value::GpuTensor(handle) = out else {
                panic!("expected gpu tensor output");
            };
            assert!(runmat_accelerate_api::handle_is_logical(&handle));
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 0.0, 0.0]);
        });
    }

    #[test]
    fn diag_vector_mode_rejects_matrix_input() {
        let matrix = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        let err = run_diag(matrix, vec![Value::from("vector")]).expect_err("expected error");
        assert!(
            err.message().contains("'vector' requires a vector input"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn diag_vector_mode_rejects_offset_combo() {
        let vector = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let err = run_diag(vector, vec![Value::Num(1.0), Value::from("vector")])
            .expect_err("expected error");
        assert!(
            err.message()
                .contains("'vector' cannot be combined with offsets or size overrides"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn diag_reports_invalid_offset_type() {
        let vector = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let err = run_diag(vector, vec![Value::CharArray(CharArray::new_row("oops"))])
            .expect_err("expected error");
        assert!(
            err.message().contains("unrecognised option")
                || err
                    .message()
                    .contains("diagonal offset must be a numeric scalar"),
            "unexpected error: {}",
            err.message()
        );
    }
}
