//! MATLAB-compatible `disp` builtin with GPU-aware formatting semantics.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CellArray, CharArray, ComplexTensor, IntValue, IntegerStorage,
    LogicalArray, StringArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::common::char_row_to_string;
use crate::console::{record_console_line, ConsoleStream};
use crate::gather_if_needed_async;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::disp")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "disp",
    op_kind: GpuOpKind::Custom("sink"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Always formats on the CPU; GPU tensors are gathered via the active provider before display.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::disp")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "disp",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Side-effecting sink; excluded from fusion planning.",
};

/// Minimum column width (in characters) for numeric and logical displays.
const NUMERIC_MIN_COLUMN_WIDTH: usize = 6;
/// Number of leading spaces for cell array rows.
const CELL_ROW_INDENT: usize = 4;
/// Number of spaces used for struct field indentation.
const STRUCT_FIELD_INDENT: usize = 4;
/// Continuation indent for multi-line struct field values.
const STRUCT_CONTINUATION_INDENT: usize = 8;

#[derive(Clone, Copy)]
enum RenderMode {
    TopLevel,
    Nested,
}

#[derive(Clone, Copy)]
enum Align {
    Left,
    Right,
}

const DISP_INPUTS_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to display in the Command Window.",
}];
const DISP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "disp(X)",
    inputs: &DISP_INPUTS_VALUE,
    outputs: &[],
}];
const DISP_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISP.ARG_CONFIG",
    identifier: None,
    when: "Too many input arguments are passed to disp.",
    message: "disp: too many input arguments",
};
const DISP_ERROR_GATHER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISP.GATHER",
    identifier: None,
    when: "Input value cannot be gathered onto the host for rendering.",
    message: "disp: failed to gather value for display",
};
const DISP_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:disp:TooManyOutputs"),
    when: "An output is requested from disp.",
    message: "disp does not return output arguments",
};
const DISP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DISP_ERROR_ARG_CONFIG,
    DISP_ERROR_GATHER,
    DISP_ERROR_TOO_MANY_OUTPUTS,
];
pub const DISP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DISP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DISP_ERRORS,
};

const DISP_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is formatted from exact authoritative storage, including 64-bit extrema and integer members nested in supported containers.",
    }];

pub const DISP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "disp(X)",
        inputs: &DISP_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "disp is a no-output host rendering sink. Resident integer values gather through exact typed provider download before decimal formatting; no floating compatibility mirror or provider output is created.",
    }];

fn disp_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    disp_error_with(error, error.message)
}

fn disp_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin("disp");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "disp",
    category = "io",
    summary = "Display values without returning output.",
    keywords = "disp,display,print,gpu",
    sink = true,
    accel = "sink",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::disp_type),
    descriptor(crate::builtins::io::disp::DISP_DESCRIPTOR),
    integer_capabilities(crate::builtins::io::disp::DISP_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::io::disp"
)]
async fn disp_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 0) {
        return Err(disp_error(&DISP_ERROR_TOO_MANY_OUTPUTS));
    }
    if !rest.is_empty() {
        return Err(disp_error(&DISP_ERROR_ARG_CONFIG));
    }

    let host_value = gather_if_needed_async(&value)
        .await
        .map_err(|e| disp_error_with(&DISP_ERROR_GATHER, format!("disp: {e}")))?;
    let lines = format_for_disp(&host_value);

    if !lines.is_empty() {
        let body = lines.join("\n");
        record_console_line(ConsoleStream::Stdout, body);
    }

    Ok(empty_return_value())
}

pub(crate) fn format_for_disp(value: &Value) -> Vec<String> {
    render_value(value, RenderMode::TopLevel)
}

pub(crate) fn format_for_display(value: &Value) -> Vec<String> {
    let rendered = format_for_disp(value);
    if !rendered.is_empty() {
        return rendered;
    }
    match value {
        Value::Tensor(_) | Value::ComplexTensor(_) | Value::LogicalArray(_) => {
            vec!["[]".to_string()]
        }
        Value::CharArray(array) => vec![format!("{}x{} empty char array", array.rows, array.cols)],
        Value::StringArray(array) => vec![format!(
            "{} empty string array",
            dims_to_string(&canonical_dims(&array.shape))
        )],
        Value::Cell(cell) => vec![format!(
            "{} empty cell array",
            dims_to_string(&canonical_dims(&cell.shape))
        )],
        _ => rendered,
    }
}

fn render_value(value: &Value, mode: RenderMode) -> Vec<String> {
    match value {
        Value::Object(_) if crate::builtins::table::is_table_value(value) => match mode {
            RenderMode::TopLevel => crate::builtins::table::table_display_text(value)
                .unwrap_or_else(|_| value.to_string())
                .lines()
                .map(|line| line.to_string())
                .collect(),
            RenderMode::Nested => vec![crate::builtins::table::table_summary_text(value)
                .unwrap_or_else(|_| value.to_string())],
        },
        Value::Object(obj) if obj.is_class("datetime") => match mode {
            RenderMode::TopLevel => crate::builtins::datetime::datetime_display_text(value)
                .map(|text| text.unwrap_or_else(|| value.to_string()))
                .unwrap_or_else(|_| value.to_string())
                .lines()
                .map(|line| line.to_string())
                .collect(),
            RenderMode::Nested => vec![crate::builtins::datetime::datetime_summary(value)
                .ok()
                .flatten()
                .unwrap_or_else(|| value.to_string())],
        },
        Value::Object(obj) if obj.is_class("duration") => match mode {
            RenderMode::TopLevel => crate::builtins::duration::duration_display_text(value)
                .map(|text| text.unwrap_or_else(|| value.to_string()))
                .unwrap_or_else(|_| value.to_string())
                .lines()
                .map(|line| line.to_string())
                .collect(),
            RenderMode::Nested => vec![crate::builtins::duration::duration_summary(value)
                .ok()
                .flatten()
                .unwrap_or_else(|| value.to_string())],
        },
        Value::String(text) => match mode {
            RenderMode::TopLevel => split_lines(text),
            RenderMode::Nested => vec![quote_double(text)],
        },
        Value::Symbolic(expr) => vec![expr.to_string()],
        Value::CharArray(array) => format_char_array(array, mode),
        Value::StringArray(array) => format_string_array(array, mode),
        Value::Num(n) => vec![format_scalar_number(*n)],
        Value::Int(i) => vec![format_int(i)],
        Value::Bool(flag) => vec![if *flag { "1".into() } else { "0".into() }],
        Value::Tensor(tensor) => match mode {
            RenderMode::TopLevel => format_numeric_tensor(tensor),
            RenderMode::Nested => format_numeric_tensor_nested(tensor),
        },
        Value::SparseTensor(sparse) => match mode {
            RenderMode::TopLevel => sparse
                .to_string()
                .lines()
                .map(|line| line.to_string())
                .collect(),
            RenderMode::Nested => {
                let nnz = sparse.nnz();
                vec![format!(
                    "<sparse {}x{} {} nnz={}>",
                    sparse.rows,
                    sparse.cols,
                    sparse.class_name(),
                    nnz
                )]
            }
        },
        Value::Complex(re, im) => vec![format!("{}", Value::Complex(*re, *im))],
        Value::ComplexTensor(tensor) => match mode {
            RenderMode::TopLevel => format_complex_tensor(tensor),
            RenderMode::Nested => format_complex_tensor_nested(tensor),
        },
        Value::LogicalArray(logical) => match mode {
            RenderMode::TopLevel => format_logical_array(logical),
            RenderMode::Nested => format_logical_array_nested(logical),
        },
        Value::Struct(struct_value) => match mode {
            RenderMode::TopLevel => format_struct(struct_value),
            RenderMode::Nested => vec!["[1x1 struct]".to_string()],
        },
        Value::Cell(cell) => match mode {
            RenderMode::TopLevel => format_cell(cell),
            RenderMode::Nested => vec![format!(
                "{} cell",
                dims_to_string(&canonical_dims(&cell.shape))
            )],
        },
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => {
            vec![value.to_string()]
        }
        Value::GpuTensor(_) => vec!["gpuArray".to_string()],
    }
}

fn format_numeric_tensor(tensor: &Tensor) -> Vec<String> {
    let shape = canonical_dims(&tensor.shape);
    let len = tensor::tensor_element_len(tensor);
    if len == 0 {
        return Vec::new();
    }
    if shape.contains(&0) {
        if shape.len() == 2 && shape[0] == 0 && shape[1] == 0 {
            return vec!["[]".to_string()];
        }
        return vec![format!("Empty matrix: {}", dims_to_by_string(&shape))];
    }
    if shape.len() <= 2 {
        let rows = shape[0];
        let cols = shape.get(1).copied().unwrap_or(1);
        if len == 1 {
            return vec![format_tensor_value(tensor, 0)];
        }
        return format_table(
            rows,
            cols,
            0,
            NUMERIC_MIN_COLUMN_WIDTH,
            Align::Right,
            |r, c| {
                let idx = r + c * rows;
                format_tensor_value(tensor, idx)
            },
        );
    }
    format_numeric_tensor_pages(tensor, &shape)
}

fn format_numeric_tensor_pages(tensor: &Tensor, dims: &[usize]) -> Vec<String> {
    debug_assert!(dims.len() > 2);
    let rows = dims[0];
    let cols = dims[1];
    if rows == 0 || cols == 0 {
        return vec![format!("Empty matrix: {}", dims_to_by_string(dims))];
    }
    let tail_dims = &dims[2..];
    let mut tail_indices = vec![0usize; tail_dims.len()];
    let mut lines = Vec::new();

    loop {
        lines.push(page_header(&tail_indices));
        let current_tail = tail_indices.clone();
        let table = format_table(
            rows,
            cols,
            0,
            NUMERIC_MIN_COLUMN_WIDTH,
            Align::Right,
            |r, c| {
                let idx = linear_index_with_tail(dims, r, c, &current_tail);
                format_tensor_value(tensor, idx)
            },
        );
        lines.extend(table);
        if !increment_multi_index(&mut tail_indices, tail_dims) {
            break;
        }
        lines.push(String::new());
    }

    lines
}

fn format_numeric_tensor_nested(tensor: &Tensor) -> Vec<String> {
    let len = tensor::tensor_element_len(tensor);
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![format_tensor_value(tensor, 0)];
    }
    let shape = canonical_dims(&tensor.shape);
    let class_name = tensor.numeric_dtype().class_name();
    vec![format!("[{} {class_name}]", dims_to_string(&shape))]
}

fn format_tensor_value(tensor: &Tensor, index: usize) -> String {
    let value = tensor
        .numeric_value_at(index)
        .expect("index within authoritative numeric storage");
    if let Some(value) = value.into_int_value() {
        value.decimal_string()
    } else {
        format_scalar_number(value.materialize_f64())
    }
}

fn format_complex_tensor(tensor: &ComplexTensor) -> Vec<String> {
    let shape = canonical_dims(&tensor.shape);
    let len = tensor::complex_tensor_element_len(tensor);
    if len == 0 {
        return Vec::new();
    }
    if shape.contains(&0) {
        if shape.len() == 2 && shape[0] == 0 && shape[1] == 0 {
            return vec!["[]".to_string()];
        }
        return vec![format!("Empty matrix: {}", dims_to_by_string(&shape))];
    }
    if shape.len() <= 2 {
        let rows = shape[0];
        let cols = shape.get(1).copied().unwrap_or(1);
        if len == 1 {
            return vec![tensor.format_element(0)];
        }
        return format_table(
            rows,
            cols,
            0,
            NUMERIC_MIN_COLUMN_WIDTH,
            Align::Right,
            |r, c| {
                let idx = r + c * rows;
                tensor.format_element(idx)
            },
        );
    }
    format_complex_tensor_pages(tensor, &shape)
}

fn format_complex_tensor_pages(tensor: &ComplexTensor, dims: &[usize]) -> Vec<String> {
    debug_assert!(dims.len() > 2);
    let rows = dims[0];
    let cols = dims[1];
    if rows == 0 || cols == 0 {
        return vec![format!("Empty matrix: {}", dims_to_by_string(dims))];
    }
    let tail_dims = &dims[2..];
    let mut tail_indices = vec![0usize; tail_dims.len()];
    let mut lines = Vec::new();

    loop {
        lines.push(page_header(&tail_indices));
        let current_tail = tail_indices.clone();
        let table = format_table(
            rows,
            cols,
            0,
            NUMERIC_MIN_COLUMN_WIDTH,
            Align::Right,
            |r, c| {
                let idx = linear_index_with_tail(dims, r, c, &current_tail);
                tensor.format_element(idx)
            },
        );
        lines.extend(table);
        if !increment_multi_index(&mut tail_indices, tail_dims) {
            break;
        }
        lines.push(String::new());
    }

    lines
}

fn format_complex_tensor_nested(tensor: &ComplexTensor) -> Vec<String> {
    let len = tensor::complex_tensor_element_len(tensor);
    if len == 0 {
        return vec!["[]".to_string()];
    }
    if len == 1 {
        return vec![tensor.format_element(0)];
    }
    let shape = canonical_dims(&tensor.shape);
    let class_name = tensor
        .integer_storage()
        .as_ref()
        .map(|storage| storage.class_name())
        .unwrap_or("double");
    vec![format!(
        "[{} complex {}]",
        dims_to_string(&shape),
        class_name
    )]
}

fn format_logical_array(logical: &LogicalArray) -> Vec<String> {
    if logical.data.is_empty() {
        return Vec::new();
    }
    match tensor::logical_to_tensor(logical) {
        Ok(tensor) => format_numeric_tensor(&tensor),
        Err(_) => split_lines(&logical.to_string()),
    }
}

fn format_logical_array_nested(logical: &LogicalArray) -> Vec<String> {
    if logical.data.is_empty() {
        return vec!["[]".to_string()];
    }
    if logical.data.len() == 1 {
        return vec![if logical.data[0] != 0 {
            "1".to_string()
        } else {
            "0".to_string()
        }];
    }
    let shape = canonical_dims(&logical.shape);
    vec![format!("[{} logical]", dims_to_string(&shape))]
}

fn format_char_array(array: &CharArray, mode: RenderMode) -> Vec<String> {
    if array.rows == 0 || array.cols == 0 {
        return match mode {
            RenderMode::TopLevel => Vec::new(),
            RenderMode::Nested => vec!["''".to_string()],
        };
    }
    let mut lines: Vec<String> = (0..array.rows)
        .map(|row| char_row_to_string(array, row))
        .collect();
    if matches!(mode, RenderMode::Nested) {
        lines = lines
            .into_iter()
            .map(|line| format!("'{}'", line.replace('\'', "''")))
            .collect();
    }
    lines
}

fn format_string_array(array: &StringArray, mode: RenderMode) -> Vec<String> {
    let shape = canonical_dims(&array.shape);
    let total = array.data.len();
    if matches!(mode, RenderMode::Nested) {
        if total == 0 {
            return vec![format!("{} string array", dims_to_string(&shape))];
        }
        if total == 1 {
            return vec![quote_double(&array.data[0])];
        }
        return vec![format!("[{} string]", dims_to_string(&shape))];
    }

    let rows = shape[0];
    let cols = shape.get(1).copied().unwrap_or(1);
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    if shape.len() > 2 {
        return vec![format!("{} string array", dims_to_string(&shape))];
    }
    format_table(rows, cols, 0, 1, Align::Left, |r, c| {
        let idx = r + c * rows;
        quote_double(&array.data[idx])
    })
}

fn format_struct(struct_value: &StructValue) -> Vec<String> {
    if struct_value.fields.is_empty() {
        return vec!["struct with no fields.".to_string()];
    }
    let mut lines = Vec::new();
    for (name, field_value) in &struct_value.fields {
        let rendered = render_value(field_value, RenderMode::Nested);
        if let Some((first, rest)) = rendered.split_first() {
            lines.push(format!(
                "{:indent$}{}: {}",
                "",
                name,
                first,
                indent = STRUCT_FIELD_INDENT
            ));
            for continuation in rest {
                lines.push(format!(
                    "{:indent$}{}",
                    "",
                    continuation,
                    indent = STRUCT_CONTINUATION_INDENT
                ));
            }
        } else {
            lines.push(format!(
                "{:indent$}{}: []",
                "",
                name,
                indent = STRUCT_FIELD_INDENT
            ));
        }
    }
    lines
}

fn format_cell(cell: &CellArray) -> Vec<String> {
    let rows = cell.rows;
    let cols = cell.cols;
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    if cell.shape.len() > 2 {
        return vec![format!(
            "{} cell",
            dims_to_string(&canonical_dims(&cell.shape))
        )];
    }
    format_table(rows, cols, CELL_ROW_INDENT, 1, Align::Left, |r, c| {
        let idx = r * cols + c;
        let handle = cell.data[idx].clone();
        let value = handle;
        summarize_for_cell(&value)
    })
}

fn summarize_for_cell(value: &Value) -> String {
    match value {
        Value::Num(n) => format!("[{}]", format_scalar_number(*n)),
        Value::Int(i) => format!("[{}]", format_int(i)),
        Value::Bool(flag) => format!("[{}]", if *flag { 1 } else { 0 }),
        Value::Complex(re, im) => format!("[{}]", Value::Complex(*re, *im)),
        Value::Tensor(tensor) => {
            let len = tensor::tensor_element_len(tensor);
            if len == 0 {
                "[]".to_string()
            } else if len == 1 {
                format!("[{}]", format_tensor_value(tensor, 0))
            } else {
                let class_name = tensor
                    .integer_storage()
                    .map(IntegerStorage::class_name)
                    .unwrap_or("double");
                format!(
                    "[{} {class_name}]",
                    dims_to_string(&canonical_dims(&tensor.shape)),
                )
            }
        }
        Value::SparseTensor(sparse) => format!(
            "[{} sparse {}]",
            dims_to_string(&[sparse.rows, sparse.cols]),
            sparse.class_name()
        ),
        Value::ComplexTensor(tensor) => {
            let len = tensor::complex_tensor_element_len(tensor);
            if len == 0 {
                "[]".to_string()
            } else if len == 1 {
                format!("[{}]", tensor.format_element(0))
            } else {
                let class_name = tensor
                    .integer_storage()
                    .as_ref()
                    .map(|storage| storage.class_name())
                    .unwrap_or("double");
                format!(
                    "[{} complex {}]",
                    dims_to_string(&canonical_dims(&tensor.shape)),
                    class_name
                )
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                "[]".to_string()
            } else if logical.data.len() == 1 {
                format!("[{}]", if logical.data[0] != 0 { 1 } else { 0 })
            } else {
                format!(
                    "[{} logical]",
                    dims_to_string(&canonical_dims(&logical.shape))
                )
            }
        }
        Value::String(text) => quote_double(text),
        Value::CharArray(array) => {
            if array.rows == 0 || array.cols == 0 {
                "''".to_string()
            } else if array.rows == 1 {
                format!("'{}'", char_row_to_string(array, 0).replace('\'', "''"))
            } else {
                format!("[{} char]", dims_to_string(&[array.rows, array.cols]))
            }
        }
        Value::StringArray(array) => {
            if array.data.is_empty() {
                format!("{} string", dims_to_string(&canonical_dims(&array.shape)))
            } else if array.data.len() == 1 {
                quote_double(&array.data[0])
            } else {
                format!("[{} string]", dims_to_string(&canonical_dims(&array.shape)))
            }
        }
        Value::Symbolic(expr) => expr.to_string(),
        Value::Struct(_) => "[1x1 struct]".to_string(),
        Value::Cell(inner) => format!("[{} cell]", dims_to_string(&canonical_dims(&inner.shape))),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => value.to_string(),
        Value::GpuTensor(_) => "gpuArray".to_string(),
    }
}

fn format_table<F>(
    rows: usize,
    cols: usize,
    indent: usize,
    min_width: usize,
    align: Align,
    mut value_at: F,
) -> Vec<String>
where
    F: FnMut(usize, usize) -> String,
{
    let mut grid = vec![vec![String::new(); cols]; rows];
    let mut widths = vec![0usize; cols];

    for (c, column_width) in widths.iter_mut().enumerate().take(cols) {
        for (r, row) in grid.iter_mut().enumerate().take(rows) {
            let cell = value_at(r, c);
            let width = cell.chars().count();
            if width > *column_width {
                *column_width = width;
            }
            row[c] = cell;
        }
        if *column_width < min_width {
            *column_width = min_width;
        }
    }

    let mut lines = Vec::with_capacity(rows);
    for row in grid.iter().take(rows) {
        let mut line = String::new();
        if indent > 0 {
            line.extend(std::iter::repeat_n(' ', indent));
        }
        for (c, cell) in row.iter().enumerate().take(cols) {
            if c > 0 {
                line.push_str("  ");
            }
            let width = widths.get(c).copied().unwrap_or(min_width);
            match align {
                Align::Left => line.push_str(&format!("{:<width$}", cell, width = width)),
                Align::Right => line.push_str(&format!("{:>width$}", cell, width = width)),
            }
        }
        let trimmed = line.trim_end().to_string();
        lines.push(trimmed);
    }
    lines
}

fn format_int(value: &IntValue) -> String {
    match value {
        IntValue::I8(v) => v.to_string(),
        IntValue::I16(v) => v.to_string(),
        IntValue::I32(v) => v.to_string(),
        IntValue::I64(v) => v.to_string(),
        IntValue::U8(v) => v.to_string(),
        IntValue::U16(v) => v.to_string(),
        IntValue::U32(v) => v.to_string(),
        IntValue::U64(v) => v.to_string(),
    }
}

fn format_scalar_number(value: f64) -> String {
    format!("{}", Value::Num(value))
}

fn split_lines(text: &str) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    text.lines().map(|line| line.to_string()).collect()
}

fn quote_double(text: &str) -> String {
    let escaped = text.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

fn page_header(tail_indices: &[usize]) -> String {
    if tail_indices.len() == 1 {
        format!("(:,:,{}) =", tail_indices[0] + 1)
    } else {
        let joined = tail_indices
            .iter()
            .map(|idx| (idx + 1).to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!("(:,:,[{}]) =", joined)
    }
}

fn increment_multi_index(indices: &mut [usize], dims: &[usize]) -> bool {
    for (idx, dim) in indices.iter_mut().zip(dims.iter()) {
        *idx += 1;
        if *idx < *dim {
            return true;
        }
        *idx = 0;
    }
    false
}

fn linear_index_with_tail(dims: &[usize], row: usize, col: usize, tail: &[usize]) -> usize {
    debug_assert!(dims.len() >= 2);
    debug_assert_eq!(dims.len() - 2, tail.len());

    let mut index = row;
    let mut stride = dims[0];

    if dims.len() > 1 {
        index += col * stride;
        stride *= dims[1];
    }

    for (tail_idx, dim_size) in tail.iter().zip(dims.iter().skip(2)) {
        index += tail_idx * stride;
        stride *= *dim_size;
    }

    index
}

fn canonical_dims(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

fn dims_to_string(dims: &[usize]) -> String {
    if dims.is_empty() {
        return "0x0".to_string();
    }
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

fn dims_to_by_string(dims: &[usize]) -> String {
    if dims.is_empty() {
        return "0-by-0".to_string();
    }
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("-by-")
}

pub(crate) fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::make_cell;
    use runmat_builtins::{ComplexTensor, IntValue, StringArray, SymbolicExpr, Tensor};

    #[test]
    fn disp_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = DISP_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"disp(X)"));
        assert!(DISP_DESCRIPTOR.signatures[0].outputs.is_empty());
    }

    #[test]
    fn disp_rejects_outputs_and_prints_nothing_for_empty_builtin_values() {
        let _outputs = crate::output_count::push_output_count(Some(1));
        let err =
            futures::executor::block_on(disp_builtin(Value::Num(1.0), Vec::new())).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:disp:TooManyOutputs"));
        drop(_outputs);
        crate::console::reset_thread_buffer();
        futures::executor::block_on(disp_builtin(
            Value::Tensor(Tensor::zeros(vec![0, 0])),
            Vec::new(),
        ))
        .unwrap();
        assert!(crate::console::take_thread_buffer().is_empty());
        for value in [
            Value::ComplexTensor(ComplexTensor::new(Vec::new(), vec![0, 0]).unwrap()),
            Value::StringArray(StringArray::new(Vec::new(), vec![0, 1, 2]).unwrap()),
        ] {
            crate::console::reset_thread_buffer();
            futures::executor::block_on(disp_builtin(value, Vec::new())).unwrap();
            assert!(crate::console::take_thread_buffer().is_empty());
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_scalar_without_quotes() {
        let lines = format_for_disp(&Value::String("Simulation complete.".into()));
        assert_eq!(lines, vec!["Simulation complete.".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn symbolic_scalar_displays_expression() {
        let lines = format_for_disp(&Value::Symbolic(SymbolicExpr::variable("x")));

        assert_eq!(lines, vec!["x".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_field_string_has_quotes() {
        let mut sv = StructValue::new();
        sv.insert("msg", Value::String("ok".into()));
        let lines = render_value(&Value::Struct(sv), RenderMode::TopLevel);
        assert_eq!(lines, vec!["    msg: \"ok\"".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_matrix_right_aligned() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("tensor");
        let lines = format_for_disp(&Value::Tensor(tensor));
        assert_eq!(
            lines,
            vec!["     1       2".to_string(), "     3       4".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_tensor_three_dimensional_pages() {
        let tensor =
            Tensor::new((1..=8).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 2, 2]).unwrap();
        let lines = format_for_disp(&Value::Tensor(tensor));
        assert_eq!(
            lines,
            vec![
                "(:,:,1) =".to_string(),
                "     1       3".to_string(),
                "     2       4".to_string(),
                String::new(),
                "(:,:,2) =".to_string(),
                "     5       7".to_string(),
                "     6       8".to_string()
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_tensor_three_dimensional_pages() {
        let data: Vec<(f64, f64)> = vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.5),
            (6.0, 0.5),
            (7.0, 0.5),
            (8.0, 0.5),
        ];
        let tensor = ComplexTensor::new(data, vec![2, 2, 2]).unwrap();
        let lines = format_for_disp(&Value::ComplexTensor(tensor));
        assert_eq!(
            lines,
            vec![
                "(:,:,1) =".to_string(),
                "     1       3".to_string(),
                "     2       4".to_string(),
                String::new(),
                "(:,:,2) =".to_string(),
                "5+0.5000i  7+0.5000i".to_string(),
                "6+0.5000i  8+0.5000i".to_string()
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_array_left_aligned() {
        let array = StringArray::new(
            vec![
                "alpha".into(),
                "gamma".into(),
                "beta".into(),
                "delta".into(),
            ],
            vec![2, 2],
        )
        .expect("string array");
        let lines = format_for_disp(&Value::StringArray(array));
        assert_eq!(
            lines,
            vec![
                "\"alpha\"  \"beta\"".to_string(),
                "\"gamma\"  \"delta\"".to_string()
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_array_summaries() {
        let cell =
            make_cell(vec![Value::Num(1.0), Value::String("alpha".into())], 1, 2).expect("cell");
        let lines = format_for_disp(&cell);
        assert_eq!(lines, vec!["    [1]  \"alpha\"".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_field_matrix_summarised() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
        let mut sv = StructValue::new();
        sv.insert("A", Value::Tensor(tensor));
        let lines = render_value(&Value::Struct(sv), RenderMode::TopLevel);
        assert_eq!(lines, vec!["    A: [2x2 double]".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integer_64_bit_display() {
        let lines = format_for_disp(&Value::Int(IntValue::U64(u64::MAX)));
        assert_eq!(lines, vec![u64::MAX.to_string()]);
    }

    #[test]
    fn every_integer_scalar_class_formats_exactly() {
        for (value, expected) in [
            (IntValue::I8(i8::MIN), i8::MIN.to_string()),
            (IntValue::I16(i16::MIN), i16::MIN.to_string()),
            (IntValue::I32(i32::MIN), i32::MIN.to_string()),
            (IntValue::I64(i64::MIN), i64::MIN.to_string()),
            (IntValue::U8(u8::MAX), u8::MAX.to_string()),
            (IntValue::U16(u16::MAX), u16::MAX.to_string()),
            (IntValue::U32(u32::MAX), u32::MAX.to_string()),
            (IntValue::U64(u64::MAX), u64::MAX.to_string()),
        ] {
            assert_eq!(format_for_disp(&Value::Int(value)), vec![expected]);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integer_tensor_display_uses_exact_backing_storage() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
                .expect("integer tensor");

        let lines = format_for_disp(&Value::Tensor(tensor));

        assert_eq!(lines, vec![format!("{}  {}", u64::MAX, 1_u64 << 63)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integer_tensor_nested_summary_preserves_class() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2])
            .expect("integer tensor");
        let mut fields = StructValue::new();
        fields.insert("values", Value::Tensor(tensor));

        let lines = render_value(&Value::Struct(fields), RenderMode::TopLevel);

        assert_eq!(lines, vec!["    values: [1x2 int16]".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn native_single_tensor_nested_summary_preserves_class() {
        let tensor = Tensor::from_numeric_storage(
            runmat_builtins::NumericStorage::F32(vec![1.0, 2.0]),
            vec![1, 2],
        )
        .expect("single tensor");
        let mut fields = StructValue::new();
        fields.insert("values", Value::Tensor(tensor));

        let lines = render_value(&Value::Struct(fields), RenderMode::TopLevel);

        assert_eq!(lines, vec!["    values: [1x2 single]".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn integer_sparse_cell_summary_preserves_class() {
        let sparse = runmat_builtins::SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
        )
        .expect("integer sparse");
        let cell = make_cell(vec![Value::SparseTensor(sparse.clone())], 1, 1).expect("cell");

        assert_eq!(
            format_for_disp(&cell),
            vec!["    [2x2 sparse uint64]".to_string()]
        );

        let mut fields = StructValue::new();
        fields.insert("values", Value::SparseTensor(sparse));
        assert_eq!(
            format_for_disp(&Value::Struct(fields)),
            vec!["    values: <sparse 2x2 uint64 nnz=2>".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn typed_complex_integer_display_uses_exact_components_and_class() {
        let storage = runmat_builtins::IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
            IntegerStorage::U64(vec![7, 0]),
        )
        .expect("matching components");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).expect("typed complex");
        assert_eq!(
            format_for_disp(&Value::ComplexTensor(tensor.clone())),
            vec![format!("{}+7i  {}", u64::MAX, 1_u64 << 63)]
        );

        let mut fields = StructValue::new();
        fields.insert("values", Value::ComplexTensor(tensor.clone()));
        assert_eq!(
            render_value(&Value::Struct(fields), RenderMode::TopLevel),
            vec!["    values: [1x2 complex uint64]".to_string()]
        );

        let cell = make_cell(vec![Value::ComplexTensor(tensor)], 1, 1).expect("cell");
        assert_eq!(
            format_for_disp(&cell),
            vec!["    [1x2 complex uint64]".to_string()]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn disp_accepts_gpu_tensor() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                futures::executor::block_on(disp_builtin(Value::GpuTensor(handle), Vec::new()))
                    .expect("disp should succeed");
            assert_eq!(result, Value::Tensor(Tensor::zeros(vec![0, 0])));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn disp_gathers_wgpu_integer_tensor_exactly() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("register wgpu provider");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let values = [(1_u64 << 53) + 1, u64::MAX];
        let handle = provider
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data: runmat_accelerate_api::HostIntegerDataView::U64(&values),
                shape: &[1, 2],
            })
            .expect("upload integer");
        crate::console::reset_thread_buffer();
        futures::executor::block_on(disp_builtin(Value::GpuTensor(handle), Vec::new()))
            .expect("disp");
        let text = crate::console::take_thread_buffer()
            .into_iter()
            .map(|entry| entry.text)
            .collect::<String>();
        assert!(text.contains("9007199254740993"));
        assert!(text.contains("18446744073709551615"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn disp_rejects_extra_arguments() {
        let err = futures::executor::block_on(disp_builtin(
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(2))],
        ))
        .expect_err("expected error");
        assert!(err.contains("too many input arguments"));
    }
}
