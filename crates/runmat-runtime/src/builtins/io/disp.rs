//! MATLAB-compatible `disp` builtin with GPU-aware formatting semantics.

use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, IntValue, LogicalArray, StringArray, StructValue, Tensor,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::common::char_row_to_string;
use crate::console::{record_console_output, ConsoleStream};
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

#[runtime_builtin(
    name = "disp",
    category = "io",
    summary = "Display value in the Command Window without returning output.",
    keywords = "disp,display,print,gpu",
    sink = true,
    accel = "sink",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::disp"
)]
async fn disp_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(("disp: too many input arguments".to_string()).into());
    }

    let host_value = gather_if_needed_async(&value)
        .await
        .map_err(|e| format!("disp: {e}"))?;
    let lines = format_for_disp(&host_value);

    if lines.is_empty() {
        record_console_output(ConsoleStream::Stdout, "");
    } else {
        for line in lines {
            record_console_output(ConsoleStream::Stdout, line.as_str());
        }
    }

    Ok(empty_return_value())
}

fn format_for_disp(value: &Value) -> Vec<String> {
    render_value(value, RenderMode::TopLevel)
}

fn render_value(value: &Value, mode: RenderMode) -> Vec<String> {
    match value {
        Value::String(text) => match mode {
            RenderMode::TopLevel => split_lines(text),
            RenderMode::Nested => vec![quote_double(text)],
        },
        Value::CharArray(array) => format_char_array(array, mode),
        Value::StringArray(array) => format_string_array(array, mode),
        Value::Num(n) => vec![format_scalar_number(*n)],
        Value::Int(i) => vec![format_int(i)],
        Value::Bool(flag) => vec![if *flag { "1".into() } else { "0".into() }],
        Value::Tensor(tensor) => match mode {
            RenderMode::TopLevel => format_numeric_tensor(tensor),
            RenderMode::Nested => format_numeric_tensor_nested(tensor),
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
        | Value::Closure(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => {
            vec![value.to_string()]
        }
        Value::GpuTensor(_) => vec!["gpuArray".to_string()],
    }
}

fn format_numeric_tensor(tensor: &Tensor) -> Vec<String> {
    let shape = canonical_dims(&tensor.shape);
    if tensor.data.is_empty() {
        if shape.len() == 2 && shape[0] == 0 && shape[1] == 0 {
            return vec!["[]".to_string()];
        }
        if shape.contains(&0) {
            return vec![format!("Empty matrix: {}", dims_to_by_string(&shape))];
        }
        return vec!["[]".to_string()];
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
        if tensor.data.len() == 1 {
            return vec![format_scalar_number(tensor.data[0])];
        }
        return format_table(
            rows,
            cols,
            0,
            NUMERIC_MIN_COLUMN_WIDTH,
            Align::Right,
            |r, c| {
                let idx = r + c * rows;
                format_scalar_number(tensor.data[idx])
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
                format_scalar_number(tensor.data[idx])
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
    if tensor.data.is_empty() {
        return vec!["[]".to_string()];
    }
    if tensor.data.len() == 1 {
        return vec![format_scalar_number(tensor.data[0])];
    }
    let shape = canonical_dims(&tensor.shape);
    vec![format!("[{} double]", dims_to_string(&shape))]
}

fn format_complex_tensor(tensor: &ComplexTensor) -> Vec<String> {
    let shape = canonical_dims(&tensor.shape);
    if tensor.data.is_empty() {
        return vec!["[]".to_string()];
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
        if tensor.data.len() == 1 {
            let (re, im) = tensor.data[0];
            return vec![format!("{}", Value::Complex(re, im))];
        }
        return format_table(
            rows,
            cols,
            0,
            NUMERIC_MIN_COLUMN_WIDTH,
            Align::Right,
            |r, c| {
                let idx = r + c * rows;
                let (re, im) = tensor.data[idx];
                format!("{}", Value::Complex(re, im))
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
                let (re, im) = tensor.data[idx];
                format!("{}", Value::Complex(re, im))
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
    if tensor.data.is_empty() {
        return vec!["[]".to_string()];
    }
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        return vec![format!("{}", Value::Complex(re, im))];
    }
    let shape = canonical_dims(&tensor.shape);
    vec![format!("[{} complex double]", dims_to_string(&shape))]
}

fn format_logical_array(logical: &LogicalArray) -> Vec<String> {
    if logical.data.is_empty() {
        return vec!["[]".to_string()];
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
            RenderMode::TopLevel => vec![String::new()],
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

    if shape.len() > 2 {
        return vec![format!("{} string array", dims_to_string(&shape))];
    }
    let rows = shape[0];
    let cols = shape.get(1).copied().unwrap_or(1);
    if rows == 0 || cols == 0 {
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
    if cell.shape.len() > 2 {
        return vec![format!(
            "{} cell",
            dims_to_string(&canonical_dims(&cell.shape))
        )];
    }
    let rows = cell.rows;
    let cols = cell.cols;
    if rows == 0 || cols == 0 {
        return vec![format!(
            "{} cell",
            dims_to_string(&canonical_dims(&cell.shape))
        )];
    }
    format_table(rows, cols, CELL_ROW_INDENT, 1, Align::Left, |r, c| {
        let idx = r * cols + c;
        let handle = cell.data[idx].clone();
        let value = unsafe { &*handle.as_raw() };
        summarize_for_cell(value)
    })
}

fn summarize_for_cell(value: &Value) -> String {
    match value {
        Value::Num(n) => format!("[{}]", format_scalar_number(*n)),
        Value::Int(i) => format!("[{}]", format_int(i)),
        Value::Bool(flag) => format!("[{}]", if *flag { 1 } else { 0 }),
        Value::Complex(re, im) => format!("[{}]", Value::Complex(*re, *im)),
        Value::Tensor(tensor) => {
            if tensor.data.is_empty() {
                "[]".to_string()
            } else if tensor.data.len() == 1 {
                format!("[{}]", format_scalar_number(tensor.data[0]))
            } else {
                format!(
                    "[{} double]",
                    dims_to_string(&canonical_dims(&tensor.shape))
                )
            }
        }
        Value::ComplexTensor(tensor) => {
            if tensor.data.is_empty() {
                "[]".to_string()
            } else if tensor.data.len() == 1 {
                let (re, im) = tensor.data[0];
                format!("[{}]", Value::Complex(re, im))
            } else {
                format!(
                    "[{} complex double]",
                    dims_to_string(&canonical_dims(&tensor.shape))
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
        Value::Struct(_) => "[1x1 struct]".to_string(),
        Value::Cell(inner) => format!("[{} cell]", dims_to_string(&canonical_dims(&inner.shape))),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => value.to_string(),
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

fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::make_cell;
    use runmat_builtins::{ComplexTensor, IntValue, StringArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_scalar_without_quotes() {
        let lines = format_for_disp(&Value::String("Simulation complete.".into()));
        assert_eq!(lines, vec!["Simulation complete.".to_string()]);
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
                "5+0.5i  7+0.5i".to_string(),
                "6+0.5i  8+0.5i".to_string()
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn disp_accepts_gpu_tensor() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
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
    fn disp_gathers_wgpu_tensor() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("register wgpu provider");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        futures::executor::block_on(disp_builtin(Value::GpuTensor(handle), Vec::new()))
            .expect("disp");
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
