use super::*;

pub(crate) fn format_for_disp(value: &Value) -> Vec<String> {
    render_value(value, RenderMode::TopLevel)
}

pub(crate) fn format_for_display(value: &Value) -> Vec<String> {
    let rendered = format_for_disp(value);
    if !rendered.is_empty() {
        return rendered;
    }
    match value {
        Value::Tensor(tensor) => vec![format_empty_numeric_tensor_for_display(tensor)],
        Value::ComplexTensor(_) | Value::LogicalArray(_) => vec!["[]".to_string()],
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

fn format_empty_numeric_tensor_for_display(tensor: &Tensor) -> String {
    let shape = canonical_dims(tensor.shape());
    let dtype = tensor.numeric_dtype();
    if dtype == runmat_value::NumericDType::F64 && shape == [0, 0] {
        return "[]".to_string();
    }
    let dimensions = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("×");
    let noun = if shape.len() <= 2 { "matrix" } else { "array" };
    format!("{dimensions} empty {} {noun}", dtype.class_name())
}

pub(super) fn render_value(value: &Value, mode: RenderMode) -> Vec<String> {
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
        Value::Object(obj) if obj.is_class(runmat_types::standard::DATETIME) => match mode {
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
        Value::Object(obj) if obj.is_class(runmat_types::standard::DURATION) => match mode {
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
        Value::String(runmat_value::StringElement::Missing) => vec!["<missing>".to_owned()],
        Value::String(runmat_value::StringElement::Text(text)) => match mode {
            RenderMode::TopLevel => split_lines(text),
            RenderMode::Nested => vec![quote_double(text)],
        },
        Value::Symbolic(expr) => vec![expr.to_string()],
        Value::SymbolicArray(array) => match mode {
            RenderMode::TopLevel => array.to_string().lines().map(ToString::to_string).collect(),
            RenderMode::Nested => vec![format!(
                "[{} sym]",
                dims_to_string(&canonical_dims(&array.shape))
            )],
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
        Value::StructArray(array) => vec![array.to_string()],
        Value::Cell(cell) => match mode {
            RenderMode::TopLevel => format_cell(cell),
            RenderMode::Nested => vec![format!(
                "{} cell",
                dims_to_string(&canonical_dims(&cell.shape))
            )],
        },
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::OutputList(_) => {
            vec![value.to_string()]
        }
        Value::GpuTensor(_) => vec!["gpuArray".to_string()],
    }
}

mod numeric;
use numeric::*;

mod containers;
use containers::*;

mod utility;
use utility::*;
