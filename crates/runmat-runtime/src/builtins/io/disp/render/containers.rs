use super::*;

fn char_row_to_string(array: &CharArray, row: usize) -> String {
    let start = row * array.cols;
    let units = &array.data()[start..start + array.cols];
    let mut rendered = String::new();
    let mut index = 0;
    while index < units.len() {
        let raw = units[index].raw();
        if (0xD800..=0xDBFF).contains(&raw) && index + 1 < units.len() {
            let low = units[index + 1].raw();
            if (0xDC00..=0xDFFF).contains(&low) {
                let scalar = 0x1_0000 + (((raw as u32 - 0xD800) << 10) | (low as u32 - 0xDC00));
                rendered.push(char::from_u32(scalar).expect("validated UTF-16 surrogate pair"));
                index += 2;
                continue;
            }
        }
        if let Some(character) = char::from_u32(raw as u32) {
            rendered.push(character);
        } else {
            rendered.push_str(&format!("\\u{raw:04X}"));
        }
        index += 1;
    }
    rendered
}

pub(super) fn format_char_array(array: &CharArray, mode: RenderMode) -> Vec<String> {
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

fn render_string_element(value: &runmat_value::StringElement) -> String {
    match value.as_str() {
        Some(text) => quote_double(text),
        None => "<missing>".to_owned(),
    }
}

pub(super) fn format_string_array(array: &StringArray, mode: RenderMode) -> Vec<String> {
    let shape = canonical_dims(&array.shape);
    let total = array.data().len();
    if matches!(mode, RenderMode::Nested) {
        if total == 0 {
            return vec![format!("{} string array", dims_to_string(&shape))];
        }
        if total == 1 {
            return vec![render_string_element(&array.data()[0])];
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
        render_string_element(&array.data()[idx])
    })
}

pub(super) fn format_struct(struct_value: &StructValue) -> Vec<String> {
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

pub(super) fn format_cell(cell: &CellArray) -> Vec<String> {
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
        let handle = cell.data()[idx].clone();
        let value = handle;
        summarize_for_cell(&value)
    })
}

pub(super) fn summarize_for_cell(value: &Value) -> String {
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
                    dims_to_string(&canonical_dims(tensor.shape())),
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
                    dims_to_string(&canonical_dims(tensor.shape())),
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
        Value::String(text) => render_string_element(text),
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
            if array.data().is_empty() {
                format!("{} string", dims_to_string(&canonical_dims(&array.shape)))
            } else if array.data().len() == 1 {
                render_string_element(&array.data()[0])
            } else {
                format!("[{} string]", dims_to_string(&canonical_dims(&array.shape)))
            }
        }
        Value::Symbolic(expr) => expr.to_string(),
        Value::SymbolicArray(array) => {
            if array.data().is_empty() {
                "[]".to_string()
            } else if array.data().len() == 1 {
                array.data()[0].to_string()
            } else {
                format!("[{} sym]", dims_to_string(&canonical_dims(&array.shape)))
            }
        }
        Value::Struct(_) => "[1x1 struct]".to_string(),
        Value::StructArray(array) => format!(
            "[{} struct]",
            dims_to_string(&canonical_dims(array.shape()))
        ),
        Value::Cell(inner) => format!("[{} cell]", dims_to_string(&canonical_dims(&inner.shape))),
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
        | Value::OutputList(_) => value.to_string(),
        Value::GpuTensor(_) => "gpuArray".to_string(),
    }
}
