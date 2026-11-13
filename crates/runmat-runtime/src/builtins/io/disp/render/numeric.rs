use super::*;

pub(super) fn format_numeric_tensor(tensor: &Tensor) -> Vec<String> {
    let shape = canonical_dims(tensor.shape());
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

pub(super) fn format_numeric_tensor_pages(tensor: &Tensor, dims: &[usize]) -> Vec<String> {
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

pub(super) fn format_numeric_tensor_nested(tensor: &Tensor) -> Vec<String> {
    let len = tensor::tensor_element_len(tensor);
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![format_tensor_value(tensor, 0)];
    }
    let shape = canonical_dims(tensor.shape());
    let class_name = tensor.numeric_dtype().class_name();
    vec![format!("[{} {class_name}]", dims_to_string(&shape))]
}

pub(super) fn format_tensor_value(tensor: &Tensor, index: usize) -> String {
    let value = tensor
        .numeric_value_at(index)
        .expect("index within authoritative numeric storage");
    if let Some(value) = value.into_int_value() {
        format_int(&value)
    } else {
        format_scalar_number(value.materialize_f64())
    }
}

pub(super) fn format_complex_tensor(tensor: &ComplexTensor) -> Vec<String> {
    let shape = canonical_dims(tensor.shape());
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

pub(super) fn format_complex_tensor_pages(tensor: &ComplexTensor, dims: &[usize]) -> Vec<String> {
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

pub(super) fn format_complex_tensor_nested(tensor: &ComplexTensor) -> Vec<String> {
    let len = tensor::complex_tensor_element_len(tensor);
    if len == 0 {
        return vec!["[]".to_string()];
    }
    if len == 1 {
        return vec![tensor.format_element(0)];
    }
    let shape = canonical_dims(tensor.shape());
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

pub(super) fn format_logical_array(logical: &LogicalArray) -> Vec<String> {
    if logical.data.is_empty() {
        return Vec::new();
    }
    match tensor::logical_to_tensor(logical) {
        Ok(tensor) => format_numeric_tensor(&tensor),
        Err(_) => split_lines(&logical.to_string()),
    }
}

pub(super) fn format_logical_array_nested(logical: &LogicalArray) -> Vec<String> {
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
