use super::*;

pub(super) fn format_table<F>(
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

pub(super) fn format_int(value: &IntValue) -> String {
    if get_display_format() == FormatMode::Hex {
        return match value {
            IntValue::I8(value) => format!("{:02x}", *value as u8),
            IntValue::I16(value) => format!("{:04x}", *value as u16),
            IntValue::I32(value) => format!("{:08x}", *value as u32),
            IntValue::I64(value) => format!("{:016x}", *value as u64),
            IntValue::U8(value) => format!("{value:02x}"),
            IntValue::U16(value) => format!("{value:04x}"),
            IntValue::U32(value) => format!("{value:08x}"),
            IntValue::U64(value) => format!("{value:016x}"),
        };
    }
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

pub(super) fn format_scalar_number(value: f64) -> String {
    format!("{}", Value::Num(value))
}

pub(super) fn split_lines(text: &str) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    text.lines().map(|line| line.to_string()).collect()
}

pub(super) fn quote_double(text: &str) -> String {
    let escaped = text.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

pub(super) fn page_header(tail_indices: &[usize]) -> String {
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

pub(super) fn increment_multi_index(indices: &mut [usize], dims: &[usize]) -> bool {
    for (idx, dim) in indices.iter_mut().zip(dims.iter()) {
        *idx += 1;
        if *idx < *dim {
            return true;
        }
        *idx = 0;
    }
    false
}

pub(super) fn linear_index_with_tail(
    dims: &[usize],
    row: usize,
    col: usize,
    tail: &[usize],
) -> usize {
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

pub(super) fn canonical_dims(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![1, shape[0]],
        _ => shape.to_vec(),
    }
}

pub(super) fn dims_to_string(dims: &[usize]) -> String {
    if dims.is_empty() {
        return "0x0".to_string();
    }
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

pub(super) fn dims_to_by_string(dims: &[usize]) -> String {
    if dims.is_empty() {
        return "0-by-0".to_string();
    }
    dims.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("-by-")
}
