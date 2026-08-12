use super::*;

pub(crate) fn format_integer_complex_value(real: &IntValue, imag: &IntValue) -> String {
    if imag.is_zero() {
        return format_integer_value(real);
    }
    if real.is_zero() {
        return format!("{}i", format_integer_value(imag));
    }
    if integer_value_is_negative(imag) {
        return format!(
            "{}-{}i",
            format_integer_value(real),
            format_integer_magnitude(imag)
        );
    }
    format!(
        "{}+{}i",
        format_integer_value(real),
        format_integer_value(imag)
    )
}

fn format_integer_value(value: &IntValue) -> String {
    match value {
        IntValue::I8(value) => value.to_string(),
        IntValue::I16(value) => value.to_string(),
        IntValue::I32(value) => value.to_string(),
        IntValue::I64(value) => value.to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

fn integer_value_is_negative(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value < 0,
        IntValue::I16(value) => *value < 0,
        IntValue::I32(value) => *value < 0,
        IntValue::I64(value) => *value < 0,
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => false,
    }
}

fn format_integer_magnitude(value: &IntValue) -> String {
    match value {
        IntValue::I8(value) => value.unsigned_abs().to_string(),
        IntValue::I16(value) => value.unsigned_abs().to_string(),
        IntValue::I32(value) => value.unsigned_abs().to_string(),
        IntValue::I64(value) => value.unsigned_abs().to_string(),
        IntValue::U8(value) => value.to_string(),
        IntValue::U16(value) => value.to_string(),
        IntValue::U32(value) => value.to_string(),
        IntValue::U64(value) => value.to_string(),
    }
}

const MAX_ND_DISPLAY_ELEMENTS: usize = 4096;

pub(crate) fn should_expand_nd_display(shape: &[usize]) -> bool {
    shape.len() > 2
        && matches!(
            total_len(shape),
            Some(total) if total > 0 && total <= MAX_ND_DISPLAY_ELEMENTS
        )
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn decode_page_coords(mut page_index: usize, page_shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(page_shape.len());
    for &dim in page_shape {
        if dim == 0 {
            coords.push(0);
        } else {
            coords.push(page_index % dim);
            page_index /= dim;
        }
    }
    coords
}

pub(crate) fn write_nd_pages(
    f: &mut fmt::Formatter<'_>,
    shape: &[usize],
    mut write_element: impl FnMut(&mut fmt::Formatter<'_>, usize) -> fmt::Result,
) -> fmt::Result {
    if shape.len() <= 2 {
        return Ok(());
    }
    let rows = shape[0];
    let cols = shape[1];
    if rows == 0 || cols == 0 {
        return write!(f, "[]");
    }
    let Some(page_count) = total_len(&shape[2..]) else {
        return write!(f, "Tensor(shape={shape:?})");
    };
    if page_count == 0 {
        return write!(f, "[]");
    }
    let strides = column_major_strides(shape);
    for page_index in 0..page_count {
        if page_index > 0 {
            write!(f, "\n\n")?;
        }
        let coords = decode_page_coords(page_index, &shape[2..]);
        write!(f, "(:, :")?;
        for &coord in &coords {
            write!(f, ", {}", coord + 1)?;
        }
        write!(f, ") =")?;

        let mut page_base = 0usize;
        for (offset, &coord) in coords.iter().enumerate() {
            page_base += coord * strides[offset + 2];
        }
        for r in 0..rows {
            writeln!(f)?;
            write!(f, "  ")?;
            for c in 0..cols {
                if c > 0 {
                    write!(f, "  ")?;
                }
                let linear = page_base + r + c * rows;
                write_element(f, linear)?;
            }
        }
    }
    Ok(())
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format_element = |idx: usize| {
            let value = self
                .numeric_value_at(idx)
                .expect("display index is within tensor storage");
            match value.into_int_value() {
                Some(value) => value.decimal_string(),
                None => format_number(value.materialize_f64()),
            }
        };

        match self.shape.len() {
            0 | 1 => {
                // Treat as row vector for display
                write!(f, "[")?;
                for i in 0..self.len() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", format_element(i))?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.rows();
                let cols = self.cols();
                // Display as matrix
                for r in 0..rows {
                    writeln!(f)?;
                    write!(f, "  ")?; // Indent
                    for c in 0..cols {
                        if c > 0 {
                            write!(f, "  ")?;
                        }
                        write!(f, "{}", format_element(r + c * rows))?;
                    }
                }
                Ok(())
            }
            _ => {
                if should_expand_nd_display(&self.shape) {
                    write_nd_pages(f, &self.shape, |f, idx| {
                        write!(f, "{}", format_element(idx))
                    })
                } else {
                    write!(f, "Tensor(shape={:?})", self.shape)
                }
            }
        }
    }
}

impl fmt::Display for SymbolicArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.shape.len() {
            0 | 1 => {
                write!(f, "[")?;
                for (i, expr) in self.data.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{expr}")?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.rows();
                let cols = self.cols();
                for r in 0..rows {
                    writeln!(f)?;
                    write!(f, "  ")?;
                    for c in 0..cols {
                        if c > 0 {
                            write!(f, "  ")?;
                        }
                        write!(f, "{}", self.data[r + c * rows])?;
                    }
                }
                Ok(())
            }
            _ => {
                if should_expand_nd_display(&self.shape) {
                    write_nd_pages(f, &self.shape, |f, idx| write!(f, "{}", self.data[idx]))
                } else {
                    write!(f, "SymbolicArray(shape={:?})", self.shape)
                }
            }
        }
    }
}

impl fmt::Display for StringArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = match self.shape.len() {
            0 => (0, 0),
            1 => (1, self.shape[0]),
            _ => (self.shape[0], self.shape[1]),
        };
        let count = self.data.len();
        if count == 1 && rows == 1 && cols == 1 {
            let v = &self.data[0];
            if v == "<missing>" {
                return write!(f, "<missing>");
            }
            let escaped = v.replace('"', "\\\"");
            return write!(f, "\"{escaped}\"");
        }
        if self.shape.len() > 2 {
            let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
            return write!(f, "{} string array", dims.join("x"));
        }
        write!(f, "{rows}x{cols} string array")?;
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        for r in 0..rows {
            writeln!(f)?;
            write!(f, "  ")?;
            for c in 0..cols {
                if c > 0 {
                    write!(f, "  ")?;
                }
                let v = &self.data[r + c * rows];
                if v == "<missing>" {
                    write!(f, "<missing>")?;
                } else {
                    let escaped = v.replace('"', "\\\"");
                    write!(f, "\"{escaped}\"")?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for LogicalArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.data.len() == 1 {
            return write!(f, "{}", if self.data[0] != 0 { 1 } else { 0 });
        }
        match self.shape.len() {
            0 => write!(f, "[]"),
            1 => {
                write!(f, "[")?;
                for (i, v) in self.data.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", if *v != 0 { 1 } else { 0 })?;
                }
                write!(f, "]")
            }
            2 => {
                let rows = self.shape[0];
                let cols = self.shape[1];
                // Display as matrix
                for r in 0..rows {
                    writeln!(f)?;
                    write!(f, "  ")?; // Indent
                    for c in 0..cols {
                        if c > 0 {
                            write!(f, "  ")?;
                        }
                        let idx = r + c * rows;
                        write!(f, "{}", if self.data[idx] != 0 { 1 } else { 0 })?;
                    }
                }
                Ok(())
            }
            _ => {
                if should_expand_nd_display(&self.shape) {
                    write_nd_pages(f, &self.shape, |f, idx| {
                        write!(f, "{}", if self.data[idx] != 0 { 1 } else { 0 })
                    })
                } else {
                    let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
                    write!(f, "{} logical array", dims.join("x"))
                }
            }
        }
    }
}

impl fmt::Display for CharArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.len() > 2 {
            let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
            return write!(f, "{} char array", dims.join("x"));
        }
        for r in 0..self.rows {
            writeln!(f)?;
            write!(f, "  ")?; // Indent
            for c in 0..self.cols {
                let ch = self.data[r * self.cols + c];
                write!(f, "{ch}")?;
            }
        }
        Ok(())
    }
}

// From implementations for Value
