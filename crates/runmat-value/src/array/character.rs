use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct CharArray {
    pub data: Vec<char>,
    /// Full MATLAB-visible shape vector. Character payloads retain their
    /// historical row-major layout within each 2-D page.
    pub shape: Vec<usize>,
    /// Cached row count for 2-D interop; equals `shape[0]` when present.
    pub rows: usize,
    /// Cached column count for 2-D interop; equals `shape[1]` when present,
    /// otherwise 1 (or 0 for an empty shape).
    pub cols: usize,
}

impl CharArray {
    pub fn new_row(s: &str) -> Self {
        CharArray {
            data: s.chars().collect(),
            shape: vec![1, s.chars().count()],
            rows: 1,
            cols: s.chars().count(),
        }
    }
    pub fn new(data: Vec<char>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new_with_shape(data, vec![rows, cols])
    }
    pub fn new_with_shape(data: Vec<char>, shape: Vec<usize>) -> Result<Self, String> {
        let expected = total_len(&shape)
            .ok_or_else(|| "Char data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Char data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = shape_rows_cols(&shape);
        Ok(CharArray {
            data,
            shape,
            rows,
            cols,
        })
    }
    pub fn from_column_major(data: Vec<char>, shape: Vec<usize>) -> Result<Self, String> {
        let normalized = match shape.as_slice() {
            [] => vec![0, 0],
            [length] => vec![1, *length],
            _ => shape,
        };
        let expected = total_len(&normalized)
            .ok_or_else(|| "Char data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Char data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                normalized,
                expected
            ));
        }
        let rows = normalized[0];
        let cols = normalized[1];
        let pages = total_len(&normalized[2..])
            .ok_or_else(|| "Char page shape exceeds platform limits".to_string())?;
        let pages = if normalized.len() <= 2 { 1 } else { pages };
        let mut row_major = Vec::with_capacity(data.len());
        for page in 0..pages {
            let page_offset = page * rows * cols;
            for row in 0..rows {
                for col in 0..cols {
                    row_major.push(data[page_offset + row + col * rows]);
                }
            }
        }
        Self::new_with_shape(row_major, normalized)
    }
    pub fn to_column_major(&self) -> Vec<char> {
        let rows = self.rows;
        let cols = self.cols;
        let pages = if self.shape.len() <= 2 {
            1
        } else {
            self.shape[2..].iter().product()
        };
        let mut column_major = Vec::with_capacity(self.data.len());
        for page in 0..pages {
            let page_offset = page * rows * cols;
            for col in 0..cols {
                for row in 0..rows {
                    column_major.push(self.data[page_offset + row * cols + col]);
                }
            }
        }
        column_major
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the character contents when this value is a MATLAB character
    /// row vector. `Display` intentionally renders arrays for the console and
    /// therefore includes layout whitespace; semantic string conversions must
    /// use this method instead.
    pub fn row_string(&self) -> Option<String> {
        (self.rows == 1).then(|| self.data.iter().collect())
    }
}
