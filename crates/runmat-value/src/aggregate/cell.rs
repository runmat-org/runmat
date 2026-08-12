use super::*;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CellArray {
    pub data: Vec<Value>,
    /// Full MATLAB-visible shape vector. Cell payloads retain their historical
    /// row-major layout within each 2-D page.
    pub shape: Vec<usize>,
    /// Cached row count for 2-D interop; equals `shape[0]` when present.
    pub rows: usize,
    /// Cached column count for 2-D interop; equals `shape[1]` when present, otherwise 1 (or 0 for empty).
    pub cols: usize,
}

impl CellArray {
    pub fn new(data: Vec<Value>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new_with_shape(data, vec![rows, cols])
    }

    pub fn new_with_shape(data: Vec<Value>, shape: Vec<usize>) -> Result<Self, String> {
        let expected = total_len(&shape)
            .ok_or_else(|| "Cell data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Cell data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = shape_rows_cols(&shape);
        Ok(CellArray {
            data,
            shape,
            rows,
            cols,
        })
    }

    pub fn from_column_major(data: Vec<Value>, shape: Vec<usize>) -> Result<Self, String> {
        let normalized = match shape.as_slice() {
            [] => vec![0, 0],
            [length] => vec![1, *length],
            _ => shape,
        };
        let expected = total_len(&normalized)
            .ok_or_else(|| "Cell data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Cell data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                normalized,
                expected
            ));
        }
        let rows = normalized[0];
        let cols = normalized[1];
        let pages = if normalized.len() <= 2 {
            1
        } else {
            total_len(&normalized[2..])
                .ok_or_else(|| "Cell page shape exceeds platform limits".to_string())?
        };
        let mut row_major = Vec::with_capacity(data.len());
        for page in 0..pages {
            let page_offset = page * rows * cols;
            for row in 0..rows {
                for col in 0..cols {
                    row_major.push(data[page_offset + row + col * rows].clone());
                }
            }
        }
        Self::new_with_shape(row_major, normalized)
    }

    pub fn to_column_major(&self) -> Vec<Value> {
        let pages = if self.shape.len() <= 2 {
            1
        } else {
            self.shape[2..].iter().product()
        };
        let mut column_major = Vec::with_capacity(self.data.len());
        for page in 0..pages {
            let page_offset = page * self.rows * self.cols;
            for col in 0..self.cols {
                for row in 0..self.rows {
                    column_major.push(self.data[page_offset + row * self.cols + col].clone());
                }
            }
        }
        column_major
    }

    pub fn get(&self, row: usize, col: usize) -> Result<Value, String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!(
                "Cell index ({row}, {col}) out of bounds for {}x{} cell array",
                self.rows, self.cols
            ));
        }
        Ok(self.data[row * self.cols + col].clone())
    }
}

pub(crate) fn total_len(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

pub(crate) fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    if shape.is_empty() {
        return (0, 0);
    }
    if shape.len() == 1 {
        return (1, shape[0]);
    }
    (shape[0], shape[1])
}

impl fmt::Display for CellArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        if self.shape.len() > 2 {
            return write!(f, "{} cell array", dims.join("x"));
        }
        write!(f, "{}x{} cell array", self.rows, self.cols)?;
        if self.rows == 0 || self.cols == 0 {
            return Ok(());
        }
        for r in 0..self.rows {
            writeln!(f)?;
            write!(f, "  ")?;
            for c in 0..self.cols {
                if c > 0 {
                    write!(f, "  ")?;
                }
                let value = self.get(r, c).unwrap_or_else(|_| Value::Num(f64::NAN));
                write!(f, "{{{value}}}")?;
            }
        }
        Ok(())
    }
}
