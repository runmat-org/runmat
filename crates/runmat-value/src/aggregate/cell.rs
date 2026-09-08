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
        let mut source = data.into_iter().map(Some).collect::<Vec<_>>();
        let mut row_major = Vec::with_capacity(source.len());
        for page in 0..pages {
            let page_offset = page * rows * cols;
            for row in 0..rows {
                for col in 0..cols {
                    let index = page_offset + row + col * rows;
                    let value = source
                        .get_mut(index)
                        .and_then(Option::take)
                        .ok_or_else(|| "cell storage does not match its shape".to_string())?;
                    row_major.push(value);
                }
            }
        }
        Self::new_with_shape(row_major, normalized)
    }

    pub fn to_column_major(&self) -> Vec<Value> {
        self.iter_column_major().cloned().collect()
    }

    /// Move values into MATLAB-visible linear order without cloning payloads.
    pub fn into_column_major(self) -> Result<Vec<Value>, String> {
        if self.data.is_empty() || self.rows <= 1 || self.cols <= 1 {
            return Ok(self.data);
        }
        let page_len = self.rows * self.cols;
        let mut source = self.data.into_iter().map(Some).collect::<Vec<_>>();
        let mut ordered = Vec::with_capacity(source.len());
        for page_offset in (0..source.len()).step_by(page_len) {
            for column in 0..self.cols {
                for row in 0..self.rows {
                    let index = page_offset + row * self.cols + column;
                    let value = source
                        .get_mut(index)
                        .and_then(Option::take)
                        .ok_or_else(|| "cell storage does not match its shape".to_string())?;
                    ordered.push(value);
                }
            }
        }
        Ok(ordered)
    }

    /// Iterate in MATLAB-visible linear order without cloning cell payloads.
    pub fn iter_column_major(&self) -> impl Iterator<Item = &Value> {
        let page_len = self.rows * self.cols;
        (0..self.data.len()).map(move |linear| {
            let page = linear / page_len.max(1);
            let within_page = linear % page_len.max(1);
            let row = within_page % self.rows.max(1);
            let col = within_page / self.rows.max(1);
            &self.data[page * page_len + row * self.cols + col]
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_column_major_iteration_matches_owned_projection() {
        let expected = vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let array = CellArray::from_column_major(expected.clone(), vec![2, 2]).unwrap();
        let borrowed = array.iter_column_major().cloned().collect::<Vec<_>>();
        assert_eq!(borrowed, array.to_column_major());
        assert_eq!(borrowed, expected);
    }

    #[test]
    fn borrowed_column_major_iteration_handles_empty_shape() {
        let array = CellArray::new(Vec::new(), 0, 0).unwrap();
        assert_eq!(array.iter_column_major().count(), 0);
    }

    #[test]
    fn owned_column_major_projection_moves_the_same_values() {
        let expected = vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let array = CellArray::from_column_major(expected.clone(), vec![2, 2]).unwrap();
        assert_eq!(array.into_column_major().unwrap(), expected);
    }
}
