use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicArray {
    pub data: Vec<SymbolicExpr>,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

impl SymbolicArray {
    pub fn new(data: Vec<SymbolicExpr>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "SymbolicArray data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        // Keep the cached `rows`/`cols` fields in lockstep with the `rows()`/`cols()`
        // accessors so the two never disagree for non-2D shapes.
        let rows = shape.first().copied().unwrap_or(1);
        let cols = shape.get(1).copied().unwrap_or(1);
        Ok(SymbolicArray {
            data,
            shape,
            rows,
            cols,
        })
    }

    pub fn new_2d(data: Vec<SymbolicExpr>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}
