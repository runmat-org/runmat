use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicArray {
    pub(crate) data: ExactArrayStorage<SymbolicExpr>,
    pub shape: SharedShape,
    pub rows: usize,
    pub cols: usize,
}

impl SymbolicArray {
    pub fn data(&self) -> &[SymbolicExpr] {
        &self.data
    }
    pub fn data_owner(&self) -> &ExactArrayStorage<SymbolicExpr> {
        &self.data
    }
    pub fn into_parts(self) -> (ExactArrayStorage<SymbolicExpr>, SharedShape, usize, usize) {
        (self.data, self.shape, self.rows, self.cols)
    }
    pub fn try_data_mut_unique(&mut self) -> Option<&mut [SymbolicExpr]> {
        self.data.try_as_mut_slice_unique()
    }
    pub fn make_data_unique_unbounded(&mut self) -> Result<(), crate::ExactArrayMutationError> {
        self.data.make_unique_unbounded()
    }
    pub fn new(data: Vec<SymbolicExpr>, shape: impl Into<SharedShape>) -> Result<Self, String> {
        let shape = shape.into();
        let expected = shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| "SymbolicArray shape exceeds platform limits".to_owned())?;
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
            data: if data.capacity() == 0 {
                ExactArrayStorage::empty()
            } else {
                ExactArrayStorage::shared_unbounded(data)
            },
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
