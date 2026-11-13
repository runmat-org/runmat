//! Borrowed original linear geometry, never the repair lift or bound rows.
use clarabel::algebra::CscMatrix;

pub(in crate::builtins::math::optim) struct OriginalLinear<'a> {
    pub(super) matrix: Option<&'a CscMatrix<f64>>,
    pub(super) rhs: &'a [f64],
    pub(super) variables: usize,
    pub(super) equalities: usize,
}
impl<'a> OriginalLinear<'a> {
    pub fn variables(&self) -> usize {
        self.variables
    }
    pub fn rows(&self) -> usize {
        self.rhs.len()
    }
    pub fn equalities(&self) -> usize {
        self.equalities
    }
    pub fn rhs(&self) -> &'a [f64] {
        self.rhs
    }
    /// O(1) lookup for an admitted metadata-only transpose. The ordinal is
    /// local to this original column, not an unrestricted backing-store slot.
    pub fn entry(&self, column: usize, ordinal: usize) -> Option<(usize, f64)> {
        if column >= self.variables {
            return None;
        }
        let matrix = self.matrix?;
        let slot = matrix.colptr[column].checked_add(ordinal)?;
        if slot >= matrix.colptr[column + 1] {
            return None;
        }
        let row = matrix.rowval[slot];
        (row < self.rhs.len()).then(|| (row, matrix.nzval[slot]))
    }
    /// Original CSC order is canonical. Prefix filtering therefore ends
    /// before any distance rows, and the checked variable extent excludes
    /// every auxiliary column. No sparse storage is copied or exposed.
    pub fn column(&self, column: usize) -> Option<impl Iterator<Item = (usize, f64)> + 'a> {
        if column >= self.variables {
            return None;
        }
        let rows = self.rhs.len();
        Some(self.matrix.into_iter().flat_map(move |matrix| {
            (matrix.colptr[column]..matrix.colptr[column + 1])
                .take_while(move |&slot| matrix.rowval[slot] < rows)
                .map(move |slot| (matrix.rowval[slot], matrix.nzval[slot]))
        }))
    }
}
