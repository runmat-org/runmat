use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use runmat_value::SparseTensor;

/// The caller has already validated canonical CSC through SparseLinearOperator.
/// This reuses the exact index slices and borrows real numeric payloads.
pub(super) fn borrowed_csc<'a, T>(
    sparse: &'a SparseTensor,
    values: &'a [T],
) -> SparseColMatRef<'a, usize, T> {
    let symbolic = SymbolicSparseColMatRef::new_checked(
        sparse.rows,
        sparse.cols,
        &sparse.col_ptrs,
        None,
        &sparse.row_indices,
    );
    SparseColMatRef::new(symbolic, values)
}
