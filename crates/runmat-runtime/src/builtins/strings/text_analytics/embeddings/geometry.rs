use runmat_value::Tensor;

/// Projects dense embedding matrices into their vocabulary-row geometry.
///
/// Rank-zero tensors have no matrix extent and rank-one tensors represent one
/// embedding row, matching the embedding domain's historical layout.
pub(super) fn embedding_matrix_dimensions(tensor: &Tensor) -> (usize, usize) {
    match tensor.shape() {
        [] => (0, 0),
        [columns] => (1, *columns),
        [rows, columns, ..] => (*rows, *columns),
    }
}
