//! Original model diagnostic adapter to the shared metadata-only CSC row index.
use super::*;
use crate::builtins::math::optim::common::csc_rows::CscRows;
pub(super) use crate::builtins::math::optim::common::csc_rows::Entry;
use runmat_value::SparseTensor;
pub(super) struct SparseRows {
    rows: CscRows,
}
impl SparseRows {
    pub(super) async fn prepare(
        matrix: &SparseTensor,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            rows: CscRows::prepare(
                matrix.cols,
                &matrix.col_ptrs,
                &matrix.row_indices,
                |error| policy.storage(error),
                |error| policy.reserve(error),
                c,
            )
            .await?,
        })
    }
    pub(super) fn row(&self, row: usize) -> &[Entry] {
        self.rows.row(row)
    }
}
