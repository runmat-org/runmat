//! Original one-shot owners adapt to the same immutable borrowed kernel used
//! by public retained DATA, keeping their existing issuer checks unchanged.
use super::{Qr, SparseCompleteQr, SparseCompleteQrError as Error, SparseQrScalar, Work};
use crate::context::RuntimeContext;
mod borrowed;
pub(in crate::builtins::math::linalg::ops::sparse_factor) use borrowed::{
    BorrowedCompleteQr, BorrowedCsc, BorrowedQr,
};

fn view<T: SparseQrScalar>(factor: &Qr<T>) -> BorrowedQr<'_, T> {
    BorrowedQr {
        rows: factor.rows,
        columns: factor.cols,
        rank: factor.rank,
        r: BorrowedCsc {
            pointers: factor.r.ptr.as_slice(),
            rows: factor.r.rows.as_slice(),
            values: factor.r.values.as_slice(),
        },
        reflectors: BorrowedCsc {
            pointers: factor.reflectors.ptr.as_slice(),
            rows: factor.reflectors.rows.as_slice(),
            values: factor.reflectors.values.as_slice(),
        },
        permutation: factor.permutation.as_slice(),
    }
}

impl<T: SparseQrScalar> SparseCompleteQr<T> {
    /// Caller-private sink; the factor remains immutable and each invocation
    /// gets independently claimed RHS/coordinate work under the actual owner.
    pub(in crate::builtins::math::linalg) async fn solve_with_output<S: Copy>(
        &self,
        input: &[S],
        columns: usize,
        runtime: &RuntimeContext,
        to_native: impl FnMut(S) -> T,
        output: impl FnMut(usize, T) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.solve_optional_with_output(input, columns, false, Some(runtime), to_native, output)
            .await
    }
    pub(in crate::builtins::math::linalg) async fn solve_optional_with_output<S: Copy>(
        &self,
        input: &[S],
        columns: usize,
        adjoint: bool,
        runtime: Option<&RuntimeContext>,
        mut to_native: impl FnMut(S) -> T,
        output: impl FnMut(usize, T) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let mut work = Work::for_optional_policy(runtime, self.fence.as_ref())?;
        BorrowedCompleteQr {
            first: view(&self.first),
            completion: self.cod.as_ref().map(view),
        }
        .solve(
            input,
            columns,
            adjoint,
            |_, value| Some(to_native(value)),
            output,
            &mut work,
        )
        .await
    }
}
