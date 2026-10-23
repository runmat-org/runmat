//! Cooperative solves against the retained CSC factor; no per-RHS allocation.
use super::{ManagedSparseLu, ManagedSparseLuError, ManagedSparseScalar, Work};
use crate::context::RuntimeContext;
mod borrowed;
pub(in crate::builtins::math::linalg) use borrowed::FactorRhsGeometry;
pub(in crate::builtins::math::linalg) use borrowed::FactorSolveOptions;
pub(in crate::builtins::math::linalg) use borrowed::SingularSolvePolicy;
pub(in crate::builtins::math::linalg::ops::sparse_factor) use borrowed::{
    BorrowedLu, LuSolveInput,
};

impl<T: ManagedSparseScalar> ManagedSparseLu<T> {
    /// Native column-major RHS. `output` must be caller-private working memory:
    /// it may be partially written on failure, and must never be published in
    /// that case. The factor itself is unchanged by numerical/authority errors.
    /// A fresh invocation cannot substitute its authority for this factor's.
    pub(in crate::builtins::math::linalg) async fn solve_into(
        &mut self,
        input: &[T],
        output: &mut [T],
        columns: usize,
        runtime: &RuntimeContext,
    ) -> Result<(), ManagedSparseLuError> {
        self.solve_converted_into(
            input,
            output,
            columns,
            runtime,
            std::convert::identity,
            Some,
        )
        .await
    }

    /// Runtime complex elements can bridge component-wise to the native
    /// factor scalar without a copied, unclaimed argument/RHS adapter.
    /// The destination has the same private-working-memory contract above.
    pub(in crate::builtins::math::linalg) async fn solve_converted_into<S: Copy>(
        &mut self,
        input: &[S],
        output: &mut [S],
        columns: usize,
        runtime: &RuntimeContext,
        to_native: impl FnMut(S) -> T,
        mut from_native: impl FnMut(T) -> Option<S>,
    ) -> Result<(), ManagedSparseLuError> {
        if self.order.checked_mul(columns) != Some(input.len()) || input.len() != output.len() {
            return Err(ManagedSparseLuError::DimensionMismatch);
        }
        self.solve_with_output(input, columns, runtime, to_native, |index, value| {
            output[index] = from_native(value).ok_or(ManagedSparseLuError::NumericalBreakdown)?;
            Ok(())
        })
        .await
    }

    /// Write directly into a caller's exact native allocation (including
    /// complex setter APIs), without a second payload or transmute adapter.
    /// The sink must remain private until success and its publication fence.
    pub(in crate::builtins::math::linalg) async fn solve_with_output<S: Copy>(
        &mut self,
        input: &[S],
        columns: usize,
        runtime: &RuntimeContext,
        mut to_native: impl FnMut(S) -> T,
        write_output: impl FnMut(usize, T) -> Result<(), ManagedSparseLuError>,
    ) -> Result<(), ManagedSparseLuError> {
        if self.order.checked_mul(columns) != Some(input.len()) {
            return Err(ManagedSparseLuError::DimensionMismatch);
        }
        let mut work = Work::for_policy(runtime, self.fence.as_ref())?;
        BorrowedLu {
            order: self.order,
            l_ptr: self.l_ptr.as_slice(),
            l_rows: self.l_rows.as_slice(),
            l_values: self.l_values.as_slice(),
            u_ptr: self.u_ptr.as_slice(),
            u_rows: self.u_rows.as_slice(),
            u_values: self.u_values.as_slice(),
            row_perm: self.row_perm.as_slice(),
            col_perm: self.col_perm.as_slice(),
            row_scale: None,
        }
        .solve(
            LuSolveInput {
                values: input,
                columns,
            },
            FactorSolveOptions {
                adjoint: false,
                singular: SingularSolvePolicy::Reject,
            },
            self.rhs.as_mut_slice(),
            |_, value| Some(to_native(value)),
            write_output,
            &mut work,
        )
        .await
        .map(|_| ())
    }

    #[cfg(test)]
    pub(in crate::builtins::math::linalg) fn order(&self) -> usize {
        self.order
    }
}
