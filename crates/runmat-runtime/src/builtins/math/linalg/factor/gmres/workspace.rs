//! Checked geometry of the complete live Arnoldi/least-squares workspace.
//! The caller quotes these exact counts against its selected native class
//! before allocating any array or invoking an operator.

use num_complex::Complex64;
use runmat_value::NumericDType;

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostTypedWorkspace,
};

use super::recurrence::GmresWork;
use super::scalar::GmresScalar;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GmresGeometry(runmat_sparse_solvers::krylov::gmres::GmresGeometry);

impl std::ops::Deref for GmresGeometry {
    type Target = runmat_sparse_solvers::krylov::gmres::GmresGeometry;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GmresGeometry {
    pub fn checked(n: usize, restart: usize, max_outer: usize) -> Option<Self> {
        runmat_sparse_solvers::krylov::gmres::GmresGeometry::checked(n, restart, max_outer)
            .map(Self)
    }

    /// Three simultaneous physical workspaces: the native vectors/basis, one
    /// complex64 Hessenberg/rotation array, and one f64 rotation/history
    /// array. Each quoted owner includes its own exact stage-claim metadata.
    pub fn scratch_bytes(
        self,
        dtype: NumericDType,
        native_lanes: usize,
    ) -> Result<u64, HostAllocationError> {
        let native_count = self
            .native_elements
            .checked_mul(native_lanes)
            .ok_or(HostAllocationError::Overflow)?;
        let complex_count = self
            .hessenberg_elements
            .checked_add(self.complex_small_elements)
            .ok_or(HostAllocationError::Overflow)?;
        let real_count = self
            .real_small_elements
            .checked_add(self.history_elements)
            .ok_or(HostAllocationError::Overflow)?;
        let native = HostAllocationPlan::numeric_workspace_bytes(dtype, native_count)?;
        let complex = HostTypedWorkspace::<Complex64>::quoted_bytes(complex_count)?;
        let real = HostTypedWorkspace::<f64>::quoted_bytes(real_count)?;
        native
            .checked_add(complex)
            .and_then(|bytes| bytes.checked_add(real))
            .ok_or(HostAllocationError::Overflow)
    }

    /// Project the three exact claimed backings into disjoint recurrence
    /// slices. No collection header, COW clone, or temporary vector appears
    /// at this handoff.
    pub fn project<'a, T: GmresScalar>(
        self,
        native: &'a mut [T],
        complex: &'a mut [Complex64],
        real: &'a mut [f64],
    ) -> Option<GmresWork<'a, T>> {
        self.0.project(native, complex, real)
    }
}
