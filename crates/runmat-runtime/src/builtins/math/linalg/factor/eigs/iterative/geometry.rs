//! Complete overlapping recurrence/projection/output geometry, not an n²
//! coefficient estimate. Sparse factor fill is admitted by its own owner.
use super::scalar::Scalar;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostTypedWorkspace,
};
use num_complex::Complex64;
#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub n: usize,
    pub p: usize,
    pub k: usize,
    pub native: usize,
    pub complex: usize,
    pub indices: usize,
}
impl Geometry {
    pub fn checked(n: usize, p: usize, k: usize) -> Option<Self> {
        Some(Self {
            n,
            p,
            k,
            native: p.checked_mul(2)?.checked_add(4)?.checked_mul(n)?,
            complex: n
                .checked_mul(k)?
                .checked_add(k)?
                .checked_add(n.checked_mul(2)?)?,
            indices: p.checked_add(k.checked_mul(2)?)?,
        })
    }
    pub fn quote<T: Scalar>(
        self,
        requested: Option<usize>,
        kind: super::projected::Kind,
    ) -> Result<HostAllocationPlan, HostAllocationError> {
        let mut scratch = HostAllocationPlan::numeric_workspace_bytes(
            T::DTYPE,
            self.native
                .checked_mul(T::LANES)
                .ok_or(HostAllocationError::Overflow)?,
        )?;
        for bytes in [
            HostTypedWorkspace::<Complex64>::quoted_bytes(self.complex)?,
            HostTypedWorkspace::<usize>::quoted_bytes(self.indices)?,
            if self.p == 0 {
                0
            } else {
                super::projected::bytes(self.p, kind)?
            },
        ] {
            scratch = scratch
                .checked_add(bytes)
                .ok_or(HostAllocationError::Overflow)?;
        }
        let mut output = 0u64;
        // Publication compacts real-valued results. Its numeric-storage owner
        // metadata can exceed the complex owner's, especially for empty arrays;
        // neither representation universally bounds the other.
        let retained = |shape: &[usize]| -> Result<u64, HostAllocationError> {
            Ok(
                HostAllocationPlan::retained_tensor_bytes(T::DTYPE, shape)?.max(
                    HostAllocationPlan::retained_complex_tensor_bytes(T::DTYPE, shape)?,
                ),
            )
        };
        if requested != Some(0) {
            if requested.is_some_and(|n| n >= 2) {
                output = retained(&[self.n, self.k])?
                    .checked_add(retained(&[self.k, self.k])?)
                    .ok_or(HostAllocationError::Overflow)?;
            } else {
                output = retained(&[self.k, 1])?;
            }
        }
        if let Some(n @ 1..=3) = requested {
            output = output
                .checked_add(
                    crate::builtins::math::linalg::factor::iterative::output::retained_list_bytes(
                        n,
                    )?,
                )
                .ok_or(HostAllocationError::Overflow)?;
        }
        HostAllocationPlan::new(scratch, output)
    }
}
