//! Exact, invocation-owned scratch for the standard real Riccati QZ path.
//!
//! It owns the physical blocks consumed by the existing low-level faer
//! algorithm, not a second solver.

use faer::col::ColMut;
use faer::dyn_stack::StackReq;
use faer::linalg::{householder, qr::no_pivoting::factor as qr};
use faer::mat::MatMut;
use faer::{c64, Par, Spec};
use std::mem::MaybeUninit;

use crate::context::host_allocation::{
    AlignedHostWorkspace, HostAllocationError, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;

use super::super::algorithm::{complex_qz_in_place, complex_qz_scratch, ComplexQzView};

/// Checked shapes and faer's own scratch requirements. Every element count
/// corresponds to an existing allocation in `riccati::solve_owned`.
pub(crate) struct RiccatiWorkspacePlan {
    pub(super) n: usize,
    pub(super) m: usize,
    pub(super) doubled: usize,
    pub(super) order: usize,
    pub(super) pencil_len: usize,
    pub(super) reduced_len: usize,
    pub(super) column_len: usize,
    pub(super) householder_len: usize,
    pub(super) subspace_len: usize,
    pub(super) qr_scratch: StackReq,
    pub(super) qz_scratch: StackReq,
}

impl RiccatiWorkspacePlan {
    pub(crate) fn new(n: usize, m: usize) -> Result<Self, HostAllocationError> {
        let doubled = n.checked_mul(2).ok_or(HostAllocationError::Overflow)?;
        let order = doubled
            .checked_add(m)
            .ok_or(HostAllocationError::Overflow)?;
        let pencil_len = square(order)?;
        let reduced_len = square(doubled)?;
        let column_len = order.checked_mul(m).ok_or(HostAllocationError::Overflow)?;
        let block = qr::recommended_block_size::<c64>(order, m);
        let householder_len = block.checked_mul(m).ok_or(HostAllocationError::Overflow)?;
        let subspace_len = square(n)?;
        let params: Spec<qr::QrParams, c64> = Default::default();
        let qr_scratch = StackReq::any_of(&[
            qr::qr_in_place_scratch::<c64>(order, m, block, Par::Seq, params),
            householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<
                c64,
            >(order, block, order),
        ]);
        // The same physical QZ buffer is reused for the selected invariant
        // subspace and the independent eigenvalues of A-BK. Ask faer for both
        // actual requirements; do not assume monotonicity in matrix order.
        let qz_scratch = StackReq::any_of(&[complex_qz_scratch(doubled), complex_qz_scratch(n)]);
        qr_scratch
            .layout()
            .map_err(|_| HostAllocationError::Overflow)?;
        qz_scratch
            .layout()
            .map_err(|_| HostAllocationError::Overflow)?;
        Ok(Self {
            n,
            m,
            doubled,
            order,
            pencil_len,
            reduced_len,
            column_len,
            householder_len,
            subspace_len,
            qr_scratch,
            qz_scratch,
        })
    }

    /// The first connected cut preallocates all blocks before transferring
    /// work to the blocking worker. Thus the complete sum, not an optimistic
    /// phase maximum, is the simultaneous scratch charge. Input and final
    /// output owners are quoted separately by the control caller.
    pub(crate) fn scratch_bytes(&self) -> Result<u64, HostAllocationError> {
        let complex_lengths = [
            self.pencil_len,
            self.pencil_len,
            self.column_len,
            self.householder_len,
            self.reduced_len,
            self.reduced_len,
            self.doubled,
            self.doubled,
            self.subspace_len,
            self.subspace_len,
        ];
        let mut total = 0_u64;
        for len in complex_lengths {
            total = total
                .checked_add(HostTypedWorkspace::<c64>::quoted_bytes(len)?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        total = total
            .checked_add(HostTypedWorkspace::<u64>::quoted_bytes(self.doubled)?)
            .ok_or(HostAllocationError::Overflow)?;
        total = total
            .checked_add(HostTypedWorkspace::<f64>::quoted_bytes(self.subspace_len)?)
            .ok_or(HostAllocationError::Overflow)?;
        total = total
            .checked_add(HostTypedWorkspace::<f64>::quoted_bytes(self.n)?)
            .ok_or(HostAllocationError::Overflow)?;
        total = total
            .checked_add(HostTypedWorkspace::<c64>::quoted_bytes(self.n)?)
            .ok_or(HostAllocationError::Overflow)?;
        for request in [self.qr_scratch, self.qz_scratch] {
            total = total
                .checked_add(AlignedHostWorkspace::quoted_bytes(
                    request
                        .layout()
                        .map_err(|_| HostAllocationError::Overflow)?,
                )?)
                .ok_or(HostAllocationError::Overflow)?;
        }
        Ok(total)
    }
}

fn square(n: usize) -> Result<usize, HostAllocationError> {
    n.checked_mul(n).ok_or(HostAllocationError::Overflow)
}

/// The existing Riccati algorithm's scratch buffers, prebuilt before an
/// installed worker can outlive its awaiting caller. No buffer is copied into
/// faer: the algorithm forms `MatMut`/`ColMut` views over these same slices.
/// After deflation, the now-dead H/J pencil blocks provide the left/right QZ
/// factors through their `[..reduced_len]` prefixes instead of allocating two
/// additional matrices.
pub(crate) struct ManagedRiccatiWorkspace {
    pub(super) h: HostTypedWorkspace<c64>,
    pub(super) j: HostTypedWorkspace<c64>,
    pub(super) columns: HostTypedWorkspace<c64>,
    pub(super) householder: HostTypedWorkspace<c64>,
    pub(super) aa: HostTypedWorkspace<c64>,
    pub(super) bb: HostTypedWorkspace<c64>,
    pub(super) alpha: HostTypedWorkspace<c64>,
    pub(super) beta: HostTypedWorkspace<c64>,
    pub(super) lu: HostTypedWorkspace<c64>,
    pub(super) rhs: HostTypedWorkspace<c64>,
    pub(super) labels: HostTypedWorkspace<u64>,
    pub(super) solution: HostTypedWorkspace<f64>,
    pub(super) state_scales: HostTypedWorkspace<f64>,
    pub(super) poles: HostTypedWorkspace<c64>,
    pub(super) qr_scratch: AlignedHostWorkspace,
    pub(super) qz_scratch: AlignedHostWorkspace,
}

/// One borrowed numerical body shared by ordinary Vec-backed execution and
/// exactly claimed Managed execution. H/J also become the QZ factor storage
/// after deflation, when their original pencil contents are dead.
pub(super) struct RiccatiScratch<'a> {
    pub(super) h: &'a mut [c64],
    pub(super) j: &'a mut [c64],
    pub(super) columns: &'a mut [c64],
    pub(super) householder: &'a mut [c64],
    pub(super) aa: &'a mut [c64],
    pub(super) bb: &'a mut [c64],
    pub(super) alpha: &'a mut [c64],
    pub(super) beta: &'a mut [c64],
    pub(super) lu: &'a mut [c64],
    pub(super) rhs: &'a mut [c64],
    pub(super) labels: &'a mut [u64],
    pub(super) solution: &'a mut [f64],
    pub(super) state_scales: &'a mut [f64],
    pub(super) qr_scratch: &'a mut [MaybeUninit<u8>],
    pub(super) qz_scratch: &'a mut [MaybeUninit<u8>],
}

impl ManagedRiccatiWorkspace {
    pub(super) fn numerical_scratch(&mut self) -> RiccatiScratch<'_> {
        RiccatiScratch {
            h: self.h.as_mut_slice(),
            j: self.j.as_mut_slice(),
            columns: self.columns.as_mut_slice(),
            householder: self.householder.as_mut_slice(),
            aa: self.aa.as_mut_slice(),
            bb: self.bb.as_mut_slice(),
            alpha: self.alpha.as_mut_slice(),
            beta: self.beta.as_mut_slice(),
            lu: self.lu.as_mut_slice(),
            rhs: self.rhs.as_mut_slice(),
            labels: self.labels.as_mut_slice(),
            solution: self.solution.as_mut_slice(),
            state_scales: self.state_scales.as_mut_slice(),
            qr_scratch: self.qr_scratch.as_mut_uninit(),
            qz_scratch: self.qz_scratch.as_mut_uninit(),
        }
    }

    pub(crate) fn prepare(
        plan: &RiccatiWorkspacePlan,
        receipt: &HostAllocationReceipt,
        context: &RuntimeContext,
    ) -> Result<Self, HostAllocationError> {
        let complex = |len| receipt.temporary_typed::<c64>(context, len);
        let h = complex(plan.pencil_len)?;
        let j = complex(plan.pencil_len)?;
        let columns = complex(plan.column_len)?;
        let householder = complex(plan.householder_len)?;
        let aa = complex(plan.reduced_len)?;
        let bb = complex(plan.reduced_len)?;
        let alpha = complex(plan.doubled)?;
        let beta = complex(plan.doubled)?;
        let lu = complex(plan.subspace_len)?;
        let rhs = complex(plan.subspace_len)?;
        let labels = receipt.temporary_typed::<u64>(context, plan.doubled)?;
        let solution = receipt.temporary_typed::<f64>(context, plan.subspace_len)?;
        let state_scales = receipt.temporary_typed::<f64>(context, plan.n)?;
        let poles = complex(plan.n)?;
        let qr_scratch = AlignedHostWorkspace::new(
            plan.qr_scratch
                .layout()
                .map_err(|_| HostAllocationError::Overflow)?,
            receipt,
            context,
        )?;
        let qz_scratch = AlignedHostWorkspace::new(
            plan.qz_scratch
                .layout()
                .map_err(|_| HostAllocationError::Overflow)?,
            receipt,
            context,
        )?;
        receipt.checkpoint(context)?;
        Ok(Self {
            h,
            j,
            columns,
            householder,
            aa,
            bb,
            alpha,
            beta,
            lu,
            rhs,
            labels,
            solution,
            state_scales,
            poles,
            qr_scratch,
            qz_scratch,
        })
    }

    /// Reuse the dead Riccati pencil and reduced-factor buffers for a second,
    /// independent QZ of (A-BK, I). The resulting poles validate the actual
    /// feedback gain; ordered extended-pencil eigenvalues are not a substitute.
    pub(crate) fn closed_loop_poles_into(
        &mut self,
        plan: &RiccatiWorkspacePlan,
        closed_loop: &[f64],
    ) -> Result<&[c64], &'static str> {
        if plan.n == 0 || closed_loop.len() != plan.subspace_len {
            return Err("closed-loop matrix dimensions are invalid");
        }
        let n = plan.n;
        let aa = &mut self.aa.as_mut_slice()[..plan.subspace_len];
        let bb = &mut self.bb.as_mut_slice()[..plan.subspace_len];
        let left = &mut self.h.as_mut_slice()[..plan.subspace_len];
        let right = &mut self.j.as_mut_slice()[..plan.subspace_len];
        for (index, value) in closed_loop.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err("closed-loop matrix is nonfinite");
            }
            aa[index] = c64::new(value, 0.0);
            bb[index] = c64::new(if index / n == index % n { 1.0 } else { 0.0 }, 0.0);
        }
        let mut form = ComplexQzView {
            aa: MatMut::from_column_major_slice_mut(aa, n, n),
            bb: MatMut::from_column_major_slice_mut(bb, n, n),
            left: MatMut::from_column_major_slice_mut(left, n, n),
            right: MatMut::from_column_major_slice_mut(right, n, n),
            alpha: ColMut::from_slice_mut(&mut self.alpha.as_mut_slice()[..n]),
            beta: ColMut::from_slice_mut(&mut self.beta.as_mut_slice()[..n]),
        };
        complex_qz_in_place(&mut form, self.qz_scratch.as_mut_uninit())?;
        for index in 0..n {
            let (real, imag) = super::super::eigvals::ratio(form.alpha[index], form.beta[index]);
            let pole = c64::new(real, imag);
            if !pole.re.is_finite() || !pole.im.is_finite() {
                return Err("closed-loop pole is nonfinite");
            }
            self.poles.as_mut_slice()[index] = pole;
        }
        Ok(self.poles.as_slice())
    }

    pub(crate) fn solution(&self) -> &[f64] {
        self.solution.as_slice()
    }

    pub(crate) fn poles(&self) -> &[c64] {
        self.poles.as_slice()
    }
}
