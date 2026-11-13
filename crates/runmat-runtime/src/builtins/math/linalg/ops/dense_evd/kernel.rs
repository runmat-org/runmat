//! The pinned faer low-level EVD shared by full and projected problems.
//! All numerical arrays and MemStack backing are borrowed from their owner.
use faer::{
    diag::DiagMut,
    dyn_stack::{MemStack, StackReq},
    linalg::evd::{self, ComputeEigenvectors, EvdParams},
    mat::{MatMut, MatRef},
    Par, Spec,
};
use num_complex::Complex64;
use std::mem::MaybeUninit;

use super::EvdFailure;

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    General,
    SelfAdjoint,
}

pub(crate) fn requirement(order: usize, kind: Kind, left: bool) -> StackReq {
    if order == 0 {
        return StackReq::EMPTY;
    }
    match kind {
        Kind::SelfAdjoint => evd::self_adjoint_evd_scratch::<Complex64>(
            order,
            ComputeEigenvectors::Yes,
            Par::Seq,
            Default::default(),
        ),
        Kind::General => evd::evd_scratch::<Complex64>(
            order,
            if left {
                ComputeEigenvectors::Yes
            } else {
                ComputeEigenvectors::No
            },
            ComputeEigenvectors::Yes,
            Par::Seq,
            Spec::<EvdParams, Complex64>::default(),
        ),
    }
}

pub(crate) fn compute(
    order: usize,
    kind: Kind,
    matrix: &[Complex64],
    values: &mut [Complex64],
    right: &mut [Complex64],
    left: Option<&mut [Complex64]>,
    scratch: &mut [MaybeUninit<u8>],
) -> Result<(), EvdFailure> {
    if order == 0 {
        return Ok(());
    }
    let matrix = MatRef::from_column_major_slice(matrix, order, order);
    let values = DiagMut::from_slice_mut(values);
    let right = MatMut::from_column_major_slice_mut(right, order, order);
    let left = left.map(|values| MatMut::from_column_major_slice_mut(values, order, order));
    match kind {
        Kind::SelfAdjoint => evd::self_adjoint_evd(
            matrix,
            values,
            Some(right),
            Par::Seq,
            MemStack::new(scratch),
            Default::default(),
        ),
        Kind::General => evd::evd_cplx(
            matrix,
            values,
            left,
            Some(right),
            Par::Seq,
            MemStack::new(scratch),
            Default::default(),
        ),
    }
    .map_err(EvdFailure)
}
