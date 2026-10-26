//! Shared original-matrix PSD certificate, independent of any optimizer kernel.
//! The caller supplies a finite symmetric matrix in upper-triangular CSC.
use super::super::ScalarCooperation;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use clarabel::algebra::CscMatrix;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::collections::BTreeMap;
type Matrix = CscMatrix<f64>;
mod exact;
pub(in crate::builtins::math::optim) use exact::{
    certify_original, certify_restricted, OriginalHessian,
};
mod floating;

pub(crate) async fn certify(
    p: &Matrix,
    owner: runmat_builtins::BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    // Gershgorin: symmetric weak diagonal dominance with nonnegative diagonal
    // is PSD. Exact sums make equality and subnormal cases sound and cheap.
    let mut diagonal = vec![0.0; p.n];
    let mut off = vec![BigRational::zero(); p.n];
    for col in 0..p.n {
        for entry in p.colptr[col]..p.colptr[col + 1] {
            let row = p.rowval[entry];
            let value = p.nzval[entry];
            if row == col {
                diagonal[col] = value;
            } else {
                let term = BigRational::from_float(value)
                    .expect("finite normalized H")
                    .abs();
                off[row] += &term;
                off[col] += term;
            }
            if entry % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    let mut dominant = true;
    for (index, (diagonal, off)) in diagonal.into_iter().zip(off).enumerate() {
        if diagonal < 0.0 {
            return Ok(false);
        }
        dominant &= BigRational::from_float(diagonal).expect("finite diagonal") >= off;
        if index % 1024 == 1023 {
            cooperation.completed(1024).await?;
        }
    }
    if dominant || floating::certify(p, cooperation, owner).await? {
        return Ok(true);
    }
    // Numerical factor failure does not prove indefiniteness: semidefinite and
    // ambiguous matrices receive an exact sparse Schur-complement decision.
    exact::certify(p, owner, cooperation).await
}
