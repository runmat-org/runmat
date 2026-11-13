//! Claimed weight checks with the same CARE and regulator contracts as the
//! ordinary Riccati path. Faer's low-level self-adjoint eigensolver borrows
//! only the explicitly admitted buffers and aligned scratch below.

use faer::diag::DiagMut;
use faer::dyn_stack::MemStack;
use faer::dyn_stack::StackReq;
use faer::linalg::cholesky::llt::factor as llt;
use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
use faer::mat::{MatMut, MatRef};
use faer::{Par, Spec};

use super::{
    weights::{WeightContract, SCHUR_NEGATIVE_TOLERANCE},
    RiccatiApi, RiccatiIssue,
};
use crate::context::host_allocation::{
    AlignedHostWorkspace, HostAllocationError, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

pub(super) struct ManagedWeightPlan {
    n: usize,
    m: usize,
    api: RiccatiApi,
}

impl ManagedWeightPlan {
    pub(super) fn new(n: usize, m: usize, api: RiccatiApi) -> Result<Self, HostAllocationError> {
        n.checked_mul(n).ok_or(HostAllocationError::Overflow)?;
        n.checked_mul(m).ok_or(HostAllocationError::Overflow)?;
        m.checked_mul(m).ok_or(HostAllocationError::Overflow)?;
        Ok(Self { n, m, api })
    }

    fn numerical_layout(&self) -> Result<std::alloc::Layout, HostAllocationError> {
        let evd_params: Spec<faer::linalg::evd::SelfAdjointEvdParams, f64> = Default::default();
        let llt_params: Spec<llt::LltParams, f64> = Default::default();
        StackReq::any_of(&[
            self_adjoint_evd_scratch::<f64>(self.n, ComputeEigenvectors::No, Par::Seq, evd_params),
            llt::cholesky_in_place_scratch::<f64>(self.m, Par::Seq, llt_params),
        ])
        .layout()
        .map_err(|_| HostAllocationError::Overflow)
    }

    pub(super) fn scratch_bytes(&self) -> Result<u64, HostAllocationError> {
        let mut bytes = HostTypedWorkspace::<f64>::quoted_bytes(self.m * self.m)?;
        if WeightContract::for_api(self.api) == WeightContract::PositiveCost {
            for quote in [
                HostTypedWorkspace::<f64>::quoted_bytes(self.m * self.n)?,
                HostTypedWorkspace::<f64>::quoted_bytes(self.n * self.n)?,
                HostTypedWorkspace::<f64>::quoted_bytes(self.n)?,
                AlignedHostWorkspace::quoted_bytes(self.numerical_layout()?)?,
            ] {
                bytes = bytes
                    .checked_add(quote)
                    .ok_or(HostAllocationError::Overflow)?;
            }
        }
        Ok(bytes)
    }

    pub(super) fn validate(
        &self,
        q: &[f64],
        r: &[f64],
        cross: &[f64],
        receipt: &HostAllocationReceipt,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<()> {
        let (n, m, api) = (self.n, self.m, self.api);
        if q.len() != n * n || r.len() != m * m || cross.len() != n * m {
            return Err(api.error(
                RiccatiIssue::InvalidDimensions,
                "Riccati weight dimensions changed",
            ));
        }
        if !q
            .iter()
            .chain(r)
            .chain(cross)
            .all(|value| value.is_finite())
        {
            return Err(api.error(
                RiccatiIssue::InvalidDimensions,
                "Riccati weights are nonfinite after normalization",
            ));
        }
        let mut factor = receipt
            .temporary_typed::<f64>(runtime, m * m)
            .map_err(|error| api.admission(Some(runtime), error))?;
        for (target, source) in factor.as_mut_slice().chunks_mut(4096).zip(r.chunks(4096)) {
            receipt
                .checkpoint(runtime)
                .map_err(|error| api.admission(Some(runtime), error))?;
            target.copy_from_slice(source);
        }
        if WeightContract::for_api(api) == WeightContract::SymmetricNonsingular {
            return validate_nonsingular(factor.as_mut_slice(), m, receipt, runtime, api);
        }
        let mut ticker = Ticker::new(receipt, runtime, api);
        // Ordinary regulator validation requires R positive definite.
        // Faer writes its lower Cholesky factor into the same claimed matrix.
        let mut scratch = AlignedHostWorkspace::new(
            self.numerical_layout()
                .map_err(|error| api.admission(Some(runtime), error))?,
            receipt,
            runtime,
        )
        .map_err(|error| api.admission(Some(runtime), error))?;
        let llt_params: Spec<llt::LltParams, f64> = Default::default();
        ticker.finish()?;
        llt::cholesky_in_place::<f64>(
            MatMut::from_column_major_slice_mut(factor.as_mut_slice(), m, m),
            llt::LltRegularization::default(),
            Par::Seq,
            MemStack::new(scratch.as_mut_uninit()),
            llt_params,
        )
        .map_err(|_| {
            api.error(
                RiccatiIssue::InvalidDimensions,
                "R must be symmetric positive definite",
            )
        })?;
        ticker.finish()?;
        // Solve R*X=N' through L and L' into one exact m-by-n workspace.
        let mut solution = receipt
            .temporary_typed::<f64>(runtime, m * n)
            .map_err(|error| api.admission(Some(runtime), error))?;
        for col in 0..n {
            for row in 0..m {
                let mut value = cross[col_major(col, row, n)];
                for k in 0..row {
                    ticker.tick()?;
                    value -= factor.as_slice()[col_major(row, k, m)]
                        * solution.as_slice()[col_major(k, col, m)];
                }
                solution.as_mut_slice()[col_major(row, col, m)] =
                    value / factor.as_slice()[col_major(row, row, m)];
            }
            for row in (0..m).rev() {
                let mut value = solution.as_slice()[col_major(row, col, m)];
                for k in (row + 1)..m {
                    ticker.tick()?;
                    value -= factor.as_slice()[col_major(k, row, m)]
                        * solution.as_slice()[col_major(k, col, m)];
                }
                solution.as_mut_slice()[col_major(row, col, m)] =
                    value / factor.as_slice()[col_major(row, row, m)];
            }
        }
        let mut schur = receipt
            .temporary_typed::<f64>(runtime, n * n)
            .map_err(|error| api.admission(Some(runtime), error))?;
        let mut scale = 0.0_f64;
        for col in 0..n {
            for row in 0..n {
                let mut value = q[col_major(row, col, n)];
                for k in 0..m {
                    ticker.tick()?;
                    value -=
                        cross[col_major(row, k, n)] * solution.as_slice()[col_major(k, col, m)];
                }
                if !value.is_finite() {
                    return Err(api.error(
                        RiccatiIssue::InvalidDimensions,
                        "Riccati Schur weight is nonfinite",
                    ));
                }
                schur.as_mut_slice()[col_major(row, col, n)] = value;
            }
        }
        for col in 0..n {
            for row in (col + 1)..n {
                let value = 0.5
                    * (schur.as_slice()[col_major(row, col, n)]
                        + schur.as_slice()[col_major(col, row, n)]);
                schur.as_mut_slice()[col_major(row, col, n)] = value;
                schur.as_mut_slice()[col_major(col, row, n)] = value;
            }
        }
        for value in schur.as_slice() {
            scale = scale.max(value.abs());
        }
        scale = scale.max(1.0);
        let mut eigenvalues = receipt
            .temporary_typed::<f64>(runtime, n)
            .map_err(|error| api.admission(Some(runtime), error))?;
        ticker.finish()?;
        let params: Spec<faer::linalg::evd::SelfAdjointEvdParams, f64> = Default::default();
        self_adjoint_evd::<f64>(
            MatRef::from_column_major_slice(schur.as_slice(), n, n),
            DiagMut::from_slice_mut(eigenvalues.as_mut_slice()),
            None,
            Par::Seq,
            MemStack::new(scratch.as_mut_uninit()),
            params,
        )
        .map_err(|_| {
            api.error(
                RiccatiIssue::InvalidDimensions,
                "Riccati Schur weight eigenanalysis failed",
            )
        })?;
        ticker.finish()?;
        if eigenvalues
            .as_slice()
            .iter()
            .any(|value| !value.is_finite() || *value < -SCHUR_NEGATIVE_TOLERANCE * scale)
        {
            return Err(api.error(
                RiccatiIssue::InvalidDimensions,
                "[Q N; N' R] must be positive semidefinite",
            ));
        }
        Ok(())
    }
}

fn validate_nonsingular(
    factor: &mut [f64],
    m: usize,
    receipt: &HostAllocationReceipt,
    runtime: &RuntimeContext,
    api: RiccatiApi,
) -> BuiltinResult<()> {
    let mut ticker = Ticker::new(receipt, runtime, api);
    let scale = factor
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return Err(api.error(
            RiccatiIssue::InvalidDimensions,
            "R must be nonsingular for the standard CARE equation",
        ));
    }
    for k in 0..m {
        let pivot = (k..m)
            .max_by(|&left, &right| {
                factor[col_major(left, k, m)]
                    .abs()
                    .total_cmp(&factor[col_major(right, k, m)].abs())
            })
            .ok_or_else(|| api.error(RiccatiIssue::InvalidDimensions, "R is empty"))?;
        if factor[col_major(pivot, k, m)] == 0.0 || !factor[col_major(pivot, k, m)].is_finite() {
            return Err(api.error(
                RiccatiIssue::InvalidDimensions,
                "R must be nonsingular for the standard CARE equation",
            ));
        }
        if pivot != k {
            for col in 0..m {
                factor.swap(col_major(k, col, m), col_major(pivot, col, m));
            }
        }
        for row in (k + 1)..m {
            ticker.tick()?;
            let multiplier = factor[col_major(row, k, m)] / factor[col_major(k, k, m)];
            if !multiplier.is_finite() {
                return Err(api.error(
                    RiccatiIssue::InvalidDimensions,
                    "R must be nonsingular for the standard CARE equation",
                ));
            }
            for col in (k + 1)..m {
                factor[col_major(row, col, m)] -= multiplier * factor[col_major(k, col, m)];
            }
        }
    }
    ticker.finish()
}

fn col_major(row: usize, col: usize, rows: usize) -> usize {
    col * rows + row
}

struct Ticker<'a> {
    receipt: &'a HostAllocationReceipt,
    runtime: &'a RuntimeContext,
    api: RiccatiApi,
    count: usize,
}

impl<'a> Ticker<'a> {
    fn new(
        receipt: &'a HostAllocationReceipt,
        runtime: &'a RuntimeContext,
        api: RiccatiApi,
    ) -> Self {
        Self {
            receipt,
            runtime,
            api,
            count: 0,
        }
    }

    fn tick(&mut self) -> BuiltinResult<()> {
        self.count += 1;
        if self.count == 4096 {
            self.count = 0;
            self.finish()?;
        }
        Ok(())
    }

    fn finish(&self) -> BuiltinResult<()> {
        self.receipt
            .checkpoint(self.runtime)
            .map_err(|error| self.api.admission(Some(self.runtime), error))
    }
}
