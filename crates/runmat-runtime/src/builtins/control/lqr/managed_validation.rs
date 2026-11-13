//! Claimed gain and original-equation publication checks for managed Riccati.
//!
//! Matrix slices are column-major. This leaf consumes the same solution as the
//! ordered-QZ worker; it does not solve a second Riccati problem. The closed-
//! loop eigenvalues are computed by that worker from `closed_loop`, not reused
//! from the transformed pencil.

use num_complex::Complex64;

use super::{validation, LqrDomain, RiccatiApi, RiccatiIssue};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

const CHECKPOINT_OPERATIONS: usize = 4096;

pub(super) struct ManagedValidationPlan {
    n: usize,
    m: usize,
}

impl ManagedValidationPlan {
    pub(super) fn new(n: usize, m: usize) -> Result<Self, HostAllocationError> {
        n.checked_mul(n).ok_or(HostAllocationError::Overflow)?;
        n.checked_mul(m).ok_or(HostAllocationError::Overflow)?;
        m.checked_mul(m).ok_or(HostAllocationError::Overflow)?;
        Ok(Self { n, m })
    }

    pub(super) fn scratch_bytes(&self) -> Result<u64, HostAllocationError> {
        let nm = self.n * self.m;
        let mut bytes = 0_u64;
        for quote in [
            HostTypedWorkspace::<f64>::quoted_bytes(nm)?, // gain
            HostTypedWorkspace::<f64>::quoted_bytes(self.m * self.m)?, // denominator
            HostTypedWorkspace::<f64>::quoted_bytes(nm)?, // S*B / control
            HostTypedWorkspace::<f64>::quoted_bytes(self.n * self.n)?, // S*A / A-B*K
        ] {
            bytes = bytes
                .checked_add(quote)
                .ok_or(HostAllocationError::Overflow)?;
        }
        Ok(bytes)
    }
}

pub(super) struct ManagedValidationResult {
    gain: HostTypedWorkspace<f64>,
    closed_loop: HostTypedWorkspace<f64>,
    n: usize,
    m: usize,
}

pub(super) struct RiccatiMatrixSlices<'a> {
    pub(super) a: &'a [f64],
    pub(super) b: &'a [f64],
    pub(super) q: &'a [f64],
    pub(super) r: &'a [f64],
    pub(super) cross: &'a [f64],
    pub(super) solution: &'a [f64],
}

impl ManagedValidationResult {
    pub(super) fn into_parts(
        self,
    ) -> (
        HostTypedWorkspace<f64>,
        HostTypedWorkspace<f64>,
        usize,
        usize,
    ) {
        (self.gain, self.closed_loop, self.n, self.m)
    }

    pub(super) fn from_parts(
        gain: HostTypedWorkspace<f64>,
        closed_loop: HostTypedWorkspace<f64>,
        n: usize,
        m: usize,
    ) -> Self {
        Self {
            gain,
            closed_loop,
            n,
            m,
        }
    }

    pub(super) fn prepare(
        plan: &ManagedValidationPlan,
        inputs: RiccatiMatrixSlices<'_>,
        domain: LqrDomain,
        receipt: &HostAllocationReceipt,
        runtime: &RuntimeContext,
        api: RiccatiApi,
    ) -> BuiltinResult<Self> {
        let n = plan.n;
        let m = plan.m;
        if inputs.a.len() != n * n
            || inputs.b.len() != n * m
            || inputs.q.len() != n * n
            || inputs.r.len() != m * m
            || inputs.cross.len() != n * m
            || inputs.solution.len() != n * n
        {
            return Err(failed(
                api,
                "Riccati matrix dimensions changed before validation",
            ));
        }
        let mut gain = claimed::<f64>(receipt, runtime, m * n, api)?;
        let mut denominator = claimed::<f64>(receipt, runtime, m * m, api)?;
        let mut control = claimed::<f64>(receipt, runtime, n * m, api)?;
        let mut closed_loop = claimed::<f64>(receipt, runtime, n * n, api)?;
        let (a, b, q, r, cross, s) = (
            inputs.a,
            inputs.b,
            inputs.q,
            inputs.r,
            inputs.cross,
            inputs.solution,
        );
        let mut ticker = Ticker::new(receipt, runtime, api);
        for col in 0..m {
            for row in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    ticker.tick()?;
                    sum += s[col_major(row, k, n)] * b[col_major(k, col, n)];
                }
                control.as_mut_slice()[col_major(row, col, n)] = sum;
            }
        }
        for col in 0..m {
            for row in 0..m {
                let mut sum = r[col_major(row, col, m)];
                if matches!(domain, LqrDomain::Discrete) {
                    for k in 0..n {
                        ticker.tick()?;
                        sum += b[col_major(k, row, n)] * control.as_slice()[col_major(k, col, n)];
                    }
                }
                denominator.as_mut_slice()[col_major(row, col, m)] = sum;
            }
        }
        for col in 0..n {
            for row in 0..m {
                let mut sum = cross[col_major(col, row, n)];
                match domain {
                    LqrDomain::Continuous => {
                        sum += control.as_slice()[col_major(col, row, n)];
                    }
                    LqrDomain::Discrete => {
                        for k in 0..n {
                            ticker.tick()?;
                            sum +=
                                a[col_major(k, col, n)] * control.as_slice()[col_major(k, row, n)];
                        }
                    }
                }
                gain.as_mut_slice()[col_major(row, col, m)] = sum;
            }
        }
        // The numerator transpose is also the control term of the original
        // equation. Retain it before the in-place pivoted solve modifies K.
        for col in 0..m {
            for row in 0..n {
                control.as_mut_slice()[col_major(row, col, n)] =
                    gain.as_slice()[col_major(col, row, m)];
            }
        }
        super::gain::solve_in_place(
            denominator.as_mut_slice(),
            gain.as_mut_slice(),
            m,
            n,
            api,
            || ticker.tick(),
        )?;
        if matches!(domain, LqrDomain::Discrete) {
            for col in 0..n {
                for row in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        ticker.tick()?;
                        sum += s[col_major(row, k, n)] * a[col_major(k, col, n)];
                    }
                    closed_loop.as_mut_slice()[col_major(row, col, n)] = sum;
                }
            }
        }
        let mut residual_max = 0.0_f64;
        let mut scale = 0.0_f64;
        for col in 0..n {
            for row in 0..n {
                let mut left = 0.0;
                let mut right = 0.0;
                let mut control_term = 0.0;
                for k in 0..n {
                    ticker.tick()?;
                    if matches!(domain, LqrDomain::Continuous) {
                        left += a[col_major(k, row, n)] * s[col_major(k, col, n)];
                    }
                    right += match domain {
                        LqrDomain::Continuous => s[col_major(row, k, n)] * a[col_major(k, col, n)],
                        LqrDomain::Discrete => {
                            a[col_major(k, row, n)] * closed_loop.as_slice()[col_major(k, col, n)]
                        }
                    };
                }
                for k in 0..m {
                    ticker.tick()?;
                    control_term += control.as_slice()[col_major(row, k, n)]
                        * gain.as_slice()[col_major(k, col, m)];
                }
                let rhs = match domain {
                    LqrDomain::Continuous => {
                        left + right - control_term + q[col_major(row, col, n)]
                    }
                    LqrDomain::Discrete => {
                        right - s[col_major(row, col, n)] - control_term + q[col_major(row, col, n)]
                    }
                };
                if matches!(domain, LqrDomain::Continuous) {
                    scale = scale.max(left.abs());
                }
                scale = scale
                    .max(right.abs())
                    .max(control_term.abs())
                    .max(q[col_major(row, col, n)].abs());
                if !rhs.is_finite() {
                    return Err(failed(api, "Riccati residual is nonfinite"));
                }
                residual_max = residual_max.max(rhs.abs());
            }
        }
        if matches!(domain, LqrDomain::Discrete) {
            scale = scale.max(
                s.iter()
                    .fold(0.0_f64, |maximum, value| maximum.max(value.abs())),
            );
        }
        if !scale.is_finite() || residual_max > validation::RESIDUAL_TOLERANCE * scale {
            return Err(failed(api, "Riccati residual is too large"));
        }
        for col in 0..n {
            for row in 0..n {
                let mut value = a[col_major(row, col, n)];
                for k in 0..m {
                    ticker.tick()?;
                    value -= b[col_major(row, k, n)] * gain.as_slice()[col_major(k, col, m)];
                }
                if !value.is_finite() {
                    return Err(failed(api, "closed-loop matrix is nonfinite"));
                }
                closed_loop.as_mut_slice()[col_major(row, col, n)] = value;
            }
        }
        ticker.finish()?;
        Ok(Self {
            gain,
            closed_loop,
            n,
            m,
        })
    }

    pub(super) fn validate_poles(
        &self,
        poles: &[Complex64],
        domain: LqrDomain,
        api: RiccatiApi,
    ) -> BuiltinResult<()> {
        if poles.len() != self.n {
            return Err(failed(api, "closed-loop pole count changed"));
        }
        for pole in poles {
            if !validation::stable(*pole, domain) {
                return Err(failed(
                    api,
                    "Riccati solution does not stabilize the closed-loop system",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn gain(&self) -> &[f64] {
        self.gain.as_slice()
    }

    pub(super) fn gain_shape(&self) -> [usize; 2] {
        [self.m, self.n]
    }
}

fn claimed<T: crate::context::host_allocation::HostTypedElement>(
    receipt: &HostAllocationReceipt,
    runtime: &RuntimeContext,
    len: usize,
    api: RiccatiApi,
) -> BuiltinResult<HostTypedWorkspace<T>> {
    receipt
        .temporary_typed(runtime, len)
        .map_err(|error| api.admission(Some(runtime), error))
}

fn col_major(row: usize, col: usize, rows: usize) -> usize {
    col * rows + row
}

fn failed(api: RiccatiApi, message: &'static str) -> crate::RuntimeError {
    api.error(RiccatiIssue::SolverFailed, message)
}

struct Ticker<'a> {
    receipt: &'a HostAllocationReceipt,
    runtime: &'a RuntimeContext,
    api: RiccatiApi,
    operations: usize,
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
            operations: 0,
        }
    }

    fn tick(&mut self) -> BuiltinResult<()> {
        self.operations += 1;
        if self.operations == CHECKPOINT_OPERATIONS {
            self.operations = 0;
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
